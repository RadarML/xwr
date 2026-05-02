/*
 * _fast.c — high-speed DCA1000EVM UDP frame receiver
 *
 * Exposes FrameStream, which uses poll(2) + recvmmsg(2) to batch-receive UDP
 * packets into a pre-allocated C ring buffer and assembles them into radar
 * frames.  Completed frames are returned as zero-copy numpy uint8 arrays that
 * view directly into the ring buffer.  The ring has `ring_frames` slots; a
 * slot is safe to read until ring_frames subsequent calls to next_frame().
 *
 * Linux-only (recvmmsg, poll, CLOCK_REALTIME, CLOCK_MONOTONIC).
 *
 * Design notes
 * ------------
 * The DCA1000EVM data socket is kept in non-blocking mode (O_NONBLOCK) by the
 * Python layer.  recvmmsg(2) on a non-blocking socket returns EAGAIN
 * immediately when the buffer is empty, causing a busy-spin.  We therefore
 * use poll(2) to block until data arrives, then recvmmsg(2) with MSG_DONTWAIT
 * to drain all immediately available packets in one shot.
 *
 * When a packet in a batch completes a frame we return immediately; the batch
 * position is saved so the next next_frame() call resumes from the following
 * packet before issuing a new recvmmsg.
 */

#define _GNU_SOURCE
#include <Python.h>
#include <numpy/arrayobject.h>

#include <errno.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <stdint.h>
#include <sys/uio.h>

#define FS_MAX_PKT  2048   /* receive buffer per packet (bytes) */
#define FS_HDR_LEN  10     /* 4-byte seqnum + 6-byte byte_count */


/* ================================================================== */
/* RingBuf — fixed ring of frame-sized slots                           */
/* ================================================================== */

typedef struct {
    uint8_t    *buf;        /* n * slot_size bytes, allocated once      */
    int         n;          /* number of slots                          */
    Py_ssize_t  slot_size;  /* bytes per slot (== frame_size)           */
    int         slot;       /* index of slot currently being filled     */
    Py_ssize_t  fill;       /* bytes written into the current slot      */
} RingBuf;

static int
ring_alloc(RingBuf *r, int n, Py_ssize_t slot_size)
{
    r->buf = (uint8_t *)malloc((size_t)n * (size_t)slot_size);
    if (!r->buf) return -1;
    r->n         = n;
    r->slot_size = slot_size;
    r->slot      = 0;
    r->fill      = 0;
    return 0;
}

static void
ring_free(RingBuf *r)
{
    free(r->buf);
    r->buf = NULL;
}

/* Base pointer of the slot currently being filled. */
static inline uint8_t *
ring_cur(const RingBuf *r)
{
    return r->buf + (size_t)r->slot * (size_t)r->slot_size;
}

/* Base pointer of any slot by index. */
static inline uint8_t *
ring_at(const RingBuf *r, int slot)
{
    return r->buf + (size_t)slot * (size_t)r->slot_size;
}

/* Advance to the next slot and reset the fill counter. */
static inline void
ring_next(RingBuf *r)
{
    r->slot = (r->slot + 1) % r->n;
    r->fill = 0;
}


/* ================================================================== */
/* PktBatch — recvmmsg pre-allocated receive buffers                   */
/* ================================================================== */

typedef struct {
    struct mmsghdr *hdrs;   /* recvmmsg message-header array            */
    struct iovec   *iovecs; /* iovec array; permanently wired into hdrs */
    uint8_t        *bufs;   /* cap * FS_MAX_PKT contiguous packet memory*/
    int             cap;    /* maximum packets per recvmmsg call        */
    int             pos;    /* next packet to process in current batch  */
    int             count;  /* number of valid packets in current batch */
} PktBatch;

static int
batch_alloc(PktBatch *b, int cap)
{
    b->hdrs   = (struct mmsghdr *)calloc((size_t)cap, sizeof(struct mmsghdr));
    b->iovecs = (struct iovec *)  malloc((size_t)cap * sizeof(struct iovec));
    b->bufs   = (uint8_t *)      malloc((size_t)cap * FS_MAX_PKT);
    if (!b->hdrs || !b->iovecs || !b->bufs) return -1;
    b->cap   = cap;
    b->pos   = 0;
    b->count = 0;
    for (int i = 0; i < cap; i++) {
        b->iovecs[i].iov_base          = b->bufs + (size_t)i * FS_MAX_PKT;
        b->iovecs[i].iov_len           = FS_MAX_PKT;
        b->hdrs[i].msg_hdr.msg_iov     = &b->iovecs[i];
        b->hdrs[i].msg_hdr.msg_iovlen  = 1;
    }
    return 0;
}

static void
batch_free(PktBatch *b)
{
    free(b->hdrs);   b->hdrs   = NULL;
    free(b->iovecs); b->iovecs = NULL;
    free(b->bufs);   b->bufs   = NULL;
}

static inline const uint8_t *
batch_data(const PktBatch *b, int i)
{
    return b->bufs + (size_t)i * FS_MAX_PKT;
}

static inline int
batch_len(const PktBatch *b, int i)
{
    return (int)b->hdrs[i].msg_len;
}


/* ================================================================== */
/* FrameStreamObject                                                    */
/* ================================================================== */

typedef struct {
    PyObject_HEAD

    /* configuration — immutable after __init__ */
    int         fd;
    double      timeout;

    /* pre-allocated ring and receive batch */
    RingBuf     ring;
    PktBatch    batch;

    /* frame assembly state */
    uint64_t    offset;       /* next expected byte_count from stream   */
    Py_ssize_t  drop_bytes;   /* zero-filled bytes accumulated in frame */
    double      frame_ts;     /* wall-clock stamp of frame's first byte */
    Py_ssize_t  pending_gap;  /* zero-fill bytes owed for multi-frame gaps */
} FrameStreamObject;


static void
FS_dealloc(FrameStreamObject *self)
{
    ring_free(&self->ring);
    batch_free(&self->batch);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
FS_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    FrameStreamObject *self = (FrameStreamObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->fd          = -1;
    self->timeout     = 1.0;
    memset(&self->ring,  0, sizeof(self->ring));
    memset(&self->batch, 0, sizeof(self->batch));
    self->offset      = 0;
    self->drop_bytes  = 0;
    self->frame_ts    = 0.0;
    self->pending_gap = 0;
    return (PyObject *)self;
}

static int
FS_init(FrameStreamObject *self, PyObject *args, PyObject *kwds)
{
    static char *kw[] = {
        "fd", "frame_size", "batch_size", "ring_frames", "timeout", NULL
    };
    int        fd, batch_cap = 64, ring_n = 4;
    Py_ssize_t frame_size;
    double     timeout = 1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "in|iid", kw,
                                     &fd, &frame_size, &batch_cap, &ring_n, &timeout))
        return -1;

    if (frame_size <= 0 || batch_cap < 1 || batch_cap > 1024 || ring_n < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid FrameStream parameter");
        return -1;
    }

    self->fd      = fd;
    self->timeout = timeout;

    if (ring_alloc(&self->ring, ring_n, frame_size) < 0) {
        PyErr_NoMemory(); return -1;
    }
    if (batch_alloc(&self->batch, batch_cap) < 0) {
        ring_free(&self->ring);
        PyErr_NoMemory(); return -1;
    }
    return 0;
}


/* ================================================================== */
/* Helpers — clocks, result builder, network I/O                       */
/* ================================================================== */

static inline double
mono_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double
wall_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*
 * Build the (timestamp, numpy_view, dropped_bytes) return tuple.
 *
 * The numpy array is a zero-copy uint8 view into ring slot `slot`.
 * Setting self as the array's base object keeps the ring alive for as
 * long as the array is referenced; the slot is safe to read for
 * ring_frames subsequent next_frame() calls.
 */
static PyObject *
make_result(FrameStreamObject *self, int slot, double ts, Py_ssize_t drop)
{
    uint8_t  *ptr  = ring_at(&self->ring, slot);
    npy_intp  dims = (npy_intp)self->ring.slot_size;
    PyObject *arr  = PyArray_SimpleNewFromData(1, &dims, NPY_UINT8, ptr);
    if (!arr) return NULL;
    Py_INCREF(self);
    if (PyArray_SetBaseObject((PyArrayObject *)arr, (PyObject *)self) < 0) {
        Py_DECREF(arr); Py_DECREF(self); return NULL;
    }
    PyObject *ret = Py_BuildValue("(dOn)", ts, arr, drop);
    Py_DECREF(arr);
    return ret;
}

/*
 * Wait up to `ms` milliseconds for `fd` to become readable.
 * Releases the GIL during the syscall.
 *
 * Returns: 1 = readable
 *          0 = timed out
 *         -1 = error (PyErr set)
 *         -2 = EINTR without a pending Python exception (caller should retry)
 */
static int
poll_readable(int fd, int ms)
{
    struct pollfd pfd = { .fd = fd, .events = POLLIN };
    int pret;
    Py_BEGIN_ALLOW_THREADS
    pret = poll(&pfd, 1, ms);
    Py_END_ALLOW_THREADS
    /* Check for KeyboardInterrupt etc. before inspecting errno. */
    if (PyErr_CheckSignals() != 0) return -1;
    if (pret < 0) {
        if (errno == EINTR) return -2;
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    return pret > 0 ? 1 : 0;
}

/*
 * Wait for data and fill self->batch with up to batch.cap packets.
 *
 * Returns:  1 = batch ready (batch.pos reset to 0)
 *           0 = deadline expired (caller should return None)
 *          -1 = error (PyErr set)
 *          -2 = transient (EINTR or poll/recvmmsg returned nothing; retry)
 */
static int
do_recv(FrameStreamObject *self, double *deadline)
{
    double remaining = *deadline - mono_now();
    if (remaining <= 0.0) return 0;

    int poll_ms = (int)(remaining * 1000.0);
    if (poll_ms <= 0) poll_ms = 1;

    int ready = poll_readable(self->fd, poll_ms);
    if (ready == -1) return -1;   /* error */
    if (ready !=  1) return -2;   /* timeout or EINTR */

    int nrecv;
    Py_BEGIN_ALLOW_THREADS
    nrecv = recvmmsg(self->fd, self->batch.hdrs, (unsigned int)self->batch.cap,
                     MSG_DONTWAIT, NULL);
    Py_END_ALLOW_THREADS

    if (nrecv < 0) {
        if (errno == EAGAIN || errno == EINTR) return -2;
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    if (nrecv == 0) return -2;

    /* Receiving data resets the inactivity deadline. */
    *deadline         = mono_now() + self->timeout;
    self->batch.count = nrecv;
    self->batch.pos   = 0;
    return 1;
}


/* ================================================================== */
/* Packet ingestion: parse → fill_gap → copy_and_complete             */
/* ================================================================== */

/* Extract the 6-byte little-endian byte_count from bytes 4-9 of the header.
 * (Bytes 0-3 are the sequence number, unused for frame assembly.) */
static inline uint64_t
parse_byte_count(const uint8_t *pkt)
{
    return  (uint64_t)pkt[4]
          | ((uint64_t)pkt[5] <<  8)
          | ((uint64_t)pkt[6] << 16)
          | ((uint64_t)pkt[7] << 24)
          | ((uint64_t)pkt[8] << 32)
          | ((uint64_t)pkt[9] << 40);
}

/* Zero-fill `gap` bytes into the current ring slot, capped to the slot boundary.
 * Any bytes that overflow the slot are stored in pending_gap for later draining. */
static void
fill_gap(FrameStreamObject *self, Py_ssize_t gap)
{
    Py_ssize_t space = self->ring.slot_size - self->ring.fill;
    Py_ssize_t fill  = gap < space ? gap : space;
    memset(ring_cur(&self->ring) + self->ring.fill, 0, (size_t)fill);
    self->ring.fill    += fill;
    self->drop_bytes   += fill;
    self->pending_gap   = gap - fill;
}

/* Copy `dlen` bytes of payload into the ring, detecting frame completion.
 * Handles the rare case where a single packet straddles a frame boundary.
 * Returns the completed ring slot index, or -1 if no frame completed. */
static int
copy_and_complete(FrameStreamObject *self, const uint8_t *data, int dlen,
                  double *out_ts, Py_ssize_t *out_drop)
{
    int pos = 0;
    while (pos < dlen) {
        Py_ssize_t space = self->ring.slot_size - self->ring.fill;
        int        chunk = dlen - pos;
        if ((Py_ssize_t)chunk > space) chunk = (int)space;

        memcpy(ring_cur(&self->ring) + self->ring.fill, data + pos, (size_t)chunk);
        self->ring.fill += (Py_ssize_t)chunk;
        self->offset    += (uint64_t)chunk;
        pos             += chunk;

        if (self->ring.fill >= self->ring.slot_size) {
            int done  = self->ring.slot;
            *out_ts   = self->frame_ts;
            *out_drop = self->drop_bytes;
            ring_next(&self->ring);
            self->drop_bytes = 0;

            /* Packet straddles a frame boundary: copy the tail into the new slot. */
            int leftover = dlen - pos;
            if (leftover > 0) {
                Py_ssize_t lcopy = (Py_ssize_t)leftover;
                if (lcopy > self->ring.slot_size) lcopy = self->ring.slot_size;
                memcpy(ring_cur(&self->ring), data + pos, (size_t)lcopy);
                self->ring.fill = lcopy;
                self->offset   += (uint64_t)lcopy;
                self->frame_ts  = wall_now();
            }
            return done;
        }
    }
    return -1;
}

/* Ingest one UDP packet; returns the completed ring slot index or -1. */
static int
feed_pkt(FrameStreamObject *self, const uint8_t *pkt, int pkt_len,
         double *out_ts, Py_ssize_t *out_drop)
{
    if (pkt_len < FS_HDR_LEN) return -1;

    uint64_t bc = parse_byte_count(pkt);

    /* Timestamp the frame from when its first byte arrives. */
    if (self->ring.fill == 0) self->frame_ts = wall_now();

    /* On the very first packet, align the offset to a frame boundary. */
    if (self->offset == 0)
        self->offset = bc - (bc % (uint64_t)self->ring.slot_size);

    int64_t missing = (int64_t)bc - (int64_t)self->offset;
    if (missing < 0) return -1;  /* out-of-order packet; discard */

    if (missing > 0) {
        self->offset = bc;
        fill_gap(self, (Py_ssize_t)missing);
    }

    return copy_and_complete(self, pkt + FS_HDR_LEN, pkt_len - FS_HDR_LEN,
                             out_ts, out_drop);
}


/* ================================================================== */
/* next_frame helpers                                                   */
/* ================================================================== */

/*
 * Drain one step of the pending gap into the current ring slot.
 *
 * Returns a new frame result (new ref) if the slot filled, NULL without
 * PyErr if the gap was fully drained before filling the slot, or NULL
 * with PyErr set on error (OOM from make_result).
 *
 * Callers should loop while pending_gap > 0 and this returns non-NULL.
 */
static PyObject *
drain_gap_frame(FrameStreamObject *self)
{
    if (self->ring.fill == 0) self->frame_ts = wall_now();

    Py_ssize_t space = self->ring.slot_size - self->ring.fill;
    Py_ssize_t fill  = self->pending_gap < space ? self->pending_gap : space;
    memset(ring_cur(&self->ring) + self->ring.fill, 0, (size_t)fill);
    self->ring.fill    += fill;
    self->drop_bytes   += fill;
    self->pending_gap  -= fill;

    if (self->ring.fill < self->ring.slot_size)
        return NULL;  /* gap exhausted before this slot was full */

    int        done = self->ring.slot;
    double     ts   = self->frame_ts;
    Py_ssize_t drop = self->drop_bytes;
    ring_next(&self->ring);
    self->drop_bytes = 0;
    return make_result(self, done, ts, drop);
}


/* ================================================================== */
/* next_frame                                                           */
/* ================================================================== */

static PyObject *
FS_next_frame(FrameStreamObject *self, PyObject *Py_UNUSED(ignored))
{
    double deadline = mono_now() + self->timeout;

    for (;;) {
        /* Phase 1: drain any pending multi-frame gap one frame at a time. */
        while (self->pending_gap > 0) {
            PyObject *frame = drain_gap_frame(self);
            if (frame || PyErr_Occurred()) return frame;
        }

        /* Phase 2: ensure the batch has unprocessed packets. */
        if (self->batch.pos >= self->batch.count) {
            int r = do_recv(self, &deadline);
            if (r ==  0) Py_RETURN_NONE;   /* deadline expired */
            if (r == -1) return NULL;       /* error, PyErr set */
            if (r == -2) continue;          /* EINTR or poll timeout; retry */
        }

        /* Phase 3: feed packets until a frame completes. */
        for (int i = self->batch.pos; i < self->batch.count; i++) {
            double ts; Py_ssize_t drop;
            int slot = feed_pkt(self, batch_data(&self->batch, i),
                                      batch_len(&self->batch, i), &ts, &drop);
            if (slot < 0) continue;
            self->batch.pos = i + 1;
            return make_result(self, slot, ts, drop);
        }
        self->batch.pos = self->batch.count = 0;
    }
}


/* ================================================================== */
/* Type and module definitions                                          */
/* ================================================================== */

static PyMethodDef FS_methods[] = {
    {
        "next_frame",
        (PyCFunction)FS_next_frame,
        METH_NOARGS,
        "Block until one complete frame arrives.\n\n"
        "Returns (timestamp: float, data: np.ndarray[uint8], dropped: int) or\n"
        "None on timeout.  dropped is the number of zero-filled bytes in the\n"
        "frame (0 means complete).  data is a zero-copy view into the internal\n"
        "ring buffer, valid for ring_frames subsequent calls."
    },
    { NULL }
};

static PyTypeObject FrameStreamType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_fast.FrameStream",
    .tp_doc       = (
        "FrameStream(fd, frame_size, batch_size=64, ring_frames=4, timeout=1.0)\n\n"
        "High-speed radar frame receiver.  Uses poll(2) + recvmmsg(2) to\n"
        "batch-receive UDP packets with minimal syscall overhead, assembles them\n"
        "into fixed-size frames, and delivers each completed frame as a zero-copy\n"
        "numpy view.\n\n"
        "Parameters\n"
        "----------\n"
        "fd          : int   — socket file descriptor (data_socket.fileno())\n"
        "frame_size  : int   — bytes per complete frame\n"
        "batch_size  : int   — max packets per recvmmsg call (default 64)\n"
        "ring_frames : int   — pre-allocated frame slots (default 4);\n"
        "                total memory = ring_frames * frame_size bytes\n"
        "timeout     : float — seconds before returning None (default 1.0)"
    ),
    .tp_basicsize = sizeof(FrameStreamObject),
    .tp_itemsize  = 0,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_new       = FS_new,
    .tp_init      = (initproc)FS_init,
    .tp_dealloc   = (destructor)FS_dealloc,
    .tp_methods   = FS_methods,
};

static PyModuleDef _fast_module = {
    PyModuleDef_HEAD_INIT,
    "_fast",
    "Fast DCA1000EVM UDP frame receiver (poll + recvmmsg + ring buffer).",
    -1,
    NULL
};

PyMODINIT_FUNC
PyInit__fast(void)
{
    import_array();  /* initialize numpy C API; returns NULL on failure */

    if (PyType_Ready(&FrameStreamType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&_fast_module);
    if (!m) return NULL;

    Py_INCREF(&FrameStreamType);
    if (PyModule_AddObject(m, "FrameStream", (PyObject *)&FrameStreamType) < 0) {
        Py_DECREF(&FrameStreamType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
