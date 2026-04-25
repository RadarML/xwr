"""Interface implementation for the DCA1000EVM capture card.

# Implementation Notes

The hot-loop for data capture processing is implemented by
[`stream`][xwr.capture.api.DCA1000EVM.stream], which dispatches to one of two
paths at runtime.

!!! danger

    If the consumer cannot keep up with the incoming frame rate, data will
    silently accumulate in the kernel socket buffer until it is full.

    - Check the timing of incoming frames against the configured frame rate to
      ensure that no buffer bloat is occuring.
    - Once the buffer is full, packets will be dropped until space is
      available. This is visible to the driver via the byte count included in
      the packet header, which is logged as a warning.

## Fast Path (C Extension)

!!! info

    The C extension is only available on Linux.

The C extension is built automatically from `src/xwr/capture/_fast.c`
during installation.

- Uses `poll(2)` + `recvmmsg(2)` to batch-receive up to 64 UDP packets per
  syscall into a pre-allocated ring buffer. Frames are yielded as they are
  completed from a copy of the ring buffer.
- Falls back to Python if the extension cannot be imported.

## Python Fallback

The python fallback is always available.

- Issues one `recvfrom` per packet, and accumulates into a bytearray.
- Adequate for moderate frame rates; expect roughly 2x higher CPU usage
  compared to the C path at high throughput.
- Note that this implementation is primarily designed to be readable, and many
  potential optimizations are intentionally not implemented.
- A warning is logged at stream start when this path is taken.
"""

from . import defines, types
from .api import DCA1000EVM, DCAError

__all__ = ["types", "defines", "DCA1000EVM", "DCAError"]
