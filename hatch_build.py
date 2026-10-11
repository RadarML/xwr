"""Hatchling build hook: compile src/xwr/capture/_fast.c."""

import os
import warnings

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        # Only compile for wheels; the sdist should ship the C source only.
        if self.target_name != "wheel":
            return

        try:
            import numpy as np
            from setuptools import Extension
            from setuptools.dist import Distribution
        except ImportError as e:
            warnings.warn(
                f"Skipping C extension build: missing dependency ({e}). "
                "Install numpy and setuptools to enable the fast capture backend.",
                stacklevel=2,
            )
            return

        try:
            ext = Extension(
                name="xwr.capture._fast",
                sources=["src/xwr/capture/_fast.c"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3", "-std=c11"],
                libraries=["rt"],  # clock_gettime on older glibc
            )

            dist = Distribution({
                "name": "xwr",
                "ext_modules": [ext],
                "package_dir": {"": "src"},  # src-layout: root package is under src/
            })
            cmd = dist.get_command_obj("build_ext")
            cmd.inplace = True
            cmd.ensure_finalized()
            cmd.run()
        except Exception as e:
            warnings.warn(
                f"Skipping C extension build: compilation failed ({e}). "
                "The package will fall back to the pure-Python capture backend.",
                stacklevel=2,
            )
            return

        # Tell hatchling to include the compiled .so in the wheel, and tag the
        # wheel as platform/interpreter-specific (e.g. cp312-cp312-linux_x86_64).
        # Use the exact output path so stale builds for other interpreters
        # (left in-place in src/) are not picked up.
        so = os.path.relpath(cmd.get_ext_fullpath(ext.name))
        build_data["artifacts"].append(so)
        build_data["force_include"][so] = os.path.relpath(so, "src")
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
