"""Hatchling build hook: compile src/xwr/capture/_fast.c."""

import glob

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        import numpy as np
        from setuptools import Extension
        from setuptools.dist import Distribution

        ext = Extension(
            name="xwr.capture._fast",
            sources=["src/xwr/capture/_fast.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native", "-ffast-math", "-std=c11"],
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

        # Tell hatchling to include the compiled .so in the wheel.
        for so in glob.glob("src/xwr/capture/_fast*.so"):
            build_data["artifacts"].append(so)
            build_data["force_include"][so] = so
