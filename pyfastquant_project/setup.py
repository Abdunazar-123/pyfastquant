#!/usr/bin/env python3
"""PyFastQuant build script -- works on Windows, Linux, and macOS.

Windows produces _core.pyd, Linux/macOS produce _core*.so; everything
else (CMake invocation, arguments) is shared so this file doesn't need
a separate branch per platform, only the final artifact filename does.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import pybind11
import numpy as np


class CMakeBuildExt(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError(
                "CMake is required to build PyFastQuant. "
                "Install it from https://cmake.org/download/ "
                "or `pip install cmake`."
            ) from e
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(extdir, exist_ok=True)

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        build_args = ['--config', 'Release']
        if sys.platform == 'win32':
            build_args += ['--', '/m']
        else:
            build_args += ['--', f'-j{os.cpu_count() or 1}']

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ['cmake', str(Path().absolute())] + cmake_args,
            cwd=str(build_temp)
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=str(build_temp)
        )

        ext_filename = '_core.pyd' if sys.platform == 'win32' else self._posix_ext_filename()

        candidates = [
            Path(extdir) / ext_filename,                          # already in place
            Path(build_temp) / 'Release' / ext_filename,           # MSVC multi-config layout
            Path(build_temp) / ext_filename,                       # single-config layout
            Path(build_temp) / 'Release' / 'pyfastquant' / ext_filename,
        ]

        target_path = Path(extdir) / ext_filename
        for source_path in candidates:
            if source_path.exists() and source_path != target_path:
                self.copy_file(str(source_path), str(target_path))
                return
            if source_path == target_path and source_path.exists():
                return

        raise RuntimeError(
            f"Built extension not found (looked for {ext_filename} under {build_temp})."
        )

    @staticmethod
    def _posix_ext_filename():
        # e.g. _core.cpython-311-x86_64-linux-gnu.so -- but CMake/pybind11
        # typically just names it _core.so in the output dir, so fall
        # back to that if the versioned name isn't found.
        import sysconfig
        suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
        return f'_core{suffix}'


extensions = [
    Extension(
        'pyfastquant._core',
        sources=[
            'src/engine/monte_carlo.cpp',
            'src/engine/black_scholes.cpp',
            'src/engine/linear_algebra.cpp',
            'src/bindings/pybind_wrapper.cpp',
        ],
        include_dirs=[
            str(Path('include')),
            pybind11.get_include(),
            np.get_include(),
        ],
        language='c++',
    ),
]


setup(
    name='pyfastquant',
    version='2.0.0',
    description='High-performance quantitative finance library',
    packages=find_packages(exclude=['tests', 'tests.*', 'benchmarks', 'benchmarks.*']),
    ext_modules=extensions,
    cmdclass={'build_ext': CMakeBuildExt},
    python_requires='>=3.8',
    install_requires=['numpy>=1.19.0'],
)
