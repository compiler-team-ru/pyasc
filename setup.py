# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import platform

from distutils.command.clean import clean
import functools

import pybind11
import setuptools
import setuptools_scm
from setuptools.command import bdist_wheel, build_ext, build_py, egg_info, install
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from typing import Optional, Tuple
import urllib

DEFAULT_VERSION = "1.1.1"


@dataclass
class Package:
    url: str
    short_name: str
    full_name: str
    sym_name: Optional[str] = None


def check_env_bool(env: str) -> bool:
    return os.environ.get(env) in ["1", "true", "ON"]


def get_cache_dir() -> Path:
    user_home = os.getenv("PYASC_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return Path(user_home).resolve() / ".pyasc"


@functools.lru_cache(maxsize=1)
def get_base_dir() -> Path:
    return Path(os.path.dirname(__file__)).resolve()


@functools.lru_cache(maxsize=1)
def get_build_dir() -> Path:
    build_dir = os.environ.get("PYASC_SETUP_BUILD_DIR", "").strip()
    if build_dir:
        return Path(build_dir).resolve()
    return get_base_dir() / "build"


@functools.lru_cache(maxsize=1)
def get_platform_suffix() -> str:
    platform_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    return f"{platform_name}-{sys.implementation.name}-{python_version}"


@functools.lru_cache(maxsize=1)
def get_cmake_dir() -> Path:
    cmake_dir = get_build_dir() / f"cmake.{get_platform_suffix()}"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def get_storage_url():
    env = "PYASC_SETUP_STORAGE_URL"
    url = os.environ.get(env)
    if url:
        return url
    raise RuntimeError(f"{env} must be set")


def get_llvm_install_prefix():
    env = "LLVM_INSTALL_PREFIX"
    llvm_path = os.environ.get(env)
    return llvm_path if (llvm_path and llvm_path.strip()) else None


def get_llvm_package_info() -> Package:
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    except KeyError:
        arch = platform.machine()
    system_suffix = None
    if system == "Linux":
        if arch == 'arm64':
            system_suffix = 'ubuntu-arm64'
        elif arch == 'x64':
            vglibc = tuple(map(int, platform.libc_ver()[1].split('.')))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                system_suffix = "ubuntu-x64"
            elif vglibc > 217:
                system_suffix = "almalinux-x64"
            else:
                system_suffix = "centos-x64"
    if system_suffix is None:
        raise RuntimeError(f"LLVM pre-compiled image is not available for {system}-{arch}")
    llvm_hash_path = os.path.join(get_base_dir(), "llvm-commit.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    full_name = f"llvm-{rev}-{system_suffix}"
    return Package(f"{get_storage_url()}/llvm/{full_name}.tar.gz", "llvm", full_name, f"llvm-{system_suffix}")


def open_url(url):
    request = urllib.request.Request(url)
    return urllib.request.urlopen(request, timeout=600)


def update_symlink(link_path, source_path) -> None:
    source_path = Path(source_path)
    link_path = Path(link_path)
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        shutil.rmtree(link_path)
    print(f"creating symlink: {link_path} -> {source_path}")
    link_path.absolute().parent.mkdir(parents=True, exist_ok=True)  # Ensure link's parent directory exists
    link_path.symlink_to(source_path, target_is_directory=True)


def download_llvm_package() -> Path:
    cache_dir = get_cache_dir()
    llvm = get_llvm_package_info()
    package_root_dir = cache_dir / llvm.short_name
    package_dir = package_root_dir / llvm.full_name
    version_file = package_dir / "version.txt"
    is_cached = version_file.is_file() and version_file.read_text() == llvm.url
    if not is_cached:
        with contextlib.suppress(Exception):
            shutil.rmtree(package_root_dir)
        os.makedirs(package_root_dir, exist_ok=True)
        print(f"downloading and extracting: {llvm.url} -> {package_dir}", flush=True)
        with open_url(llvm.url) as response:
            with tarfile.open(fileobj=response, mode="r|*") as file:
                file.extractall(path=package_root_dir)
        version_file.write_text(llvm.url)
        sym_name = llvm.sym_name
        if sym_name:
            sym_link_path = os.path.join(package_root_dir, sym_name)
            update_symlink(sym_link_path, package_dir)
    return package_dir


def require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        raise RuntimeError(f"{name} is required")
    return tool


def require_tools(*names) -> Tuple[str, ...]:
    return tuple(require_tool(name) for name in names)


@functools.lru_cache(maxsize=1)
def get_requested_devtools() -> Tuple[str, ...]:
    if check_env_bool("PYASC_SETUP_DEVTOOLS"):
        return "ascir-lsp", "ascir-opt", "ascir-translate"
    return tuple()


class LocalExtension(setuptools.Extension):

    def __init__(self, modulename):
        super().__init__(modulename, sources=[])


class LocalBuildPy(build_py.build_py):

    def initialize_options(self):
        super().initialize_options()
        self.build_lib = str(get_build_dir() / f"lib.{get_platform_suffix()}")

    def run(self) -> None:
        self.run_command("build_ext")
        return super().run()


class LocalBuildExt(build_ext.build_ext):

    def initialize_options(self):
        super().initialize_options()
        build_dir = get_build_dir()
        self.build_lib = str(build_dir / f"lib.{get_platform_suffix()}")
        self.build_temp = str(build_dir / f"temp.{get_platform_suffix()}")

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: LocalExtension):
        cmake, ninja = require_tools("cmake", "ninja")
        llvm_dir = get_llvm_install_prefix()
        if llvm_dir is None:
            llvm_dir = download_llvm_package()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cmake_dir = str(get_cmake_dir())
        python_include_dir = sysconfig.get_path("platinclude")
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_config = os.environ.get("PYASC_SETUP_CONFIG", "Release")
        configure_args = [
            cmake,
            "-S",
            str(get_base_dir()),
            "-B",
            cmake_dir,
            "-G",
            "Ninja",
            "-DCMAKE_MAKE_PROGRAM=" + ninja,
            "-DCMAKE_BUILD_TYPE=" + build_config,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPython3_INCLUDE_DIR=" + python_include_dir,
            "-Dpybind11_INCLUDE_DIR=" + pybind11.get_include(),
            "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
            "-DLLVM_PREFIX_PATH=" + str(llvm_dir),
        ]
        if check_env_bool("PYASC_SETUP_CCACHE"):
            configure_args.append("-DASCIR_CCACHE=ON")
        if check_env_bool("PYASC_SETUP_CLANG_LLD"):
            configure_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]
        if check_env_bool("PYASC_SETUP_COVERAGE"):
            configure_args.append("-DASCIR_COVERAGE=ON")
        if check_env_bool("PYASC_SETUP_ASAN"):
            configure_args.append("-DASCIR_ASAN=ON")
        subprocess.check_call(configure_args)
        targets = ["libpyasc", *get_requested_devtools()]
        if check_env_bool("PYASC_SETUP_DOCS"):
            targets.append("mlir-doc")
        build_args = [
            cmake,
            "--build",
            cmake_dir,
            "--target",
            *targets,
            "--parallel",
        ]
        subprocess.check_call(build_args)


class LocalBdistWheel(bdist_wheel.bdist_wheel):

    def initialize_options(self):
        super().initialize_options()
        self.bdist_dir = str(get_build_dir() / f"bdist.{get_platform_suffix()}")


class LocalClean(clean):

    def initialize_options(self):
        super().initialize_options()
        self.build_temp = get_cmake_dir()


class LocalEggInfo(egg_info.egg_info):

    def initialize_options(self):
        super().initialize_options()
        try:
            rel_dir = get_build_dir().relative_to(get_base_dir())
            if rel_dir.is_dir():
                self.egg_base = str(rel_dir)
        except ValueError:
            self.egg_base = os.curdir


class LocalInstall(install.install):

    def initialize_options(self):
        super().initialize_options()
        self.build_lib = str(get_build_dir() / f"lib.{get_platform_suffix()}")


def wheel_base_version(version: setuptools_scm.ScmVersion) -> str:
    return os.environ.get("PYASC_SETUP_VERSION", "1.1.1")


def wheel_local_version(version: setuptools_scm.ScmVersion) -> str:
    local = ""
    commit_date = version.node_date
    if commit_date:
        date_num = commit_date.strftime("%Y%m%d")
        local += f".dev{date_num}"
    local += f"+{version.node}"
    if version.dirty:
        today = version.time.strftime("%Y%m%d")
        local += f".d{today}"
    suffix = os.environ.get("PYASC_SETUP_VERSION_SUFFIX")
    if suffix:
        local += suffix
    return local


def get_project_version():
    try:
        return setuptools_scm.get_version(
            version_scheme=wheel_base_version,
            root=".",
            relative_to=__file__,
        )
    except Exception:
        return DEFAULT_VERSION


packages = [
    "asc",
    "asc/_C",
    "asc/codegen",
    "asc/common",
    "asc/language",
    "asc/language/adv",
    "asc/language/basic",
    "asc/language/core",
    "asc/language/fwk",
    "asc/language/tile",
    "asc/lib",
    "asc/lib/host",
    "asc/lib/runtime",
    "asc/runtime",
    "asc2",
]

extras_require = {
    "coverage": [
        "coverage[toml]",
        "gcovr",
        "pytest-cov",
    ],
    "dev": [
        "isort>=6.0.1",
        "mypy>=1.15.0",
        "ruff>=0.11.5",
        "yapf>=0.43.0",
    ],
    "docs": [
        "myst-parser==4.0.0",
        "Sphinx",
        "sphinx-rtd-theme==3.0.2",
        "sphinx-markdown-builder==0.6.8",
    ],
    "test": [
        "lit",
        "pytest",
        "pytest-xdist",
    ],
}


def setup() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9+ is required")
    data_files = None
    devtools = get_requested_devtools()
    if devtools:
        print("packaging development tools:", *devtools)
        data_files = [("bin", [str(get_cmake_dir() / "bin" / tool) for tool in devtools])]
    cur_dir = Path(__file__).parent
    setuptools.setup(
        name=os.environ.get("PYASC_SETUP_NAME", "pyasc"),
        description="Programming language for writing efficient custom operators " \
            "with native support for Python standard specifications",
        long_description=(cur_dir / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        version=get_project_version(),
        license="CANN Open Software License Agreement Version 2.0",
        url="https://gitcode.com/cann/pyasc",
        packages=packages,
        package_dir={"": "python"},
        data_files=data_files,
        ext_modules=[
            LocalExtension("asc._C.libpyasc"),
        ],
        package_data={
            "asc/lib/runtime": ["rt_wrapper.cpp", "npu_utils.cpp", "print_utils.cpp"],
            "asc/lib/host": ["bindings/*.cpp"],
        },
        install_requires=[
            "pybind11==2.13.1",
            "numpy<2",
            "typing_extensions",
        ],
        extras_require=extras_require,
        cmdclass={
            "bdist_wheel": LocalBdistWheel,
            "build_ext": LocalBuildExt,
            "build_py": LocalBuildPy,
            "clean": LocalClean,
            "egg_info": LocalEggInfo,
            "install": LocalInstall,
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    setup()
