# Developer Images

**English** | [中文](README_zh.md)

This directory provides Dockerfiles for building PyAsc development images on Ascend. Four base distributions are currently supported:

- `Dockerfile.ubuntu22.04` — Ubuntu 22.04
- `Dockerfile.ubuntu24.04` — Ubuntu 24.04
- `Dockerfile.openeuler22.03` — openEuler 22.03
- `Dockerfile.openeuler24.03` — openEuler 24.03

All images install a C/C++ toolchain, Python 3.11.15, and LLVM 19.1.7, along with the Ascend CANN toolkit and a matching ops package.
Opening a terminal automatically `source`s `/usr/local/Ascend/cann/set_env.sh` via `~/.bashrc`. LLVM is installed at `/opt/LLVM-19.1.7`, with the `LLVM_INSTALL_PREFIX` environment variable pointing to that path. Its MLIR Python bindings are exposed through `PYTHONPATH=/opt/LLVM-19.1.7/python_packages/mlir_core`.

> Build the image on a host whose CPU architecture matches the CANN toolkit package you downloaded, for example aarch64 or x86_64. Compiling LLVM from source needs ample CPU, memory, and disk; the first build can take a long time.

## CANN package URLs

The Dockerfiles do not hard-code a fixed CANN package version. You can get download links [here](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/). You will typically see channels such as `master/` and `legacy/`. After entering a snapshot directory, choose the toolkit and ops installers. Filenames look like `Ascend-cann-toolkit_<version>_linux-<arch>.run` and `Ascend-cann-910b-ops_<version>_linux-<arch>.run`. Pass the full URL of each file as the corresponding build argument:

- `CANN_TOOLKIT_URL` — Ascend CANN toolkit installer
- `CANN_OPS_URL` — Ascend CANN ops installer

## Build

From the repository root, build the Ubuntu image as follows. The openEuler image is built the same way:

```shell
docker build \
	-f docker/Dockerfile.ubuntu24.04 \
	--build-arg CANN_TOOLKIT_URL='YOUR CANN_TOOLKIT_URL' \
	--build-arg CANN_OPS_URL='YOUR CANN_OPS_URL' \
	-t pyasc-dev:ubuntu24.04 \
	.
```

Both `CANN_TOOLKIT_URL` and `CANN_OPS_URL` are required. The build fails if either argument is omitted or points to an unreachable URL.
