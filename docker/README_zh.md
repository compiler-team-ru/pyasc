# 开发镜像

[English](README.md) | **中文**

本目录提供在昇腾环境上构建PyAsc开发镜像的Dockerfile，当前支持四种基础发行版：

- `Dockerfile.ubuntu22.04` — Ubuntu 22.04
- `Dockerfile.ubuntu24.04` — Ubuntu 24.04
- `Dockerfile.openeuler22.03` — openEuler 22.03
- `Dockerfile.openeuler24.03` — openEuler 24.03

所有镜像都会安装C/C++工具链、Python 3.11.15、LLVM 19.1.7，并安装昇腾CANN toolkit以及配套的算子包。
打开终端会通过`~/.bashrc`自动`source /usr/local/Ascend/cann/set_env.sh`。LLVM安装在`/opt/LLVM-19.1.7`，并设置环境变量`LLVM_INSTALL_PREFIX`指向该路径，其MLIR Python bindings通过`PYTHONPATH=/opt/LLVM-19.1.7/python_packages/mlir_core`对外暴露。

> 请在CPU架构与所下载CANN toolkit包一致的机器上构建镜像，例如aarch64或x86_64，从源码编译LLVM需要充足的CPU、内存和磁盘，首次构建会比较耗时

## CANN 软件包 URL

Dockerfile没有硬编码固定的CANN软件包版本，可以在[这里](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/)中获取下载链接，通常能看到`master/`和`legacy/`等通道，进入某个快照目录后选择 toolkit与ops安装包。文件名形如`Ascend-cann-toolkit_<version>_linux-<arch>.run`和`Ascend-cann-910b-ops_<version>_linux-<arch>.run`。构建时将每个文件的完整URL，作为对应的构建参数传入

- `CANN_TOOLKIT_URL` — 昇腾CANN toolkit安装包
- `CANN_OPS_URL` — 昇腾CANN ops安装包

## 构建

在仓库根目录构建Ubuntu镜像的命令如下，openEuler镜像的构建方式相同：

```shell
docker build \
	-f docker/Dockerfile.ubuntu24.04 \
	--build-arg CANN_TOOLKIT_URL='YOUR CANN_TOOLKIT_URL' \
	--build-arg CANN_OPS_URL='YOUR CANN_OPS_URL' \
	-t pyasc-dev:ubuntu24.04 \
	.
```

`CANN_TOOLKIT_URL` 和 `CANN_OPS_URL` 都是必填项，任一参数缺失或指向无法访问的URL时，构建会失败