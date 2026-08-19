#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# ====================== 配置项（固定版本要求）======================
TORCH_VERSION="2.7.1"
# torch_npu发行版本用于release tag，wheel版本用于Python包文件名。
TORCH_NPU_RELEASE_VERSION="7.3.0"
TORCH_NPU_WHEEL_VERSION="2.7.1.post2"
NPU_TAG="v${TORCH_NPU_RELEASE_VERSION}-pytorch${TORCH_VERSION}"
# =================================================================

echo "============================================="
echo " PyTorch + torch_npu 一键安装脚本 "
echo " 固定版本：PyTorch=$TORCH_VERSION | torch_npu发行版=$TORCH_NPU_RELEASE_VERSION"
echo " torch_npu wheel版本：$TORCH_NPU_WHEEL_VERSION"
echo "============================================="

# 1. 检查 Python 环境
echo -e "\n[1/5] 检测 Python 环境..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD=$(command -v python3)
elif command -v python &> /dev/null; then
    PYTHON_CMD=$(command -v python)
else
    echo "错误：未检测到 Python，请先安装 Python3"
    exit 1
fi

# 获取版本号（如 3.7、3.8、3.9、3.10、3.11、3.12）
PY_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_TAG="cp${PY_VERSION//./}"
echo "检测到 Python：$PYTHON_CMD，版本：$PY_VERSION ($PY_TAG)"

# ====================== 精准判断：只支持 3.9/3.10/3.11/3.12 ======================
if [[ "$PY_VERSION" == "3.12" || "$PY_VERSION" == "3.9" || "$PY_VERSION" == "3.10" || "$PY_VERSION" == "3.11" ]]; then
    echo "✅ Python 版本符合要求，继续安装..."
else
    echo -e "\n❌ 错误：当前 Python 版本为 $PY_VERSION"
    echo "   本脚本仅支持 Python 3.9 / 3.10 / 3.11 / 3.12"
    echo "   请切换 Python 版本后重试！"
    exit 1
fi
# ==============================================================================

if ! "$PYTHON_CMD" -m pip --version > /dev/null 2>&1; then
    echo "错误：当前 Python 环境未安装 pip：$PYTHON_CMD"
    exit 1
fi

# 2. 检测系统架构
echo -e "\n[2/5] 检测系统架构..."
ARCH=$(uname -m)
echo "检测到系统架构：$ARCH"

if [ "$ARCH" = "x86_64" ]; then
    ARCH_TAG="manylinux_2_28_x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH_TAG="manylinux_2_28_aarch64"
else
    echo "错误：不支持的架构：$ARCH"
    exit 1
fi

# 3. 生成下载链接
echo -e "\n[3/5] 生成安装包链接..."
TORCH_WHL="torch-${TORCH_VERSION}+cpu-${PY_TAG}-${PY_TAG}-${ARCH_TAG}.whl"
TORCH_URL="https://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-${PY_TAG}-${PY_TAG}-${ARCH_TAG}.whl"
NPU_WHL="torch_npu-${TORCH_NPU_WHEEL_VERSION}-${PY_TAG}-${PY_TAG}-${ARCH_TAG}.whl"
NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/${NPU_TAG}/${NPU_WHL}"

echo "PyTorch: $TORCH_WHL"
echo "torch_npu: $NPU_WHL"

download_wheel()
{
    local url="$1"
    local target="$2"
    if [[ -s "$target" ]]; then
        return
    fi

    local temporary
    temporary=$(mktemp "${target}.part.XXXXXX")
    if wget -q --show-progress -O "$temporary" "$url"; then
        mv -- "$temporary" "$target"
    else
        rm -f -- "$temporary"
        return 1
    fi
}

# 4. 安装 PyTorch
echo -e "\n[4/5] 安装 PyTorch..."
download_wheel "$TORCH_URL" "$TORCH_WHL"
"$PYTHON_CMD" -m pip install --upgrade "$TORCH_WHL"

# 5. 安装 torch_npu
echo -e "\n[5/5] 安装 torch_npu..."
download_wheel "$NPU_URL" "$NPU_WHL"
"$PYTHON_CMD" -m pip install --upgrade "$NPU_WHL"

# 验证
echo -e "\n============================================="
echo "✅ 安装成功！"
"$PYTHON_CMD" -c "import torch; import torch_npu; print('PyTorch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('NPU:', torch.npu.is_available())"
echo "============================================="
