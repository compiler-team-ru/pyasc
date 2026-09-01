#!/bin/bash
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eu -o pipefail

BASE_PATH=$(cd "$(dirname "$0")" && pwd)
BUILD_PATH="${BASE_PATH}/build"
CORE_NUMS=$(nproc 2>/dev/null || echo 8)
if [ ${CORE_NUMS} -gt 8 ]; then CORE_NUMS=8; fi

if [ "${1:-}" = "--make_clean" ]; then
  rm -rf "${BUILD_PATH}"
  echo "[INFO] Build directory cleaned."
  exit 0
fi

mkdir -p "${BUILD_PATH}"
cd "${BUILD_PATH}"
ASC_DIR=${ASC_DIR:-${ASCEND_HOME_PATH}/lib64/cmake}
ASC_LINUX_PATH=${ASCEND_CANN_PACKAGE_LINUX_PATH:-${ASCEND_HOME_PATH}/aarch64-linux}
export ASCEND_CANN_PACKAGE_LINUX_PATH=${ASC_LINUX_PATH}
cmake \
  -DASC_DIR="${ASC_DIR}" \
  -DASCEND_CANN_PACKAGE_LINUX_PATH="${ASC_LINUX_PATH}" \
  ..
make -j ${CORE_NUMS}

if [ ! -f "${BUILD_PATH}/demo" ]; then
  echo "[ERROR] Build failed: ${BUILD_PATH}/demo not found"
  exit 1
fi

echo "[INFO] Build completed: ${BUILD_PATH}/demo"
