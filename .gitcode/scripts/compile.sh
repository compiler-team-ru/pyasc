#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
set -o pipefail
apt update
apt install -y lcov
REPOSITORY_NAME="pyasc"

# Print and execute command
function LOG_DO() {
   local date_time
   date_time=$(date +%Y%m%d-%H%M%S)
   echo "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
   "$@"
}

cd "${WORKSPACE}" || exit 1
# export LLVM_INSTALL_PREFIX=/home/jenkins/opensource/llvm
ver=$(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
# if [[ "$ver" != "20.04" ]] ; then
    # export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
    # if [[ "${task_name}" == *_ubuntu24 ]] ; then
    #     update-alternatives --set gcc /usr/bin/gcc-14
    # fi
# fi
gcc --version
cmake --version
lcov --version
gcov --version
mkdir -p "${WORKSPACE}/build_out" || exit 1

set +e
LOG_DO python3 -m pip wheel . --wheel-dir="${WORKSPACE}" --default-timeout=100 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
ret=$?
if [[ ${ret} -eq 0 ]]; then
    echo "Install ${REPOSITORY_NAME} via pip wheel success"
else
    echo "Install ${REPOSITORY_NAME} via pip wheel failed"
    exit 1
fi

if [[ "${task_name}" == *x86* ]]; then
    WHEEL_FILE=$(find "${WORKSPACE}" -name "pyasc-*-linux_x86_64.whl" | head -n 1)
else
    WHEEL_FILE=$(find "${WORKSPACE}" -name "pyasc-*-linux_aarch64.whl" | head -n 1)
fi

if [[ -z "${WHEEL_FILE}" ]]; then
    echo "Can not find wheel file under ${WORKSPACE} for task_name='${task_name:-}'"
    exit 1
fi

cp "${WHEEL_FILE}" "${WORKSPACE}/build_out/${package_name}" || exit 1
LOG_DO python3 -m pip install "${WHEEL_FILE}"
ret=$?
if [[ ${ret} -eq 0 ]]; then
    echo "Install ${REPOSITORY_NAME} via pip install success"
else
    echo "Install ${REPOSITORY_NAME} via pip install failed"
    exit 1
fi

source /usr/local/Ascend/cann/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH:-}"
echo "Build ${REPOSITORY_NAME}."
# 跑仿真用例，非NPU用例
LOG_DO bash test/run_presmoke_model_test.sh
ret=$?
if [[ ${ret} -eq 0 ]]; then
    echo "Build ${REPOSITORY_NAME} run_presmoke_model_test success"
else
    echo "Build ${REPOSITORY_NAME} run_presmoke_model_test failed"
    exit 1
fi

# 除了类似pre_commit检查外，还有其它用例
LOG_DO bash scripts/static_check.sh
ret=$?
if [[ ${ret} -ne 0 ]]; then
    echo "bash scripts/static_check.sh failed"
    exit 1
fi
echo "package_name=${package_name}" >> "$ATOMGIT_OUTPUT"
exit 0
