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

echo "${ut_type:-}"
echo "${TARGET_BRANCH:-}"
echo "${obs_path:-}"

grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version

# Print and execute command
function LOG_DO() {
   local date_time
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
   "$@"
}

main(){
    local CURR_DIR REPO_HOME ret coverage_save
    #获取参数
    CURR_DIR=$(dirname "${BASH_SOURCE[0]:-$0}")
    CURR_DIR=$(cd -P "${CURR_DIR}" && pwd -P) || exit 1
    REPO_HOME=$(cd -P "${CURR_DIR}/../../../../" && pwd -P) || exit 1
    #########
    # install #
    #########
    export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
    source /home/jenkins/Ascend/cann/bin/setenv.bash
    export LLVM_INSTALL_PREFIX=/home/jenkins/opensource/llvm

    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH:-}"
    ln -sf /opt/buildtools/python-3.10.2/bin/coverage /usr/local/bin/coverage || { echo "Failed to ln coverage"; exit 1; }

    set +e
    local obs_base="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/cann-pyasc_linux-x86_64_ubuntu24.whl"
    local p_name="pyasc-1.1.1+gf870bf7-cp310-cp310-linux_x86_64.whl"
    if ! wget -O "${p_name}" "${obs_base}"; then
        echo "Download ${p_name} from ${obs_base} failed"
        exit 1
    fi
    if ! pip3 install "${p_name}"; then
        echo "Install ${p_name} failed"
        exit 1
    fi

    echo "Start run c++ testcase"
    cd "${WORKSPACE}/test" || exit 1
    coverage_save="false"
    if [[ "${ut_type}" == "python" ]]; then
        LOG_DO bash build_llt.sh --cov --run_python_ut --llvm_install_path "${LLVM_INSTALL_PREFIX}" -f "${WORKSPACE}/pr_filelist.txt"
        ret=$?
        if [[ ${ret} -eq 200 ]]; then
            echo "not need run ut"
            exit 0
        fi
    elif [[ "${ut_type}" == "cpp" ]]; then
        export PATH="/opt/buildtools/python-3.10.2/bin:${PATH}"
        export LIT_INSTALL_PATH="$(dirname "$(dirname "$(which lit)")")"
        echo $LIT_INSTALL_PATH
        LOG_DO bash build_llt.sh --cov --check-ascir --llvm_install_path "${LLVM_INSTALL_PREFIX}" --lit_install_path "${LIT_INSTALL_PATH}" -f "${WORKSPACE}/pr_filelist.txt"
        ret=$?
        if [[ ${ret} -eq 200 ]]; then
            echo "not need run ut"
            exit 0
        fi
    else
        echo "Unknown ut_type: '${ut_type:-}'"
        exit 1
    fi

    if [[ ${ret} -ne 200 && ${ret} -ne 0 ]]; then
        echo "run ut fail"
        exit 1
    fi
    echo "ut_process=coverage" >> "${ATOMGIT_OUTPUT}"
    exit 0
}

main "$@"
