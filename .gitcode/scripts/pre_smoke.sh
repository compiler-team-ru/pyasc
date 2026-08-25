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

echo "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

source /usr/local/Ascend/cann/set_env.sh
obs_base="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${package_name}"
wget -O "pyasc-1.1.1+ge7eeb79-cp311-cp311-linux_aarch64.whl" ${obs_base}
/opt/conda/bin/python3.11 -m pip install --force-reinstall --quiet pyasc-1.1.1+ge7eeb79-cp311-cp311-linux_aarch64.whl
/opt/conda/bin/python3.11 -m pip install pyyaml --disable-pip-version-check
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
bash test/run_presmoke_npu_test.sh 2>&1 | tee -a ./run_test.log

# 打包plog
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf slog.tar.gz -C /root/ascend log

date_time=$(date +%Y%m%d.%H%M%S)
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log
if grep "dcmi module initialize failed" "./npu_log/npu_info.log";then
  echo "$date_time : dcmi module initialize failed" >> ./npu_log/`date +%Y%m%d`.log
  exit 1
fi
if grep -w -e "FAIL" -e "errors" -e "fail" -e "failed" -e "error" -e "ERROR:" -e "Error" -e "error:" "./run_test.log"; then
  echo "$date_time : run test case failed"
  exit 1
fi
