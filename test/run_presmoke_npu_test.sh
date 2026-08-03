#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

mkdir copy_csv
rm -rf profiling/OPPROF_*
rm -rf copy_csv/*

run_test() {
    local test_py_file=$1
    local test_name=${test_py_file##*/}
    echo "[INFO] Running npu test: ${test_name}"
    if python3 "$test_py_file"; then
        echo "[INFO] ${test_name} example passed!"
        return 0
    else
        echo "[ERROR] ${test_name} example failed!"
        return 1
    fi
}

run_test_npu() {
    local test_py_file=$1
    local test_name=${test_py_file##*/}
    echo "[INFO] Running profiling test: ${test_name}"
    echo " $test_py_file"
    case_param=$(echo $test_py_file |awk -F "direct/" '{print $2}')
    case_param=$(echo $case_param |awk -F "." '{print $1}')
    msprof --application="python3 $test_py_file" --output="./profiling"
}

echo casename,prf_golden,prf,result >result.csv
threshold=0.1
run_test_npu_perf() {
    local test_py_file=$1
    local test_name=${test_py_file##*/}
    echo "[INFO] Running profiling data: ${test_name}"
    echo " $test_py_file"
    case_param=$(echo $test_py_file |awk -F "direct/" '{print $2}')
    case_param=$(echo $case_param |awk -F "." '{print $1}')
    prf=$(grep "$case_param.csv" all_task_durations.csv)

    if [ -n "$prf" ]; then
        prf=$(echo "$prf" | awk -F "," '{print $2}')
    else
        echo "$line case fail"
    fi

    prf_golden=$(grep "$case_param.csv" ./test/all_task_durations_golden.csv)
    prf_golden=$(echo $prf_golden |awk -F "," '{print $2}')
    awk_result=$(awk -v a="$prf" -v b="$prf_golden" 'BEGIN{ printf "%.2f", (a - b)/b}')

    if (( $(echo "$awk_result > $threshold" | bc -l) )); then
        echo "$case_param: fail, performance is $prf, golden is $prf_golden"
        result=fail
    else
        echo "$case_param: pass, performance is $prf, golden is $prf_golden"
        result=pass
    fi
    if [ -z "$prf" ]; then
        result="fail"
    fi
    if [ -z "$prf_golden" ]; then
        result="fail"
    fi

    echo $case_param,$prf_golden,$prf,$result >> result.csv
}

test_examples=(
    "./examples/01_add/add.py"
    "./examples/02_add_framework/add_framework.py"
    "./examples/03_matmul_mix/matmul_mix.py"
    "./examples/04_matmul_cube_only/matmul_cube_only.py"
    "./examples/05_matmul_leakyrelu/matmul_leakyrelu.py"
    "./examples/06_gelu/gelu.py"
    "./examples/07_swiglu/swiglu.py"
    "./examples/08_rmsnorm/rmsnorm.py"
)

test_examples_perf=(
    "./python/test/kernels/test_vadd.py"
    "./python/test/kernels/test_matmul.py"
)

passed_examples=()
failed_examples=()
sed -i 's/Model/NPU/g' ./python/test/kernels/*.py
sed -i 's/default="Model"/default="NPU"/g' ./examples/0*/*.py

for example_torch in "${test_examples[@]}"; do
    if run_test "$example_torch"; then
        passed_examples+=("$example_torch")
    else
        failed_examples+=("$example_torch")
    fi
done
for example in "${test_examples_perf[@]}"; do
    if run_test_npu "$example"; then
        passed_examples+=("$example")
    else
        failed_examples+=("$example")
    fi
done

echo "[INFO] Passed npu tests list:"
for test in "${passed_examples[@]}"; do
    echo " ${test##*/}"
done

if [ ${#failed_examples[@]} -eq 0 ]; then
    echo "[INFO] All ${#test_examples[@]} npu tests passed!"
else
    echo "[ERROR] ${#test_examples[@]} / ${#test_examples[@]} npu tests failed!"
    echo "[INFO] Failed npu tests list:"
    for test in "${failed_examples[@]}"; do
        echo " ${test##*/}"
    done
fi

pass_count=0
fail_count=0

awk -F, '
NR>1 {
    if ($4 == "pass") {
        pass_count++;
        print "[INFO] Passed profiling tests list:"$1
    } else if ($4 == "fail") {
        fail_count++;
        print "[INFO] Failed profiling tests list:"$1
    }
}
END {
    print "[INFO] Passed profiling tests: "pass_count
    if (fail_count > 0) {
        print "[ERROR] Failed profiling tests: "fail_count
    }
}' result.csv
