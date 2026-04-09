// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-input-output-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @input_output_tensor_ub_gm
// CHECK: %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
// CHECK: %1 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK: %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
func.func @input_output_tensor_ub_gm(%arg0 : !ascendc.global_tensor<*xf32>, %arg1 : !ascendc.global_tensor<*xf32>, %arg2 : !ascendc.global_tensor<*xf32>) {
    %c64_i32 = arith.constant 64 : i32
    %0 = ascendc.local_tensor_auto vecin() : <64xf32>
    ascendc.data_copy_l2 %0, %arg0, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %1, %arg1, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %2 = ascendc.local_tensor_auto vecout() : <64xf32>
    ascendc.data_copy_l2 %arg2, %2, %c64_i32 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @input_output_tensor_ub_ub
// CHECK: %0 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK: %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
func.func @input_output_tensor_ub_ub(%arg0 : !ascendc.global_tensor<*xf32>) {
    %c64_i32 = arith.constant 64 : i32
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %0, %arg0, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %1, %0, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}
