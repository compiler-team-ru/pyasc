// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-unify-bias-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @unify_identical_c2_tensors
// CHECK: %[[BIAS:[0-9]+]] = ascendc.local_tensor_v3 c2, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NOT: ascendc.local_tensor_v3 c2, {{[0-9]+}}, 64 : !ascendc.local_tensor<64xf32>
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg1
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg2
// CHECK: return
// CHECK-NEXT:}
func.func @unify_identical_c2_tensors(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  %bias1 = ascendc.local_tensor_v3 c2, 0, 64 : !ascendc.local_tensor<64xf32>
  %params1 = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias1, %arg1, %params1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  %bias2 = ascendc.local_tensor_v3 c2, 0, 64 : !ascendc.local_tensor<64xf32>
  %params2 = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias2, %arg2, %params2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  return
}

// CHECK-LABEL: func.func @unify_different_offset
// CHECK: %[[BIAS:[0-9]+]] = ascendc.local_tensor_v3 c2, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NOT: ascendc.local_tensor_v3 c2, {{[0-9]+}}, 64 : !ascendc.local_tensor<64xf32>
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg1
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg2
// CHECK: return
// CHECK-NEXT:}
func.func @unify_different_offset(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  %bias1 = ascendc.local_tensor_v3 c2, 0, 64 : !ascendc.local_tensor<64xf32>
  %params1 = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias1, %arg1, %params1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  %bias2 = ascendc.local_tensor_v3 c2, 64, 64 : !ascendc.local_tensor<64xf32>
  %params2 = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias2, %arg2, %params2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  return
}

// CHECK-LABEL: func.func @set_addr_to_zero
// CHECK: %[[BIAS:[0-9]+]] = ascendc.local_tensor_v3 c2, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK: ascendc.data_copy_l0 %[[BIAS]], %arg1
// CHECK: return
// CHECK-NEXT:}
func.func @set_addr_to_zero(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  %bias = ascendc.local_tensor_v3 c2, 128, 64 : !ascendc.local_tensor<64xf32>
  %params = ascendc.construct !ascendc.data_copy_params(%c256) : i32
  ascendc.data_copy_l0 %bias, %arg1, %params : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<*xf16>, !ascendc.data_copy_params
  return
}
