// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -canonicalize -cse -affine-loop-coalescing -affine-loop-normalize -affine-simplify-structures -affine-expand-index-ops -lower-affine -canonicalize -cse -ascvf2-lower-to-reg -inline -symbol-dce %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @lower_asctile_float_operations(
func.func @lower_asctile_float_operations(%arg0: tensor<70xf32, #asctile.local<UB>>, %scalar: f32) -> tensor<70xf32, #asctile.local<UB>> {
  // CHECK: ascendc.duplicate_reg
  // CHECK: ascendc.mul_reg
  // CHECK: ascendc.relu_reg
  %0 = asctile.muls %arg0, %scalar : tensor<70xf32, #asctile.local<UB>>
  %1 = asctile.relu %0 : tensor<70xf32, #asctile.local<UB>>
  return %1 : tensor<70xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_asctile_leaky_relu(
func.func @lower_asctile_leaky_relu(%arg0: tensor<70xf32, #asctile.local<UB>>, %scalar: f32) -> tensor<70xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg0 : tensor<70xf32, #asctile.local<UB>>
  // CHECK: ascendc.leaky_relu_reg
  %1 = asctile.leaky_relu %0, %scalar : tensor<70xf32, #asctile.local<UB>>
  return %1 : tensor<70xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_asctile_integer_operations(
func.func @lower_asctile_integer_operations(%arg0: tensor<70xi32, #asctile.local<UB>>, %shift: i32) -> tensor<70xi32, #asctile.local<UB>> {
  // CHECK: ascendc.shift_lefts_reg
  // CHECK: ascendc.not_reg
  %0 = asctile.shls %arg0, %shift : tensor<70xi32, #asctile.local<UB>>
  %1 = asctile.bitwise_not %0 : tensor<70xi32, #asctile.local<UB>>
  return %1 : tensor<70xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_asctile_broadcast_last_axis(
func.func @lower_asctile_broadcast_last_axis(%arg0: tensor<70x1xf32, #asctile.local<UB>>, %arg1: tensor<70x70xf32, #asctile.local<UB>>) -> tensor<70x70xf32, #asctile.local<UB>> {
  // CHECK: ascendc.duplicate_reg
  // CHECK: ascendc.add_reg
  %0 = asctile.broadcast %arg0 : tensor<70x1xf32, #asctile.local<UB>> to tensor<70x70xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<70x70xf32, #asctile.local<UB>>
  return %1 : tensor<70x70xf32, #asctile.local<UB>>
}

}
