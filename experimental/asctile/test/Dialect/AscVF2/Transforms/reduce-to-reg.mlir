// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -canonicalize -cse -ascvf2-fuse-vf-loops -affine-loop-coalescing -affine-loop-normalize -affine-simplify-structures -affine-expand-index-ops -lower-affine -canonicalize -cse -ascvf2-lower-to-reg -inline -symbol-dce %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @lower_final_axis_sum(
func.func @lower_final_axis_sum(%arg0: tensor<129xf32, #asctile.local<UB>>, %arg1: tensor<129xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: ascendc.duplicate_reg %[[ACC:[^,]+]],
  // CHECK: ascendc.data_copy_vld_reg
  // CHECK: ascendc.sub_reg
  // CHECK: ascendc.data_copy_vst_reg
  // CHECK: ascendc.data_copy_vld_reg
  // CHECK: ascendc.add_reg %[[ACC:[^,]+]], %[[ACC]],
  // CHECK: ascendc.reduce_sum_reg %{{.+}}, %[[ACC]],
  // CHECK: ascendc.data_copy_vst_reg
  // CHECK-NOT: scf.yield
  // CHECK-NOT: vector.transfer_read
  // CHECK-NOT: vector.transfer_write
  // CHECK-NOT: vector.mask
  // CHECK-NOT: ascvf2.reduce
  %0 = arith.subf %arg0, %arg1 : tensor<129xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<129xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_final_axis_max(
func.func @lower_final_axis_max(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: %[[MAX_NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: ascendc.duplicate_reg %[[ACC:[^,]+]],
  // CHECK-SAME: %[[MAX_NEUTRAL]]
  // CHECK: ascendc.mul_reg
  // CHECK: ascendc.max_reg
  // CHECK: ascendc.reduce_max_reg
  %0 = arith.mulf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <max> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_final_axis_min(
func.func @lower_final_axis_min(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: %[[MIN_NEUTRAL:.*]] = arith.constant 0x7F800000 : f32
  // CHECK: ascendc.duplicate_reg %[[ACC:[^,]+]],
  // CHECK-SAME: %[[MIN_NEUTRAL]]
  // CHECK: ascendc.mul_reg
  // CHECK: ascendc.min_reg
  // CHECK: ascendc.reduce_min_reg
  %0 = arith.mulf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <min> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_final_axis_integer_max(
func.func @lower_final_axis_integer_max(%arg0: tensor<129xi32, #asctile.local<UB>>, %arg1: tensor<129xi32, #asctile.local<UB>>) -> tensor<1xi32, #asctile.local<UB>> {
  // CHECK: ascendc.duplicate_reg %[[ACC:[^,]+]],
  // CHECK: ascendc.sub_reg
  // CHECK: ascendc.max_reg %[[ACC:[^,]+]], %[[ACC]],
  // CHECK: ascendc.reduce_max_reg %{{.+}}, %[[ACC]],
  // CHECK: ascendc.data_copy_vst_reg
  // CHECK-NOT: scf.yield
  // CHECK-NOT: vector.transfer_read
  // CHECK-NOT: vector.transfer_write
  // CHECK-NOT: vector.mask
  // CHECK-NOT: ascvf2.reduce
  %0 = arith.subi %arg0, %arg1 : tensor<129xi32, #asctile.local<UB>>
  %1 = asctile.reduce <max> %0 {dims = [0 : i32]} : tensor<129xi32, #asctile.local<UB>>, tensor<1xi32, #asctile.local<UB>>
  return %1 : tensor<1xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_as_1d_sum(
// CHECK-NOT: memref.reinterpret_cast
func.func @lower_reduce_as_1d_sum(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<2x129xf32, #asctile.local<UB>>
  %1 = asctile.reduce_as_1d <sum> %0 : tensor<2x129xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
  // CHECK: ascendc.duplicate_reg %[[ACC:[^,]+]],
  // CHECK: scf.for %arg2 = %c0 to %c5 step %c1
  // CHECK: arith.muli %arg2, %c64
  // CHECK: ascendc.data_copy_vld_reg
  // CHECK: ascendc.sub_reg
  // CHECK: ascendc.data_copy_vst_reg
  // CHECK: ascendc.data_copy_vld_reg
  // CHECK: ascendc.add_reg %[[ACC:[^,]+]], %[[ACC]],
  // CHECK: ascendc.reduce_sum_reg %{{.+}}, %[[ACC]],
  // CHECK: ascendc.data_copy_vst_reg
  // CHECK-NOT: vector.transfer_read
  // CHECK-NOT: vector.transfer_write
  // CHECK-NOT: vector.mask
  // CHECK-NOT: ascvf2.reduce
}

}
