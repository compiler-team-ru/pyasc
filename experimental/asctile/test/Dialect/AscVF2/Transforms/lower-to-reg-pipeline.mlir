// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -canonicalize -cse -affine-loop-coalescing -affine-loop-normalize -affine-simplify-structures -affine-expand-index-ops -lower-affine -canonicalize -cse -ascvf2-lower-to-reg -canonicalize -cse -inline -symbol-dce -ascvf2-legalize-vf-group -canonicalize -ascvf2-assign-vf-group-tensors %s | FileCheck %s

// CHECK-NOT: tensor.empty

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @masked_integer_arithmetic(
func.func @masked_integer_arithmetic(%arg0: tensor<129xi32, #asctile.local<UB>>, %arg1: tensor<129xi32, #asctile.local<UB>>) -> tensor<129xi32, #asctile.local<UB>> {
  // CHECK: ascendc.local_tensor_auto veccalc() : <129xi32>
  // CHECK: ascendc.local_tensor_auto veccalc() : <129xi32>
  // CHECK: ascvf2.vf_group outs(%[[OUTPUT:.*]] : !ascendc.local_tensor<129xi32>) tmps(%[[TEMPORARY:.*]] : !ascendc.local_tensor<129xi32>) {
  // CHECK: %[[FIRST_BUFFER:.*]] = ascendc.local_tensor.get_phy_addr_v2
  // CHECK: %[[SECOND_BUFFER:.*]] = ascendc.local_tensor.get_phy_addr_v2
  // CHECK: ascendc.sub_reg
  // CHECK: emitasc.ptr_offset %[[FIRST_BUFFER]]
  // CHECK: ascendc.mul_reg
  // CHECK: emitasc.ptr_offset %[[SECOND_BUFFER]]
  %0 = arith.subi %arg0, %arg1 : tensor<129xi32, #asctile.local<UB>>
  %1 = arith.muli %0, %arg0 : tensor<129xi32, #asctile.local<UB>>
  return %1 : tensor<129xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @masked_math_unary(
func.func @masked_math_unary(%arg0: tensor<70xf32, #asctile.local<UB>>) -> tensor<70xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  // CHECK: ascendc.local_tensor_auto veccalc() : <70xf32>
  // CHECK: %[[BUFFER:.*]] = ascendc.local_tensor.get_phy_addr_v2
  // CHECK: ascendc.abs_reg
  // CHECK: emitasc.ptr_offset %[[BUFFER]]
  %0 = math.absf %arg0 : tensor<70xf32, #asctile.local<UB>>
  return %0 : tensor<70xf32, #asctile.local<UB>>
}

}
