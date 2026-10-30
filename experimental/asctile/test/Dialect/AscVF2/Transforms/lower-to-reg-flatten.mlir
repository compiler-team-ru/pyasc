// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -canonicalize -cse -affine-loop-coalescing -affine-loop-normalize -affine-simplify-structures -affine-expand-index-ops -lower-affine -canonicalize -cse -ascvf2-lower-to-reg -canonicalize -cse -inline -symbol-dce -ascvf2-legalize-vf-group -canonicalize -ascvf2-assign-vf-group-tensors %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @lower_flattened_elementwise
// CHECK-NOT: memref.reinterpret_cast
func.func @lower_flattened_elementwise(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %c16 step %c1
  // CHECK: ascendc.sub_reg
  // CHECK: ascendc.mul_reg
}

// CHECK-LABEL: func.func @lower_flattened_scalar_broadcast
// CHECK-NOT: memref.reinterpret_cast
func.func @lower_flattened_scalar_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = asctile.broadcast %arg0 : tensor<1xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %c16 step %c1
  // CHECK: ascendc.data_copy_vld_reg %{{.*}}, %{{.*}} {dist = 3 : i32}
  // CHECK: ascendc.add_reg
  // CHECK: ascendc.data_copy_vst_reg
}

}
