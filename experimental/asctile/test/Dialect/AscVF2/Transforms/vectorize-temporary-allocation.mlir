// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -canonicalize -cse -affine-simplify-structures -ascvf2-fuse-vf-loops -inline -symbol-dce %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @distinct_temporary_allocations(
func.func @distinct_temporary_allocations(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>) {
  // CHECK: %[[FIRST:.*]] = memref.alloca() : memref<128xf32, 26>
  // CHECK: %[[SECOND:.*]] = memref.alloca() : memref<128xf32, 26>
  // CHECK: affine.for
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK-NOT: affine.for
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  return %0, %1 : tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>
}

}
