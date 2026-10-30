// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -ascvf2-fuse-vf-loops -canonicalize -cse -ascvf2-insert-vf-sync -canonicalize -inline -symbol-dce %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @insert_barrier_after_reduction_store
// CHECK: ascvf2.reduce
// CHECK: vector.transfer_write
// CHECK-NEXT: ascvf2.barrier
// CHECK-NEXT: affine.for
// CHECK: ascvf2.to_tensor
func.func @insert_barrier_after_reduction_store(%arg0: tensor<129xf32, #asctile.local<UB>>) -> tensor<129xf32, #asctile.local<UB>> {
  %difference = arith.subf %arg0, %arg0 : tensor<129xf32, #asctile.local<UB>>
  %max = asctile.reduce <max> %difference {dims = [0 : i32]} : tensor<129xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  %broadcast = asctile.broadcast %max : tensor<1xf32, #asctile.local<UB>> to tensor<129xf32, #asctile.local<UB>>
  return %broadcast : tensor<129xf32, #asctile.local<UB>>
}

}
