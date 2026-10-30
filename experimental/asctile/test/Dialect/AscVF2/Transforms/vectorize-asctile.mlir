// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -canonicalize -cse -inline -symbol-dce %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @vectorize_asctile_float_operations(
func.func @vectorize_asctile_float_operations(%arg0: tensor<70xf32, #asctile.local<UB>>, %scalar: f32) -> tensor<70xf32, #asctile.local<UB>> {
  // CHECK: vector.mask %{{.+}} { ascvf2.muls %{{.+}}, %arg1 : vector<64xf32>, f32 }
  // CHECK: vector.mask %{{.+}} { ascvf2.relu %{{.+}} : vector<64xf32> }
  %0 = asctile.muls %arg0, %scalar : tensor<70xf32, #asctile.local<UB>>
  %1 = asctile.relu %0 : tensor<70xf32, #asctile.local<UB>>
  return %1 : tensor<70xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_asctile_integer_operations(
func.func @vectorize_asctile_integer_operations(%arg0: tensor<70xi32, #asctile.local<UB>>, %shift: i32) -> tensor<70xi32, #asctile.local<UB>> {
  // CHECK: vector.mask %{{.+}} { ascvf2.shls %{{.+}}, %arg1 : vector<64xi32>, i32 }
  // CHECK: vector.mask %{{.+}} { ascvf2.not %{{.+}} : vector<64xi32> }
  %0 = asctile.shls %arg0, %shift : tensor<70xi32, #asctile.local<UB>>
  %1 = asctile.bitwise_not %0 : tensor<70xi32, #asctile.local<UB>>
  return %1 : tensor<70xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_asctile_broadcast_last_axis(
func.func @vectorize_asctile_broadcast_last_axis(%arg0: tensor<70x1xf32, #asctile.local<UB>>, %arg1: tensor<70x64xf32, #asctile.local<UB>>) -> tensor<70x64xf32, #asctile.local<UB>> {
  // CHECK: vector.broadcast %{{.+}} : vector<1xf32> to vector<64xf32>
  // CHECK: arith.addf %{{.+}}, %{{.+}} : vector<64xf32>
  %0 = asctile.broadcast %arg0 : tensor<70x1xf32, #asctile.local<UB>> to tensor<70x64xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<70x64xf32, #asctile.local<UB>>
  return %1 : tensor<70x64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_asctile_rank_promoting_broadcast(
func.func @vectorize_asctile_rank_promoting_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<1x70xf32, #asctile.local<UB>>) -> tensor<1x70xf32, #asctile.local<UB>> {
  // CHECK: affine.for %{{.+}} = 0 to 70 step 64
  // CHECK: vector.create_mask %{{.+}} : vector<64xi1>
  // CHECK-NOT: vector.broadcast
  // CHECK: vector.transfer_read %{{.+}}[%{{.+}}], %{{.+}}, %{{.+}} {in_bounds = [true]} : memref<70xf32, 26>, vector<64xf32>
  // CHECK: vector.mask %{{.+}} { arith.addf %{{.+}}, %{{.+}} : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %0 = asctile.broadcast %arg0 : tensor<1xf32, #asctile.local<UB>> to tensor<1x70xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<1x70xf32, #asctile.local<UB>>
  return %1 : tensor<1x70xf32, #asctile.local<UB>>
}

}
