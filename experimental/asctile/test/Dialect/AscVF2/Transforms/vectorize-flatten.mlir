// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -canonicalize -cse -inline -symbol-dce %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (0)>

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @flatten_elementwise_2d(
// CHECK: %[[DEST:.*]] = memref.alloca() : memref<32x32xf32, 26>
// CHECK: %[[FLAT_DEST:.*]] = memref.reinterpret_cast %[[DEST]]
// CHECK: affine.for %{{.*}} = 0 to 1024 step 64
// CHECK: vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} {in_bounds = [true]} : memref<1024xf32, 26>, vector<64xf32>
// CHECK: arith.subf %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[FLAT_DEST]][%{{.*}}] {in_bounds = [true]} : vector<64xf32>, memref<1024xf32, 26>
func.func @flatten_elementwise_2d(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @flatten_reduce_as_1d(
// CHECK: affine.for %{{.*}} = 0 to 1024 step 64
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: ascvf2.reduce <sum> %{{.*}} [0 : i32] : vector<64xf32> to vector<1xf32>
func.func @flatten_reduce_as_1d(%arg0: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = arith.mulf %arg0, %arg0 : tensor<32x32xf32, #asctile.local<UB>>
  %1 = asctile.reduce_as_1d <sum> %0 : tensor<32x32xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @flatten_scalar_broadcast(
// CHECK: affine.for %{{.*}} = 0 to 1024 step 64
// CHECK: vector.transfer_read %{{.*}}[%c0], %{{.*}} {in_bounds = [true], permutation_map = #map} : memref<1xf32, 26>, vector<64xf32>
// CHECK-NOT: vector.broadcast
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<64xf32>, memref<1024xf32, 26>
func.func @flatten_scalar_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = asctile.broadcast %arg0 : tensor<1xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @keep_row_broadcast_shape
// CHECK: affine.for %{{.+}} = 0 to 32 step 32
// CHECK: vector.broadcast %{{.+}} : vector<1xf32> to vector<32xf32>
// CHECK: arith.addf %{{.+}}, %{{.+}} : vector<32xf32>
func.func @keep_row_broadcast_shape(%arg0: tensor<32x1xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  %0 = asctile.broadcast %arg0 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.addf %0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %1 : tensor<32x32xf32, #asctile.local<UB>>
}

}
