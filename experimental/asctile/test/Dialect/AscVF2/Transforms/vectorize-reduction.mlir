// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -canonicalize -cse -inline -symbol-dce %s | FileCheck %s

// CHECK-NOT: tensor.empty

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @vectorize_final_axis_sum(
func.func @vectorize_final_axis_sum(%arg0: tensor<129xf32, #asctile.local<UB>>, %arg1: tensor<129xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: memref.alloca() : memref<129xf32, 26>
  // CHECK: vector.transfer_write
  // CHECK: memref.alloca() : memref<1xf32, 26>
  // CHECK: ascvf2.reduce <sum>
  // CHECK: ascvf2.to_tensor
  %0 = arith.subf %arg0, %arg1 : tensor<129xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<129xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_final_axis_max(
func.func @vectorize_final_axis_max(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: memref.alloca() : memref<128xf32, 26>
  // CHECK: memref.alloca() : memref<1xf32, 26>
  // CHECK: ascvf2.reduce <max>
  %0 = arith.mulf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <max> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_final_axis_min(
func.func @vectorize_final_axis_min(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: memref.alloca() : memref<128xf32, 26>
  // CHECK: memref.alloca() : memref<1xf32, 26>
  // CHECK: ascvf2.reduce <min>
  %0 = arith.mulf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <min> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_final_axis_multidimensional(
func.func @vectorize_final_axis_multidimensional(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<2x1xf32, #asctile.local<UB>> {
  // CHECK: memref.alloca() : memref<2x129xf32, 26>
  // CHECK: memref.alloca() : memref<2x1xf32, 26>
  // CHECK: ascvf2.reduce <sum>
  %0 = arith.subf %arg0, %arg1 : tensor<2x129xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [1 : i32]} : tensor<2x129xf32, #asctile.local<UB>>, tensor<2x1xf32, #asctile.local<UB>>
  return %1 : tensor<2x1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_reduce_as_1d(
func.func @vectorize_reduce_as_1d(%arg0: tensor<2x129xf32, #asctile.local<UB>>, %arg1: tensor<2x129xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: memref.alloca() : memref<2x129xf32, 26>
  // CHECK: memref.alloca() : memref<1xf32, 26>
  // CHECK: ascvf2.reduce <sum>
  %0 = arith.subf %arg0, %arg1 : tensor<2x129xf32, #asctile.local<UB>>
  %1 = asctile.reduce_as_1d <sum> %0 : tensor<2x129xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

}
