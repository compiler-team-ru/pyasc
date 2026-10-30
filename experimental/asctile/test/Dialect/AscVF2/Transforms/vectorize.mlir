// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -canonicalize -cse -inline -symbol-dce %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (-d0 + 129)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> (-d0 + 70)>

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @vectorize_float_2d(%arg0: tensor<2x128xf32, #asctile.local<UB>>, %arg1: tensor<2x128xf32, #asctile.local<UB>>) -> tensor<2x128xf32, #asctile.local<UB>> {
  // CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: %0 = ascvf2.vf_group {
  // CHECK-NEXT: %[[FIRST:.*]] = memref.alloca() : memref<2x128xf32, 26>
  // CHECK: memref.reinterpret_cast %[[FIRST]]
  // CHECK: %[[ARG0:.*]] = ascvf2.to_buffer %arg0 : tensor<2x128xf32, #asctile.local<UB>> to memref<2x128xf32, 26>
  // CHECK: memref.reinterpret_cast %[[ARG0]]
  // CHECK: %[[ARG1:.*]] = ascvf2.to_buffer %arg1 : tensor<2x128xf32, #asctile.local<UB>> to memref<2x128xf32, 26>
  // CHECK: memref.reinterpret_cast %[[ARG1]]
  // CHECK: affine.for %arg2 = 0 to 256 step 64 {
  // CHECK: %[[LHS:.*]] = vector.transfer_read %{{.+}}[%arg2], %cst {in_bounds = [true]} : memref<256xf32, 26>, vector<64xf32>
  // CHECK: %[[RHS:.*]] = vector.transfer_read %{{.+}}[%arg2], %cst {in_bounds = [true]} : memref<256xf32, 26>, vector<64xf32>
  // CHECK-NEXT: %[[COMPUTE:.*]] = arith.subf %[[LHS]], %[[RHS]] : vector<64xf32>
  // CHECK-NEXT: vector.transfer_write %[[COMPUTE]], %{{.+}}[%arg2] {in_bounds = [true]} : vector<64xf32>, memref<256xf32, 26>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[SECOND:.*]] = memref.alloca() : memref<2x128xf32, 26>
  // CHECK: memref.reinterpret_cast %[[SECOND]]
  // CHECK: affine.for %arg2 = 0 to 256 step 64 {
  // CHECK: %[[LHS2:.*]] = vector.transfer_read %{{.+}}[%arg2], %cst {in_bounds = [true]} : memref<256xf32, 26>, vector<64xf32>
  // CHECK: %[[RHS2:.*]] = vector.transfer_read %{{.+}}[%arg2], %cst {in_bounds = [true]} : memref<256xf32, 26>, vector<64xf32>
  // CHECK-NEXT: %[[RESULT2:.*]] = arith.mulf %[[LHS2]], %[[RHS2]] : vector<64xf32>
  // CHECK-NEXT: vector.transfer_write %[[RESULT2]], %{{.+}}[%arg2] {in_bounds = [true]} : vector<64xf32>, memref<256xf32, 26>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[RESULT:.*]] = ascvf2.to_tensor %[[SECOND]] : memref<2x128xf32, 26> to tensor<2x128xf32, #asctile.local<UB>>
  // CHECK-NEXT: ascvf2.yield %[[RESULT]] : tensor<2x128xf32, #asctile.local<UB>>
  // CHECK-NEXT: } : tensor<2x128xf32, #asctile.local<UB>>
  // CHECK-NEXT: return %0 : tensor<2x128xf32, #asctile.local<UB>>
  // CHECK-NEXT: }
func.func @vectorize_float_2d(%arg0: tensor<2x128xf32, #asctile.local<UB>>, %arg1: tensor<2x128xf32, #asctile.local<UB>>) -> tensor<2x128xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<2x128xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<2x128xf32, #asctile.local<UB>>
  return %1 : tensor<2x128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_integer_tail(%arg0: tensor<129xi32, #asctile.local<UB>>, %arg1: tensor<129xi32, #asctile.local<UB>>) -> (tensor<129xi32, #asctile.local<UB>>, tensor<129xi32, #asctile.local<UB>>) {
  // CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT: %0:2 = ascvf2.vf_group {
  // CHECK-NEXT: %[[FIRST:.*]] = memref.alloca() : memref<129xi32, 26>
  // CHECK-NEXT: %[[ARG0:.*]] = ascvf2.to_buffer %arg0 : tensor<129xi32, #asctile.local<UB>> to memref<129xi32, 26>
  // CHECK-NEXT: %[[ARG1:.*]] = ascvf2.to_buffer %arg1 : tensor<129xi32, #asctile.local<UB>> to memref<129xi32, 26>
  // CHECK-NEXT: affine.for %arg2 = 0 to 129 step 64 {
  // CHECK-NEXT: %5 = affine.apply #map(%arg2)
  // CHECK-NEXT: %6 = vector.create_mask %5 : vector<64xi1>
  // CHECK-NEXT: %7 = vector.transfer_read %[[ARG0]][%arg2], %c0_i32, %6 {in_bounds = [true]} : memref<129xi32, 26>, vector<64xi32>
  // CHECK-NEXT: %8 = vector.transfer_read %[[ARG1]][%arg2], %c0_i32, %6 {in_bounds = [true]} : memref<129xi32, 26>, vector<64xi32>
  // CHECK-NEXT: %9 = vector.mask %6 { arith.subi %7, %8 : vector<64xi32> } : vector<64xi1> -> vector<64xi32>
  // CHECK-NEXT: vector.transfer_write %9, %[[FIRST]][%arg2], %6 {in_bounds = [true]} : vector<64xi32>, memref<129xi32, 26>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[FIRST_RESULT:.*]] = ascvf2.to_tensor %[[FIRST]] : memref<129xi32, 26> to tensor<129xi32, #asctile.local<UB>>
  // CHECK-NEXT: %[[SECOND:.*]] = memref.alloca() : memref<129xi32, 26>
  // CHECK-NEXT: affine.for %arg2 = 0 to 129 step 64 {
  // CHECK-NEXT: %5 = affine.apply #map(%arg2)
  // CHECK-NEXT: %6 = vector.create_mask %5 : vector<64xi1>
  // CHECK-NEXT: %7 = vector.transfer_read %[[FIRST]][%arg2], %c0_i32, %6 {in_bounds = [true]} : memref<129xi32, 26>, vector<64xi32>
  // CHECK-NEXT: %8 = vector.transfer_read %[[ARG0]][%arg2], %c0_i32, %6 {in_bounds = [true]} : memref<129xi32, 26>, vector<64xi32>
  // CHECK-NEXT: %9 = vector.mask %6 { arith.muli %7, %8 : vector<64xi32> } : vector<64xi1> -> vector<64xi32>
  // CHECK-NEXT: vector.transfer_write %9, %[[SECOND]][%arg2], %6 {in_bounds = [true]} : vector<64xi32>, memref<129xi32, 26>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[SECOND_RESULT:.*]] = ascvf2.to_tensor %[[SECOND]] : memref<129xi32, 26> to tensor<129xi32, #asctile.local<UB>>
  // CHECK-NEXT: ascvf2.yield %[[FIRST_RESULT]], %[[SECOND_RESULT]] : tensor<129xi32, #asctile.local<UB>>, tensor<129xi32, #asctile.local<UB>>
  // CHECK-NEXT: } : tensor<129xi32, #asctile.local<UB>>, tensor<129xi32, #asctile.local<UB>>
  // CHECK-NEXT: return %0#0, %0#1 : tensor<129xi32, #asctile.local<UB>>, tensor<129xi32, #asctile.local<UB>>
  // CHECK-NEXT: }
func.func @vectorize_integer_tail(%arg0: tensor<129xi32, #asctile.local<UB>>, %arg1: tensor<129xi32, #asctile.local<UB>>) -> (tensor<129xi32, #asctile.local<UB>>, tensor<129xi32, #asctile.local<UB>>) {
  %0 = arith.subi %arg0, %arg1 : tensor<129xi32, #asctile.local<UB>>
  %1 = arith.muli %0, %arg0 : tensor<129xi32, #asctile.local<UB>>
  return %0, %1 : tensor<129xi32, #asctile.local<UB>>, tensor<129xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_math_unary(%arg0: tensor<70xf32, #asctile.local<UB>>) -> tensor<70xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  // CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[DEST:.*]] = memref.alloca() : memref<70xf32, 26>
// CHECK-NEXT: %[[SOURCE:.*]] = ascvf2.to_buffer %arg0 : tensor<70xf32, #asctile.local<UB>> to memref<70xf32, 26>
  // CHECK-NEXT: affine.for %arg1 = 0 to 70 step 64 {
// CHECK-NEXT: %[[REMAINING:.*]] = affine.apply #map1(%arg1)
// CHECK-NEXT: %[[MASK:.*]] = vector.create_mask %[[REMAINING]] : vector<64xi1>
// CHECK-NEXT: %[[INPUT:.*]] = vector.transfer_read %[[SOURCE]][%arg1], %cst, %[[MASK]] {in_bounds = [true]} : memref<70xf32, 26>, vector<64xf32>
// CHECK-NEXT: %[[UNARY_RESULT:.*]] = vector.mask %[[MASK]] { math.absf %[[INPUT]] : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
// CHECK-NEXT: vector.transfer_write %[[UNARY_RESULT]], %[[DEST]][%arg1], %[[MASK]] {in_bounds = [true]} : vector<64xf32>, memref<70xf32, 26>
  // CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT:.*]] = ascvf2.to_tensor %[[DEST]] : memref<70xf32, 26> to tensor<70xf32, #asctile.local<UB>>
// CHECK-NEXT: return %[[RESULT]] : tensor<70xf32, #asctile.local<UB>>
  // CHECK-NEXT: }
func.func @vectorize_math_unary(%arg0: tensor<70xf32, #asctile.local<UB>>) -> tensor<70xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = math.absf %arg0 : tensor<70xf32, #asctile.local<UB>>
  return %0 : tensor<70xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @vectorize_partial_hardware_width(
func.func @vectorize_partial_hardware_width(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  // CHECK: affine.for %arg1 = 0 to 32 step 32
  // CHECK: %[[REMAINING:.*]] = affine.apply #map2(%arg1)
  // CHECK: %[[MASK:.*]] = vector.create_mask %[[REMAINING]] : vector<32xi1>
  // CHECK: vector.mask %[[MASK]]
  // CHECK: vector.transfer_write {{.+}}, {{.+}}[%arg1], %[[MASK]]
  %0 = math.absf %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

}
