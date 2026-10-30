// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg %s | FileCheck %s

func.func @transfer_read_write(%arg0: memref<4xi32, 26>, %arg1: memref<4xi32, 26>, %arg2: memref<4xi32, 26>, %arg3: index) attributes {ascvf2.vector_func} {
  %c0 = arith.constant 0 : i32
  %mask = vector.create_mask %arg3 : vector<4xi1>
  %lhs = vector.transfer_read %arg0[%arg3], %c0, %mask {in_bounds = [true]} : memref<4xi32, 26>, vector<4xi32>
  %rhs = vector.transfer_read %arg1[%arg3], %c0, %mask {in_bounds = [true]} : memref<4xi32, 26>, vector<4xi32>
  %result = vector.mask %mask { arith.addi %lhs, %rhs : vector<4xi32> } : vector<4xi1> -> vector<4xi32>
  vector.transfer_write %result, %arg2[%arg3], %mask {in_bounds = [true]} : vector<4xi32>, memref<4xi32, 26>
  return
}

// CHECK-LABEL: func.func @transfer_read_write(
// CHECK: ascvf2.simple_mask_reg i32 dynamic %arg3
// CHECK-NOT: arith.index_cast
// CHECK: ascendc.data_copy_vld_reg
// CHECK: ascendc.data_copy_vld_reg
// CHECK: ascendc.add_reg
// CHECK: ascendc.data_copy_vst_reg
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write

// CHECK-LABEL: func.func @vector_broadcast(
func.func @vector_broadcast(%arg0: memref<1xi32, 26>, %arg1: memref<4xi32, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  %c0 = arith.constant 0 : i32
  %scalar = vector.transfer_read %arg0[%arg2], %c0 {in_bounds = [true]} : memref<1xi32, 26>, vector<1xi32>
  %result = vector.broadcast %scalar : vector<1xi32> to vector<4xi32>
  vector.transfer_write %result, %arg1[%arg2] {in_bounds = [true]} : vector<4xi32>, memref<4xi32, 26>
  return
}

// CHECK: ascendc.data_copy_vld_reg
// CHECK: ascendc.duplicate_reg
// CHECK: ascendc.data_copy_vst_reg
// CHECK-NOT: vector.broadcast

// CHECK-LABEL: func.func @partial_width_broadcast_store(
func.func @partial_width_broadcast_store(%arg0: memref<1xf32, 26>, %arg1: memref<32xf32, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  %c0 = arith.constant 0.0 : f32
  %scalar = vector.transfer_read %arg0[%arg2], %c0 {in_bounds = [true]} : memref<1xf32, 26>, vector<1xf32>
  %result = vector.broadcast %scalar : vector<1xf32> to vector<32xf32>
  vector.transfer_write %result, %arg1[%arg2] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, 26>
  return
}

// CHECK: ascvf2.simple_mask_reg f32 static 32
// CHECK: ascendc.duplicate_reg
// CHECK: ascvf2.simple_mask_reg f32 static 32
// CHECK: ascendc.data_copy_vst_reg
// CHECK-NOT: vector.broadcast

// CHECK-LABEL: func.func @broadcast_transfer_read(
func.func @broadcast_transfer_read(%arg0: memref<1xf32, 26>, %arg1: index) -> vector<64xf32> attributes {ascvf2.vector_func} {
  %pad = arith.constant 0.0 : f32
  %result = vector.transfer_read %arg0[%arg1], %pad
      {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>}
      : memref<1xf32, 26>, vector<64xf32>
  return %result : vector<64xf32>
}

// CHECK: ascendc.data_copy_vld_reg {{.*}} {dist = 3 : i32}
// CHECK-NOT: ascendc.duplicate_reg

// CHECK-LABEL: func.func @broadcast_transfer_read_partial_width(
func.func @broadcast_transfer_read_partial_width(%arg0: memref<1xf32, 26>, %arg1: index) -> vector<32xf32> attributes {ascvf2.vector_func} {
  %pad = arith.constant 0.0 : f32
  %result = vector.transfer_read %arg0[%arg1], %pad
      {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>}
      : memref<1xf32, 26>, vector<32xf32>
  return %result : vector<32xf32>
}

// CHECK: ascendc.data_copy_vld_reg {{.*}} {dist = 3 : i32}
// CHECK-NOT: ascendc.duplicate_reg

// CHECK-LABEL: func.func @broadcast_transfer_read_f16(
func.func @broadcast_transfer_read_f16(%arg0: memref<1xf16, 26>, %arg1: index) -> vector<128xf16> attributes {ascvf2.vector_func} {
  %pad = arith.constant 0.0 : f16
  %result = vector.transfer_read %arg0[%arg1], %pad
      {in_bounds = [true], permutation_map = affine_map<(d0) -> (0)>}
      : memref<1xf16, 26>, vector<128xf16>
  return %result : vector<128xf16>
}

// CHECK: ascendc.data_copy_vld_reg {{.*}} {dist = 2 : i32}
// CHECK-NOT: ascendc.duplicate_reg

// CHECK-LABEL: func.func @first_element_store_b8(
func.func @first_element_store_b8(%arg0: memref<1xi8, 26>, %arg1: memref<1xi8, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  %pad = arith.constant 0 : i8
  %value = vector.transfer_read %arg0[%arg2], %pad {in_bounds = [true]} : memref<1xi8, 26>, vector<1xi8>
  vector.transfer_write %value, %arg1[%arg2] {in_bounds = [true]} : vector<1xi8>, memref<1xi8, 26>
  return
}

// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 3 : i32}

// CHECK-LABEL: func.func @first_element_store_b16(
func.func @first_element_store_b16(%arg0: memref<1xf16, 26>, %arg1: memref<1xf16, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  %pad = arith.constant 0.0 : f16
  %value = vector.transfer_read %arg0[%arg2], %pad {in_bounds = [true]} : memref<1xf16, 26>, vector<1xf16>
  vector.transfer_write %value, %arg1[%arg2] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, 26>
  return
}

// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 4 : i32}

// CHECK-LABEL: func.func @first_element_store_b32(
func.func @first_element_store_b32(%arg0: memref<1xf32, 26>, %arg1: memref<1xf32, 26>, %arg2: index) attributes {ascvf2.vector_func} {
  %pad = arith.constant 0.0 : f32
  %value = vector.transfer_read %arg0[%arg2], %pad {in_bounds = [true]} : memref<1xf32, 26>, vector<1xf32>
  vector.transfer_write %value, %arg1[%arg2] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, 26>
  return
}

// CHECK: ascendc.data_copy_vst_reg {{.*}} {dist = 5 : i32}
