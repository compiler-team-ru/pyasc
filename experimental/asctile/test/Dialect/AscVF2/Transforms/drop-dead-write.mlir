// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-drop-dead-write %s | FileCheck %s

module {

// CHECK-LABEL: func.func @remove_masked_overwrites_in_loop
// CHECK: affine.for
// CHECK-COUNT-1: vector.transfer_write
// CHECK-NOT: vector.transfer_write
func.func @remove_masked_overwrites_in_loop() -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<4xf32>
  %third = arith.constant dense<3.000000e+00> : vector<4xf32>
  %buffer = memref.alloca() : memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  affine.for %index = 0 to 8 step 4 {
    %remaining = affine.apply affine_map<(d0) -> (8 - d0)>(%index)
    %mask = vector.create_mask %remaining : vector<4xi1>
    vector.transfer_write %first, %buffer[%index], %mask {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %firstResult = vector.mask %mask { arith.addf %first, %first : vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    vector.transfer_write %firstResult, %buffer[%index], %mask {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %secondResult = vector.mask %mask { arith.mulf %firstResult, %firstResult : vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    vector.transfer_write %secondResult, %buffer[%index], %mask {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @keep_write_observed_by_read
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @keep_write_observed_by_read() attributes {ascvf2.vector_func} {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %buffer = memref.alloca() : memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %first, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  vector.transfer_write %second, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return
}

// CHECK-LABEL: func.func @keep_write_across_barrier
// CHECK: vector.transfer_write
// CHECK: ascvf2.barrier
// CHECK: vector.transfer_write
func.func @keep_write_across_barrier() -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<4xf32>
  %buffer = memref.alloca() : memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  %c0 = arith.constant 0 : index
  vector.transfer_write %first, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  ascvf2.barrier
  vector.transfer_write %second, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @remove_unread_private_temporary_writes
// CHECK-NOT: vector.transfer_write
func.func @remove_unread_private_temporary_writes() attributes {ascvf2.vector_func} {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<4xf32>
  %buffer = memref.alloca() : memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  vector.transfer_write %first, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %second, %buffer[%c4] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return
}

// CHECK-LABEL: func.func @ignore_non_vector_function
// CHECK-COUNT-2: vector.transfer_write
func.func @ignore_non_vector_function() {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<4xf32>
  %buffer = memref.alloca() : memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %first, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %second, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return
}

}
