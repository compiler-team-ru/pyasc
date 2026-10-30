// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-insert-vf-sync -ascvf2-insert-vf-sync %s | FileCheck %s

module attributes {asc.vf_vec_len = 256 : i32} {

// CHECK-LABEL: func.func @insert_barrier_between_loops
// CHECK: vector.transfer_write
// CHECK-NEXT: {{^ *}$}}
// CHECK-NEXT: ascvf2.barrier
// CHECK-NEXT: affine.for
// CHECK-NEXT: vector.transfer_read
func.func @insert_barrier_between_loops() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @no_barrier_inside_fused_loop
// CHECK: vector.transfer_write
// CHECK-NEXT: vector.transfer_read
// CHECK-NOT: ascvf2.barrier
func.func @no_barrier_inside_fused_loop() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @one_barrier_before_reader_of_multiple_roots
// CHECK: affine.for
// CHECK: vector.transfer_write
// CHECK: affine.for
// CHECK: vector.transfer_write
// CHECK: {{^ *}$}}
// CHECK-NEXT: ascvf2.barrier
// CHECK-NEXT: affine.for
// CHECK-NOT: ascvf2.barrier
func.func @one_barrier_before_reader_of_multiple_roots() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %firstTensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %firstBuffer = ascvf2.to_buffer %firstTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %secondTensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %secondBuffer = ascvf2.to_buffer %secondTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %firstBuffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %secondBuffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %first = vector.transfer_read %firstBuffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %second = vector.transfer_read %secondBuffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %sum = arith.addf %first, %second : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @no_barrier_for_independent_roots
// CHECK-NOT: ascvf2.barrier
func.func @no_barrier_for_independent_roots() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %firstTensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %firstBuffer = ascvf2.to_buffer %firstTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %secondTensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %secondBuffer = ascvf2.to_buffer %secondTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %firstBuffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %loaded = vector.transfer_read %secondBuffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @recognize_alias_chain_and_keep_existing_barrier
// CHECK: vector.transfer_write
// CHECK-NEXT: {{^ *}$}}
// CHECK-NEXT: ascvf2.barrier
// CHECK-NEXT: affine.for
// CHECK-NEXT: vector.transfer_read
func.func @recognize_alias_chain_and_keep_existing_barrier() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %aliasedTensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  %aliasedBuffer = ascvf2.to_buffer %aliasedTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  ascvf2.barrier
  affine.for %index = 0 to 8 step 4 {
    %loaded = vector.transfer_read %aliasedBuffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @insert_barrier_before_loop_with_internal_dependency
// CHECK: vector.transfer_write
// CHECK: {{^ *}$}}
// CHECK-NEXT: ascvf2.barrier
// CHECK-NEXT: affine.for
// CHECK-NEXT: vector.transfer_write
// CHECK-NOT: ascvf2.barrier
func.func @insert_barrier_before_loop_with_internal_dependency() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

}
