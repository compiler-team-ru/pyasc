// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-promote-mem-to-reg %s | FileCheck %s

#map = affine_map<(d0) -> (0)>

module {

// CHECK-LABEL: func.func @forward_scalar_broadcast_read
// CHECK: vector.transfer_write
// CHECK: vector.broadcast %{{.+}} : vector<1xf32> to vector<4xf32>
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @forward_scalar_broadcast_read() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<1xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<1xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true], permutation_map = #map} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @forward_unmasked_write
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @forward_unmasked_write() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @forward_unread_temporary_write
// CHECK-COUNT-1: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @forward_unread_temporary_write() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @forward_masked_write
// CHECK: vector.create_mask
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @forward_masked_write() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %mask = vector.create_mask %c4 : vector<4xi1>
  vector.transfer_write %value, %buffer[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding, %mask {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @forward_through_alias
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @forward_through_alias() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %aliasedTensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  %aliasedBuffer = ascvf2.to_buffer %aliasedTensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %aliasedBuffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @forward_each_exact_intermediate_write
// CHECK-COUNT-2: vector.transfer_write
// CHECK-NOT: vector.transfer_read
func.func @forward_each_exact_intermediate_write() attributes {ascvf2.vector_func} {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %first, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loadedFirst = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %firstSum = arith.addf %loadedFirst, %loadedFirst : vector<4xf32>
  vector.transfer_write %second, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @keep_read_after_overlapping_write
// CHECK-COUNT-2: vector.transfer_write
// CHECK: vector.transfer_read
func.func @keep_read_after_overlapping_write() attributes {ascvf2.vector_func} {
  %first = arith.constant dense<1.000000e+00> : vector<4xf32>
  %second = arith.constant dense<2.000000e+00> : vector<1xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %first, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  vector.transfer_write %second, %buffer[%c0] {in_bounds = [true]} : vector<1xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  %sum = arith.addf %loaded, %loaded : vector<4xf32>
  return
}

// CHECK-LABEL: func.func @keep_read_with_different_index
// CHECK: vector.transfer_write
// CHECK: vector.transfer_read
func.func @keep_read_with_different_index() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  %loaded = vector.transfer_read %buffer[%c4], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @keep_read_after_barrier
// CHECK: vector.transfer_write
// CHECK: ascvf2.barrier
// CHECK: vector.transfer_read
func.func @keep_read_after_barrier() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  ascvf2.barrier
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @promote_inside_loop_body
// CHECK: affine.for
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @promote_inside_loop_body() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    %loaded = vector.transfer_read %buffer[%index], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %sum = arith.addf %loaded, %loaded : vector<4xf32>
  }
  return
}

// CHECK-LABEL: func.func @forward_write_before_loop
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK: arith.addf
func.func @forward_write_before_loop() attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  affine.for %index = 0 to 8 {
    %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %sum = arith.addf %loaded, %loaded : vector<4xf32>
    affine.yield
  }
  return
}

// CHECK-LABEL: func.func @keep_write_before_zero_trip_loop
// CHECK: %[[INITIAL:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
// CHECK: %[[UPDATED:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK: vector.transfer_write %[[INITIAL]]
// CHECK: vector.transfer_write %[[UPDATED]]
// CHECK: vector.transfer_read
func.func @keep_write_before_zero_trip_loop() -> vector<4xf32> attributes {ascvf2.vector_func} {
  %initial = arith.constant dense<1.000000e+00> : vector<4xf32>
  %updated = arith.constant dense<2.000000e+00> : vector<4xf32>
  %padding = arith.constant 0.000000e+00 : f32
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %initial, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  affine.for %index = 0 to 0 {
    vector.transfer_write %updated, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
    affine.yield
  }
  %loaded = vector.transfer_read %buffer[%c0], %padding {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
  return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @keep_write_to_returned_temporary
// CHECK: vector.transfer_write
// CHECK-NOT: vector.transfer_read
func.func @keep_write_to_returned_temporary() -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<4xf32>
  %tensor = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %c0 = arith.constant 0 : index
  vector.transfer_write %value, %buffer[%c0] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}

}
