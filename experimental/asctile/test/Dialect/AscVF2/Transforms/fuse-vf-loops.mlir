// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-fuse-vf-loops %s | FileCheck %s
// RUN: asctile-opt -ascvf2-outline-vf-group -ascvf2-vectorize -ascvf2-reorder-grouped-ops -canonicalize -cse -ascvf2-fuse-vf-loops -inline -symbol-dce %s | FileCheck %s --check-prefix=GEN
// RUN: asctile-opt -ascvf2-reorder-grouped-ops -ascvf2-fuse-vf-loops %s | FileCheck %s --check-prefix=REINTERPRET

#map = affine_map<(d0) -> (128 - d0)>
#scaled = affine_map<(d0) -> (d0 * 64)>
#shift = affine_map<(d0) -> (d0 + 1)>
#row = affine_map<(d0) -> (d0 floordiv 5)>
#column = affine_map<(d0) -> (d0 * 64 - (d0 floordiv 5) * 320)>

module attributes {asc.vf_vec_len = 256 : i32} {

func.func @fuse_nested_masked_loops(%arg0: tensor<2x128xf32, #asctile.local<UB>>) -> tensor<2x128xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %cst = arith.constant 0.000000e+00 : f32
  %input = ascvf2.to_buffer %arg0 : tensor<2x128xf32, #asctile.local<UB>> to memref<2x128xf32, 26>
  %temporary = tensor.empty() : tensor<2x128xf32, #asctile.local<UB>>
  %temporaryBuffer = ascvf2.to_buffer %temporary : tensor<2x128xf32, #asctile.local<UB>> to memref<2x128xf32, 26>
  %temporaryTensor = ascvf2.to_tensor %temporaryBuffer : memref<2x128xf32, 26> to tensor<2x128xf32, #asctile.local<UB>>
  %aliasedTemporary = ascvf2.to_buffer %temporaryTensor : tensor<2x128xf32, #asctile.local<UB>> to memref<2x128xf32, 26>
  %output = tensor.empty() : tensor<2x128xf32, #asctile.local<UB>>
  %outputBuffer = ascvf2.to_buffer %output : tensor<2x128xf32, #asctile.local<UB>> to memref<2x128xf32, 26>
  affine.for %row = 0 to 2 {
    affine.for %chunk = 0 to 128 step 64 {
      %remaining = affine.apply #map(%chunk)
      %mask = vector.create_mask %remaining : vector<64xi1>
      %value = vector.transfer_read %input[%row, %chunk], %cst, %mask {in_bounds = [true]} : memref<2x128xf32, 26>, vector<64xf32>
      vector.transfer_write %value, %temporaryBuffer[%row, %chunk], %mask {in_bounds = [true]} : vector<64xf32>, memref<2x128xf32, 26>
    }
  }
  affine.for %row = 0 to 2 {
    affine.for %chunk = 0 to 128 step 64 {
      %remaining = affine.apply #map(%chunk)
      %mask = vector.create_mask %remaining : vector<64xi1>
      %value = vector.transfer_read %aliasedTemporary[%row, %chunk], %cst, %mask {in_bounds = [true]} : memref<2x128xf32, 26>, vector<64xf32>
      vector.transfer_write %value, %outputBuffer[%row, %chunk], %mask {in_bounds = [true]} : vector<64xf32>, memref<2x128xf32, 26>
    }
  }
  return %output : tensor<2x128xf32, #asctile.local<UB>>
}

func.func @fuse_independent_vector_yields() -> vector<4xf32> attributes {ascvf2.vector_func} {
  %zero = arith.constant dense<0.000000e+00> : vector<4xf32>
  %one = arith.constant dense<1.000000e+00> : vector<4xf32>
  %first = affine.for %index = 0 to 8 step 4 iter_args(%accumulator = %zero) -> (vector<4xf32>) {
    %next = arith.addf %accumulator, %one : vector<4xf32>
    affine.yield %next : vector<4xf32>
  }
  %second = affine.for %index = 0 to 8 step 4 iter_args(%accumulator = %one) -> (vector<4xf32>) {
    %next = arith.mulf %accumulator, %one : vector<4xf32>
    affine.yield %next : vector<4xf32>
  }
  %result = arith.addf %first, %second : vector<4xf32>
  return %result : vector<4xf32>
}

func.func @fuse_elementwise_with_vector_yield(%arg0: tensor<8xf32, #asctile.local<UB>>) -> vector<4xf32> attributes {ascvf2.vector_func} {
  %zero = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %input = ascvf2.to_buffer %arg0 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %temporary = tensor.empty() : tensor<8xf32, #asctile.local<UB>>
  %temporaryBuffer = ascvf2.to_buffer %temporary : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    %value = vector.transfer_read %input[%index], %cst {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %temporaryBuffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  %result = affine.for %index = 0 to 8 step 4 iter_args(%accumulator = %zero) -> (vector<4xf32>) {
    %value = vector.transfer_read %temporaryBuffer[%index], %cst {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    %next = arith.addf %accumulator, %value : vector<4xf32>
    affine.yield %next : vector<4xf32>
  }
  return %result : vector<4xf32>
}

func.func @fuse_scaled_affine_indices(%arg0: tensor<1024xf32, #asctile.local<UB>>) -> tensor<1024xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %cst = arith.constant 0.000000e+00 : f32
  %input = ascvf2.to_buffer %arg0 : tensor<1024xf32, #asctile.local<UB>> to memref<1024xf32, 26>
  %temporary = tensor.empty() : tensor<1024xf32, #asctile.local<UB>>
  %temporaryBuffer = ascvf2.to_buffer %temporary : tensor<1024xf32, #asctile.local<UB>> to memref<1024xf32, 26>
  affine.for %chunk = 0 to 16 {
    %index = affine.apply #scaled(%chunk)
    %value = vector.transfer_read %input[%index], %cst {in_bounds = [true]} : memref<1024xf32, 26>, vector<64xf32>
    vector.transfer_write %value, %temporaryBuffer[%index] {in_bounds = [true]} : vector<64xf32>, memref<1024xf32, 26>
  }
  affine.for %chunk = 0 to 16 {
    %index = affine.apply #scaled(%chunk)
    %value = vector.transfer_read %temporaryBuffer[%index], %cst {in_bounds = [true]} : memref<1024xf32, 26>, vector<64xf32>
    %doubled = arith.addf %value, %value : vector<64xf32>
    vector.transfer_write %doubled, %temporaryBuffer[%index] {in_bounds = [true]} : vector<64xf32>, memref<1024xf32, 26>
  }
  %result = ascvf2.to_tensor %temporaryBuffer : memref<1024xf32, 26> to tensor<1024xf32, #asctile.local<UB>>
  return %result : tensor<1024xf32, #asctile.local<UB>>
}

func.func @fuse_floordiv_affine_indices(%arg0: tensor<3x304xf32, #asctile.local<UB>>) -> tensor<3x304xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %cst = arith.constant 0.000000e+00 : f32
  %input = ascvf2.to_buffer %arg0 : tensor<3x304xf32, #asctile.local<UB>> to memref<3x304xf32, 26>
  %temporary = tensor.empty() : tensor<3x304xf32, #asctile.local<UB>>
  %temporaryBuffer = ascvf2.to_buffer %temporary : tensor<3x304xf32, #asctile.local<UB>> to memref<3x304xf32, 26>
  %output = tensor.empty() : tensor<3x304xf32, #asctile.local<UB>>
  %outputBuffer = ascvf2.to_buffer %output : tensor<3x304xf32, #asctile.local<UB>> to memref<3x304xf32, 26>
  affine.for %chunk = 0 to 15 {
    %row = affine.apply #row(%chunk)
    %column = affine.apply #column(%chunk)
    %value = vector.transfer_read %input[%row, %column], %cst {in_bounds = [true]} : memref<3x304xf32, 26>, vector<64xf32>
    vector.transfer_write %value, %temporaryBuffer[%row, %column] {in_bounds = [true]} : vector<64xf32>, memref<3x304xf32, 26>
  }
  affine.for %chunk = 0 to 15 {
    %row = affine.apply #row(%chunk)
    %column = affine.apply #column(%chunk)
    %value = vector.transfer_read %temporaryBuffer[%row, %column], %cst {in_bounds = [true]} : memref<3x304xf32, 26>, vector<64xf32>
    %doubled = arith.addf %value, %value : vector<64xf32>
    vector.transfer_write %doubled, %outputBuffer[%row, %column] {in_bounds = [true]} : vector<64xf32>, memref<3x304xf32, 26>
  }
  %result = ascvf2.to_tensor %outputBuffer : memref<3x304xf32, 26> to tensor<3x304xf32, #asctile.local<UB>>
  return %result : tensor<3x304xf32, #asctile.local<UB>>
}

func.func @do_not_fuse_shifted_access(%arg0: tensor<8xf32, #asctile.local<UB>>) -> tensor<8xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %cst = arith.constant 0.000000e+00 : f32
  %buffer = ascvf2.to_buffer %arg0 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    %value = vector.transfer_read %buffer[%index], %cst {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  affine.for %index = 0 to 8 step 4 {
    %shifted = affine.apply #shift(%index)
    %value = vector.transfer_read %buffer[%shifted], %cst {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %buffer[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  %result = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  return %result : tensor<8xf32, #asctile.local<UB>>
}

func.func @do_not_fuse_result_dependence() -> vector<4xf32> attributes {ascvf2.vector_func} {
  %zero = arith.constant dense<0.000000e+00> : vector<4xf32>
  %one = arith.constant dense<1.000000e+00> : vector<4xf32>
  %first = affine.for %index = 0 to 8 step 4 iter_args(%accumulator = %zero) -> (vector<4xf32>) {
    %next = arith.addf %accumulator, %one : vector<4xf32>
    affine.yield %next : vector<4xf32>
  }
  %second = affine.for %index = 0 to 8 step 4 iter_args(%accumulator = %first) -> (vector<4xf32>) {
    %next = arith.mulf %accumulator, %one : vector<4xf32>
    affine.yield %next : vector<4xf32>
  }
  return %second : vector<4xf32>
}

func.func @fuse_outer_repeated_write_loops(%arg0: tensor<2xf32, #asctile.local<UB>>) -> tensor<2xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %value = arith.constant dense<1.000000e+00> : vector<2xf32>
  %buffer = ascvf2.to_buffer %arg0 : tensor<2xf32, #asctile.local<UB>> to memref<2xf32, 26>
  affine.for %row = 0 to 2 step 2 {
    affine.for %repeat = 0 to 2 {
      vector.transfer_write %value, %buffer[%row] {in_bounds = [true]} : vector<2xf32>, memref<2xf32, 26>
    }
  }
  affine.for %row = 0 to 2 step 2 {
    affine.for %repeat = 0 to 2 {
      vector.transfer_write %value, %buffer[%row] {in_bounds = [true]} : vector<2xf32>, memref<2xf32, 26>
    }
  }
  %result = ascvf2.to_tensor %buffer : memref<2xf32, 26> to tensor<2xf32, #asctile.local<UB>>
  return %result : tensor<2xf32, #asctile.local<UB>>
}

func.func @fuse_after_hoisting_static_reinterpret(%arg0: tensor<2x4xf32, #asctile.local<UB>>) -> tensor<2x4xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %cst = arith.constant 0.000000e+00 : f32
  %input = ascvf2.to_buffer %arg0 : tensor<2x4xf32, #asctile.local<UB>> to memref<2x4xf32, 26>
  %inputFlat = memref.reinterpret_cast %input to offset: [0], sizes: [8], strides: [1] : memref<2x4xf32, 26> to memref<8xf32, 26>
  %temporary = memref.alloca() : memref<2x4xf32, 26>
  %temporaryFlat = memref.reinterpret_cast %temporary to offset: [0], sizes: [8], strides: [1] : memref<2x4xf32, 26> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    %value = vector.transfer_read %inputFlat[%index], %cst {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %temporaryFlat[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  %output = memref.alloca() : memref<2x4xf32, 26>
  %outputFlat = memref.reinterpret_cast %output to offset: [0], sizes: [8], strides: [1] : memref<2x4xf32, 26> to memref<8xf32, 26>
  affine.for %index = 0 to 8 step 4 {
    %value = vector.transfer_read %temporaryFlat[%index], %cst {in_bounds = [true]} : memref<8xf32, 26>, vector<4xf32>
    vector.transfer_write %value, %outputFlat[%index] {in_bounds = [true]} : vector<4xf32>, memref<8xf32, 26>
  }
  %result = ascvf2.to_tensor %output : memref<2x4xf32, 26> to tensor<2x4xf32, #asctile.local<UB>>
  return %result : tensor<2x4xf32, #asctile.local<UB>>
}

func.func @fuse_generated_elementwise(%arg0: tensor<2x128xf32, #asctile.local<UB>>, %arg1: tensor<2x128xf32, #asctile.local<UB>>) -> tensor<2x128xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %0 = arith.subf %arg0, %arg1 : tensor<2x128xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<2x128xf32, #asctile.local<UB>>
  %2 = arith.divf %1, %arg0 : tensor<2x128xf32, #asctile.local<UB>>
  return %2 : tensor<2x128xf32, #asctile.local<UB>>
}

}

// CHECK-LABEL: func.func @fuse_nested_masked_loops(
// CHECK-NOT: ascvf2.vf_group
// CHECK-COUNT-1: affine.for %{{.+}} = 0 to 2 {
// CHECK-COUNT-1: affine.for %{{.+}} = 0 to 128 step 64 {
// CHECK-NEXT: %{{.+}} = affine.apply #map(%{{.+}})
// CHECK-NEXT: %{{.+}} = vector.create_mask %{{.+}} : vector<64xi1>
// CHECK-NEXT: %{{.+}} = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}], %cst, %{{.+}} {in_bounds = [true]} : memref<2x128xf32, 26>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}], %{{.+}} {in_bounds = [true]} : vector<64xf32>, memref<2x128xf32, 26>
// CHECK-NEXT: %{{.+}} = affine.apply #map(%{{.+}})
// CHECK-NEXT: %{{.+}} = vector.create_mask %{{.+}} : vector<64xi1>
// CHECK-NEXT: %{{.+}} = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}], %cst, %{{.+}} {in_bounds = [true]} : memref<2x128xf32, 26>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}], %{{.+}} {in_bounds = [true]} : vector<64xf32>, memref<2x128xf32, 26>

// CHECK-LABEL: func.func @fuse_independent_vector_yields(
// CHECK: %{{.+}}:2 = affine.for %{{.+}} = 0 to 8 step 4 iter_args(%{{.+}} = %cst, %{{.+}} = %cst_0) -> (vector<4xf32>, vector<4xf32>)
// CHECK: arith.addf %{{.+}}#0, %{{.+}}#1 : vector<4xf32>

// CHECK-LABEL: func.func @fuse_elementwise_with_vector_yield(
// CHECK-COUNT-1: affine.for %{{.+}} = 0 to 8 step 4
// CHECK-NOT: affine.for
// CHECK: arith.addf

// CHECK-LABEL: func.func @fuse_scaled_affine_indices(
// CHECK-COUNT-1: affine.for %{{.+}} = 0 to 16 {

// CHECK-LABEL: func.func @fuse_floordiv_affine_indices(
// CHECK-COUNT-1: affine.for %{{.+}} = 0 to 15 {

// CHECK-LABEL: func.func @do_not_fuse_shifted_access(
// CHECK-COUNT-2: affine.for %{{.+}} = 0 to 8 step 4

// CHECK-LABEL: func.func @do_not_fuse_result_dependence(
// CHECK-COUNT-2: affine.for %{{.+}} = 0 to 8 step 4

// CHECK-LABEL: func.func @fuse_outer_repeated_write_loops(
// CHECK-COUNT-1: affine.for %{{.+}} = 0 to 2 step 2
// CHECK-COUNT-2: affine.for %{{.+}} = 0 to 2 {

// REINTERPRET-LABEL: func.func @fuse_after_hoisting_static_reinterpret(
// REINTERPRET-COUNT-1: affine.for %{{.+}} = 0 to 8 step 4
// REINTERPRET-NOT: affine.for

// GEN-LABEL: func.func @fuse_generated_elementwise(
// GEN: ascvf2.vf_group
// GEN-COUNT-1: affine.for %{{.+}} = 0 to 256 step 64 {
// GEN: arith.subf
