// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-reorder-grouped-ops %s | FileCheck %s

// CHECK-LABEL: func.func @reorder_pure_ops_between_loops(
func.func @reorder_pure_ops_between_loops(%arg0: tensor<4xf32, #asctile.local<UB>>) -> tensor<4xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %1 = tensor.empty() : tensor<4xf32, #asctile.local<UB>>
  %2 = ascvf2.to_buffer %1 : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
  affine.for %arg1 = 0 to 4 {
    %cst = arith.constant 0.000000e+00 : f32
    memref.store %cst, %2[%arg1] : memref<4xf32, 26>
  }
  %3 = ascvf2.to_tensor %2 : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  %4 = ascvf2.to_buffer %3 : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
  affine.for %arg2 = 0 to 4 {
    %5 = memref.load %4[%arg2] : memref<4xf32, 26>
  }
  return %3 : tensor<4xf32, #asctile.local<UB>>
}

// CHECK: affine.for %arg1 = 0 to 4 {
// CHECK: memref.store %{{.+}}, %1[%arg1] : memref<4xf32, 26>
// CHECK: }
// CHECK-NEXT: %2 = ascvf2.to_tensor %1 : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
// CHECK-NEXT: %3 = ascvf2.to_buffer %2 : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
// CHECK-NEXT: affine.for %arg1 = 0 to 4 {
// CHECK-NEXT: %4 = memref.load %3[%arg1] : memref<4xf32, 26>
// CHECK-NEXT: }
