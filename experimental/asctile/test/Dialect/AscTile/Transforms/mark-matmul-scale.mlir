// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-mark-matmul-scale %s | FileCheck %s

// CHECK-LABEL:  func.func @mark_matmul(%arg0: tensor<32x64xf8E5M2, #asctile.local<L1>>, %arg1: tensor<64x32xf8E5M2, #asctile.local<L1>>, %arg2: tensor<2x1x16x2xi8, #asctile.local<L1>>, %arg3: tensor<1x2x2x16xi8, #asctile.local<L1>>) -> tensor<32x32xf32, #asctile.local<L0C>> {
// CHECK:         %0 = asctile.copy_scale %arg0[%c0_i32, %c0_i32], %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] {asctile.is_matrix_a} : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
// CHECK-NEXT:    %1 = asctile.copy_scale %arg1[%c0_i32, %c0_i32], %arg3[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<1x2x2x16xi8, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
// CHECK-NEXT:    %2 = asctile.matmul %0, %1 {asctile.mmad_scale} : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<64x32xf8E5M2, #asctile.local<L0B>> -> tensor<32x32xf32, #asctile.local<L0C>>
// CHECK-NEXT:    return %2 : tensor<32x32xf32, #asctile.local<L0C>>

func.func @mark_matmul(%arg0: tensor<32x64xf8E5M2, #asctile.local<L1>>, %arg1: tensor<64x32xf8E5M2, #asctile.local<L1>>, %arg2: tensor<2x1x16x2xi8, #asctile.local<L1>>, %arg3: tensor<1x2x2x16xi8, #asctile.local<L1>>) -> tensor<32x32xf32, #asctile.local<L0C>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy_scale %arg0[%c0_i32, %c0_i32], %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>> 
  %1 = asctile.copy_scale %arg1[%c0_i32, %c0_i32], %arg3[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<1x2x2x16xi8, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
  %2 = asctile.matmul %0, %1 : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<64x32xf8E5M2, #asctile.local<L0B>> -> tensor<32x32xf32, #asctile.local<L0C>>
  return %2 : tensor<32x32xf32, #asctile.local<L0C>>
}

// CHECK-LABEL:  func.func @skip_matmul_no_bind
// CHECK:         %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
// CHECK-NEXT:    %1 = asctile.copy %arg1[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
// CHECK-NEXT:    %2 = asctile.matmul %0, %1 : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<64x32xf8E5M2, #asctile.local<L0B>> -> tensor<32x32xf32, #asctile.local<L0C>>
// CHECK-NEXT:    return %2 : tensor<32x32xf32, #asctile.local<L0C>>

func.func @skip_matmul_no_bind(%arg0: tensor<32x64xf8E5M2, #asctile.local<L1>>, %arg1: tensor<64x32xf8E5M2, #asctile.local<L1>>) -> tensor<32x32xf32, #asctile.local<L0C>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
  %1 = asctile.copy %arg1[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
  %2 = asctile.matmul %0, %1 : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<64x32xf8E5M2, #asctile.local<L0B>> -> tensor<32x32xf32, #asctile.local<L0C>>
  return %2 : tensor<32x32xf32, #asctile.local<L0C>>
}

// CHECK-LABEL:  func.func @skip_matmul_single_bind
// CHECK:         %0 = asctile.copy_scale %arg0[%c0_i32, %c0_i32], %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
// CHECK-NEXT:    %1 = asctile.copy %arg1[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
// CHECK-NEXT:    %2 = asctile.matmul %0, %1 : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<64x32xf8E5M2, #asctile.local<L0B>> -> tensor<32x32xf32, #asctile.local<L0C>>
// CHECK-NEXT:    return %2 : tensor<32x32xf32, #asctile.local<L0C>>

func.func @skip_matmul_single_bind(%arg0: tensor<32x64xf8E5M2, #asctile.local<L1>>, %arg1: tensor<64x32xf8E5M2, #asctile.local<L1>>, %arg2: tensor<2x1x16x2xi8, #asctile.local<L1>>) -> tensor<32x32xf32, #asctile.local<L0C>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy_scale %arg0[%c0_i32, %c0_i32], %arg2[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
  %1 = asctile.copy %arg1[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
  %2 = asctile.matmul %0, %1 : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<64x32xf8E5M2, #asctile.local<L0B>> -> tensor<32x32xf32, #asctile.local<L0C>>
  return %2 : tensor<32x32xf32, #asctile.local<L0C>>
}
