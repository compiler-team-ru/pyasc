// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-bind-load-scale %s | FileCheck %s


// CHECK-LABEL:    func.func @test_load_a
// CHECK:            %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
// CHECK-NEXT:       %1 = asctile.load %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32], %c0_i8 {asctile.is_matrix_a, asctile.load_scale} : tensor<2x1x16x2xi8, #asctile.global>, tensor<2x1x16x2xi8, #asctile.local<L1>>
// CHECK-NEXT:       %2 = asctile.copy_scale %arg0[%c0_i32, %c0_i32], %1[%c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
// CHECK-NEXT:       %3 = asctile.copy %1[%c0_i32, %c0_i32] : tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L0A>>
// CHECK-NEXT:       return %2 : tensor<32x64xf8E5M2, #asctile.local<L0A>>
func.func @test_load_a(%arg0: tensor<32x64xf8E5M2, #asctile.local<L1>>, %arg1: tensor<2x1x16x2xi8, #asctile.global>) -> tensor<32x64xf8E5M2, #asctile.local<L0A>> {
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf8E5M2, #asctile.local<L1>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
  %1 = asctile.load %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32], %c0_i8 : tensor<2x1x16x2xi8, #asctile.global>, tensor<2x1x16x2xi8, #asctile.local<L1>>
  %2 = asctile.copy %1[%c0_i32, %c0_i32] : tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L0A>>
  %3 = asctile.bind_scale %0, %2 : tensor<32x64xf8E5M2, #asctile.local<L0A>>, tensor<2x1x16x2xi8, #asctile.local<L0A>>, tensor<32x64xf8E5M2, #asctile.local<L0A>>
  return %3 : tensor<32x64xf8E5M2, #asctile.local<L0A>>
}

// CHECK-LABEL:  func.func @test_load_b(%arg0: tensor<64x32xf8E5M2, #asctile.local<L1>>, %arg1: tensor<2x1x16x2xi8, #asctile.global>) -> tensor<64x32xf8E5M2, #asctile.local<L0B>> {
// CHECK:         %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
// CHECK-NEXT:    %1 = asctile.load %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32], %c0_i8 {asctile.load_scale} : tensor<2x1x16x2xi8, #asctile.global>, tensor<2x1x16x2xi8, #asctile.local<L1>>
// CHECK-NEXT:    %2 = asctile.copy_scale %arg0[%c0_i32, %c0_i32], %1[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
// CHECK-NEXT:    %3 = asctile.copy %1[%c0_i32, %c0_i32] : tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L0B>>
// CHECK-NEXT:    return %2 : tensor<64x32xf8E5M2, #asctile.local<L0B>>
func.func @test_load_b(%arg0: tensor<64x32xf8E5M2, #asctile.local<L1>>, %arg1: tensor<2x1x16x2xi8, #asctile.global>) -> tensor<64x32xf8E5M2, #asctile.local<L0B>> {
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<64x32xf8E5M2, #asctile.local<L1>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
  %1 = asctile.load %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32], %c0_i8 : tensor<2x1x16x2xi8, #asctile.global>, tensor<2x1x16x2xi8, #asctile.local<L1>>
  %2 = asctile.copy %1[%c0_i32, %c0_i32] : tensor<2x1x16x2xi8, #asctile.local<L1>>, tensor<2x1x16x2xi8, #asctile.local<L0B>>
  %3 = asctile.bind_scale %0, %2 : tensor<64x32xf8E5M2, #asctile.local<L0B>>, tensor<2x1x16x2xi8, #asctile.local<L0B>>, tensor<64x32xf8E5M2, #asctile.local<L0B>>
  return %3 : tensor<64x32xf8E5M2, #asctile.local<L0B>>
}

