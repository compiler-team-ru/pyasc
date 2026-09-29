// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-prepare-cv-strategy %s | FileCheck %s

// CHECK-LABEL: func.func @prepare_split_by_m_flow(
// CHECK-SAME:   %arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>,
// CHECK-SAME:   %arg2: tensor<32x128xf32, #asctile.local<UB>>, %arg3: tensor<32x128xf32, #asctile.global>) {
// CHECK:       asctile.cv_strategy <split_by_m> {
// CHECK-NEXT:    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %1 = arith.mulf %0, %arg1 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %2 = asctile.reduce <sum> %1 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 1>, dims = [1 : i32]}
// CHECK-NEXT:    %3 = asctile.broadcast %2 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    %4 = arith.addf %3, %arg2 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:    asctile.store %4, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 128>}
// CHECK-NEXT:  }
func.func @prepare_split_by_m_flow(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x128xf32, #asctile.local<UB>>, %arg3: tensor<32x128xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %1 = arith.mulf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    %2 = asctile.reduce <sum> %1 {dims = [1 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x1xf32, #asctile.local<UB>>
    %3 = asctile.broadcast %2 : tensor<32x1xf32, #asctile.local<UB>> to tensor<32x128xf32, #asctile.local<UB>>
    %4 = arith.addf %3, %arg2 : tensor<32x128xf32, #asctile.local<UB>>
    asctile.store %4, %arg3[%c0_i32, %c0_i32] : tensor<32x128xf32, #asctile.local<UB>>, tensor<32x128xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @prepare_split_by_n_flow(
// CHECK-SAME:   %arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>,
// CHECK-SAME:   %arg2: tensor<32x64xf32, #asctile.local<UB>>, %arg3: tensor<32x64xf32, #asctile.global>) {
// CHECK:       asctile.cv_strategy <split_by_n> {
// CHECK-NEXT:    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %1 = arith.addf %0, %arg1 {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %2 = asctile.reduce <sum> %1 {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 1, 32>, dims = [0 : i32]}
// CHECK-NEXT:    %3 = asctile.broadcast %2 {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %4 = arith.mulf %3, %arg2 {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    asctile.store %4, %arg3[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:  }
func.func @prepare_split_by_n_flow(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.local<UB>>, %arg3: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_n> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %1 = arith.addf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    %2 = asctile.reduce <sum> %1 {dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.local<UB>>
    %3 = asctile.broadcast %2 : tensor<1x64xf32, #asctile.local<UB>> to tensor<32x64xf32, #asctile.local<UB>>
    %4 = arith.mulf %3, %arg2 : tensor<32x64xf32, #asctile.local<UB>>
    asctile.store %4, %arg3[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  }
  return
}
