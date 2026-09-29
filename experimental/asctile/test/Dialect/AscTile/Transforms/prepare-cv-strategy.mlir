// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-prepare-cv-strategy %s | FileCheck %s

// CHECK-LABEL: func.func @prepare_split_by_m_from_copy(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
// CHECK:       asctile.cv_strategy <split_by_m> {
// CHECK-NEXT:    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %1 = arith.mulf %0, %arg1 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    %2 = arith.addf %1, %arg1 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:    asctile.store %2, %arg2[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>}
// CHECK-NEXT:  }
func.func @prepare_split_by_m_from_copy(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %1 = arith.mulf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    %2 = arith.addf %1, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    asctile.store %2, %arg2[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @prepare_split_by_n_from_copy(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
// CHECK:       asctile.cv_strategy <split_by_n> {
// CHECK-NEXT:    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    %1 = arith.addf %0, %arg1 {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:    asctile.store %1, %arg2[%c0_i32, %c0_i32] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>}
// CHECK-NEXT:  }
func.func @prepare_split_by_n_from_copy(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_n> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    %1 = arith.addf %0, %arg1 : tensor<32x64xf32, #asctile.local<UB>>
    asctile.store %1, %arg2[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  }
  return
}
