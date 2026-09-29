// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-apply-cv-strategy -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func.func @m_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
// CHECK-NEXT:  %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:  %0 = asctile.copy %arg0[%arg1, %arg2] {split = #asctile.split_mode<split_by_m>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %2 = arith.muli %1, %c16_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg1, %2 : i32
// CHECK-NEXT:  asctile.store %0, %arg3[%3, %arg2] : tensor<16x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg3[%arg1, %arg2] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
// CHECK-NEXT:  %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:  %0 = asctile.copy %arg0[%arg1, %arg2] {split = #asctile.split_mode<split_by_n>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %2 = arith.muli %1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  asctile.store %0, %arg3[%arg1, %3] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_copy_l0c_ub(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: i32, %arg2: i32, %arg3: tensor<32x64xf32, #asctile.global>) {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg3[%arg1, %arg2] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @m_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
// CHECK-NEXT:  %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:  %0 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %1 = arith.muli %0, %c16_i32 : i32
// CHECK-NEXT:  %2 = arith.addi %arg1, %1 : i32
// CHECK-NEXT:  %3 = asctile.copy %arg0[%2, %arg2] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @m_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
  return %0 : tensor<32x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @n_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
// CHECK-NEXT:  %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:  %0 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %1 = arith.muli %0, %c32_i32 : i32
// CHECK-NEXT:  %2 = arith.addi %arg2, %1 : i32
// CHECK-NEXT:  %3 = asctile.copy %arg0[%arg1, %2] : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  return %3 : tensor<32x64xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @n_copy_ub_l1(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: i32, %arg2: i32) -> tensor<32x64xf32, #asctile.local<L1>> {
  %0 = asctile.copy %arg0[%arg1, %arg2] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.local<L1>>
  return %0 : tensor<32x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @m_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:  %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:  %c16 = arith.constant 16 : index
// CHECK-NEXT:  %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c16 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[%1, 0] [16, 64] [64, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %extracted_slice_0 = tensor.extract_slice %arg1[%1, 0] [16, 64] [64, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.addf %extracted_slice, %extracted_slice_0 : tensor<16x64xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c16_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg2[%5, %arg4] : tensor<16x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @m_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
  %0 = arith.addf %arg0, %arg1 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg2[%arg3, %arg4] {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 16, 64>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @n_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:  %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %0 = ascendc.get_sub_block_idx : index
// CHECK-NEXT:  %1 = arith.muli %0, %c32 : index
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %arg0[0, %1] [32, 32] [32, 1] : tensor<32x64xf32, #asctile.local<UB>> to tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = math.absf %extracted_slice : tensor<32x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.get_sub_block_idx : i32
// CHECK-NEXT:  %4 = arith.muli %3, %c32_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %arg4, %4 : i32
// CHECK-NEXT:  asctile.store %2, %arg2[%arg3, %5] : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @n_elementwise(%arg0: tensor<32x64xf32, #asctile.local<UB>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>, %arg2: tensor<32x64xf32, #asctile.global>, %arg3: i32, %arg4: i32) {
  %0 = math.absf %arg0 {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>
  asctile.store %0, %arg2[%arg3, %arg4] {asctile.need_split = #asctile.split_mode<split_by_n>, asctile.split_shape = array<i64: 32, 32>} : tensor<32x64xf32, #asctile.local<UB>>, tensor<32x64xf32, #asctile.global>
  return
}
