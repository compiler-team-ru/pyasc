// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-fold-cast %s | FileCheck %s

// CHECK-LABEL: func.func @fold_cast_i8_to_i32(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_cast_i8_to_i32(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xi16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_cast_i8_to_f32(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.cast <default> %0 : tensor<32xi16, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_cast_i8_to_f32(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xi16, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  %2 = asctile.cast <default> %1 : tensor<32xf16, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
  return %2 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_cast_f32_to_i32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_cast_f32_to_i32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @no_fold_unsupported_cast(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi64, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.cast <default> %0 : tensor<32xi32, #asctile.local<UB>> to tensor<32xi64, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xi64, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @no_fold_unsupported_cast(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi64, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xi32, #asctile.local<UB>> to tensor<32xi64, #asctile.local<UB>>
  return %1 : tensor<32xi64, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @no_fold_non_chained_cast(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @no_fold_non_chained_cast(%arg0: tensor<32xi8, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
  %1 = asctile.cast <default> %arg0 : tensor<32xi8, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @supported_single_cast_i32_to_i16(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi16, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xi32, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @supported_single_cast_i32_to_i16(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi16, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xi32, #asctile.local<UB>> to tensor<32xi16, #asctile.local<UB>>
  return %0 : tensor<32xi16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @no_fold_explicit_round_mode(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.cast <floor> %0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @no_fold_explicit_round_mode(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  %1 = asctile.cast <floor> %0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @no_fold_mismatched_round_mode(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <ceil> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.cast <default> %0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @no_fold_mismatched_round_mode(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <ceil> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_matching_explicit_round_mode(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cast <floor> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_matching_explicit_round_mode(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <floor> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  %1 = asctile.cast <floor> %0 : tensor<32xf16, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}
