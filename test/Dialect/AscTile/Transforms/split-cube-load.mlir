// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt --asctile-split-cube-load %s | FileCheck %s

// CHECK-LABEL: func.func @load_to_l0a(%arg0: tensor<64x128xf16, #asctile.global>) -> tensor<64x64xf16, #asctile.local<L0A>> {
// CHECK:      %0 = asctile.load %arg0[%c0_i32, %c0_i32], %cst {asctile.is_matrix_a} : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L1>>
// CHECK-NEXT: %1 = asctile.copy %0[%c0_i32, %c0_i32] : tensor<64x64xf16, #asctile.local<L1>>, tensor<64x64xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %1 : tensor<64x64xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @load_to_l0a(%arg0: tensor<64x128xf16, #asctile.global>) -> tensor<64x64xf16, #asctile.local<L0A>> {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.load %arg0[%c0, %c0], %cst : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L0A>>
  return %0 : tensor<64x64xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @load_to_l0b(%arg0: tensor<128x256xf16, #asctile.global>) -> tensor<64x256xf16, #asctile.local<L0B>> {
// CHECK:      %0 = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<128x256xf16, #asctile.global>, tensor<64x256xf16, #asctile.local<L1>>
// CHECK-NEXT: %1 = asctile.copy %0[%c0_i32, %c0_i32] : tensor<64x256xf16, #asctile.local<L1>>, tensor<64x256xf16, #asctile.local<L0B>>
// CHECK-NEXT: return %1 : tensor<64x256xf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @load_to_l0b(%arg0: tensor<128x256xf16, #asctile.global>) -> tensor<64x256xf16, #asctile.local<L0B>> {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.load %arg0[%c0, %c0], %cst : tensor<128x256xf16, #asctile.global>, tensor<64x256xf16, #asctile.local<L0B>>
  return %0 : tensor<64x256xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @load_to_ub(%arg0: tensor<64x128xf16, #asctile.global>) -> tensor<64x64xf16, #asctile.local<UB>> {
// CHECK:      %0 = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<64x64xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @load_to_ub(%arg0: tensor<64x128xf16, #asctile.global>) -> tensor<64x64xf16, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.load %arg0[%c0, %c0], %cst : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<UB>>
  return %0 : tensor<64x64xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @mark_matrix_a(%arg0: tensor<64x128xf16, #asctile.global>, %arg1: tensor<16x8xf16, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:      %0 = asctile.load %arg0[%c0_i32, %c0_i32], %cst {asctile.is_matrix_a} : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L1>>
// CHECK-NEXT: %1 = asctile.copy %0[%c0_i32, %c0_i32] : tensor<64x64xf16, #asctile.local<L1>>, tensor<64x64xf16, #asctile.local<L0A>>
// CHECK-NEXT: %2 = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT: asctile.matmul_acc %2, %1, %arg1 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<64x64xf16, #asctile.local<L0A>>, tensor<16x8xf16, #asctile.local<L0B>>
// CHECK-NEXT: return %2 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @mark_matrix_a(%arg0: tensor<64x128xf16, #asctile.global>, %arg1: tensor<16x8xf16, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.load %arg0[%c0, %c0], %cst : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L0A>>
  %acc = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
  asctile.matmul_acc %acc, %0, %arg1 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<64x64xf16, #asctile.local<L0A>>, tensor<16x8xf16, #asctile.local<L0B>>
  return %acc : tensor<8x8xf32, #asctile.local<L0C>>
}
