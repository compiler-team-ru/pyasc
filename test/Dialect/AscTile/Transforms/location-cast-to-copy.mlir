// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-location-cast-to-copy %s | FileCheck %s

// CHECK-LABEL: func.func @cast_ub_to_l1(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<L1>> {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.copy %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<L1>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @cast_ub_to_l1(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<L1>> {
  %0 = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<L1>>
  return %0 : tensor<32xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @cast_l0c_to_ub_2d(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @cast_l0c_to_ub_2d(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = tensor.cast %arg0 : tensor<16x16xf32, #asctile.local<L0C>> to tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}
