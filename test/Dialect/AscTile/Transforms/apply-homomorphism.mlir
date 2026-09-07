// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-apply-homomorphism %s | FileCheck %s

// CHECK-LABEL: func.func @push_cast_through_select(%arg0: tensor<32xi1, #asctile.local<UB>>, %arg1: f32) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<1> : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.fptosi %arg1 : f32 to i32
// CHECK-NEXT:  %splat = tensor.splat %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = arith.select %arg0, %cst, %splat : tensor<32xi1, #asctile.local<UB>>, tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @push_cast_through_select(%arg0: tensor<32xi1, #asctile.local<UB>>, %arg1: f32) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.2> : tensor<32xf32, #asctile.local<UB>>
  %0 = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
  %1 = arith.select %arg0, %cst, %0 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  %2 = asctile.cast <default> %1 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %2 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @skip_cast_through_select(%arg0: tensor<32xi1, #asctile.local<UB>>, %arg1: f32, %arg2: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %splat = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.select %arg0, %splat, %arg2 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.cast <default> %0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @skip_cast_through_select(%arg0: tensor<32xi1, #asctile.local<UB>>, %arg1: f32, %arg2: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %splat = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.select %arg0, %splat, %arg2 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @push_splat_through_addi(%arg0: i32) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:  %0 = arith.addi %arg0, %c3_i32 : i32
// CHECK-NEXT:  %splat = tensor.splat %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %splat : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @push_splat_through_addi(%arg0: i32) -> tensor<32xi32, #asctile.local<UB>> {
  %c3_i32 = arith.constant dense<3> : tensor<32xi32, #asctile.local<UB>>
  %splat = tensor.splat %arg0 : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.addi %c3_i32, %splat : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @push_splat_through_absf(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = math.absf %arg0 : f32
// CHECK-NEXT:  %splat = tensor.splat %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %splat : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @push_splat_through_absf(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
  %splat = tensor.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
  %0 = math.absf %splat : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @skip_splat_into_select(%arg0: f32, %arg1: f32, %arg2: tensor<32xi1, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %splat = tensor.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %splat_0 = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.select %arg2, %splat, %splat_0 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @skip_splat_into_select(%arg0: f32, %arg1: f32, %arg2: tensor<32xi1, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %splat = tensor.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
  %splat_0 = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
  %2 = arith.select %arg2, %splat, %splat_0 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %2 : tensor<32xf32, #asctile.local<UB>>
}
