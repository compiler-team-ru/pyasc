// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asclower-redress-i1-tensor -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @redress_splat_constant(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<-1> : tensor<2xi8, #asctile.local<UB>>
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %cst : tensor<2xi8, #asctile.local<UB>> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %1 = "select_like"(%0, %arg0, %arg1) : (tensor<16xi1, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @redress_splat_constant(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<true> : tensor<16xi1, #asctile.local<UB>>
  %0 = "select_like"(%cst, %arg0, %arg1) : (tensor<16xi1, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>)
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @redress_dense_constant(%arg0: tensor<16x2xf32, #asctile.local<UB>>, %arg1: tensor<16x2xf32, #asctile.local<UB>>) -> tensor<16x2xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<[85, -86, 85, -86]> : tensor<4xi8, #asctile.local<UB>>
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %cst : tensor<4xi8, #asctile.local<UB>> to tensor<16x2xi1, #asctile.local<UB>>
// CHECK-NEXT:  %1 = "select_like"(%0, %arg0, %arg1) : (tensor<16x2xi1, #asctile.local<UB>>, tensor<16x2xf32, #asctile.local<UB>>, tensor<16x2xf32, #asctile.local<UB>>) -> tensor<16x2xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<16x2xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @redress_dense_constant(%arg0: tensor<16x2xf32, #asctile.local<UB>>, %arg1: tensor<16x2xf32, #asctile.local<UB>>) -> tensor<16x2xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<[[true, false], [true, false], [true, false], [true, false],
                               [false, true], [false, true], [false, true], [false, true],
                               [true, false], [true, false], [true, false], [true, false],
                               [false, true], [false, true], [false, true], [false, true]]> : tensor<16x2xi1, #asctile.local<UB>>
  %0 = "select_like"(%cst, %arg0, %arg1) : (tensor<16x2xi1, #asctile.local<UB>>, tensor<16x2xf32, #asctile.local<UB>>, tensor<16x2xf32, #asctile.local<UB>>) -> (tensor<16x2xf32, #asctile.local<UB>>)
  return %0 : tensor<16x2xf32, #asctile.local<UB>>
}
