// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt --asclower-expand-math %s | FileCheck %s

// CHECK-LABEL: func.func @expand_math_rsqrt(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<1.000000e+00> : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = math.sqrt %arg0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = arith.divf %cst, %0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @expand_math_rsqrt(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = math.rsqrt %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @expand_math_exp2(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<0.693147182> : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.mulf %arg0, %cst : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = math.exp %0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @expand_math_exp2(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = math.exp2 %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}
