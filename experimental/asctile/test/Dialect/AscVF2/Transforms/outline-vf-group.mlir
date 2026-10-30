// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group %s | FileCheck %s --check-prefix=OUTLINE
// RUN: asctile-opt -ascvf2-outline-vf-group -canonicalize -inline -symbol-dce %s | FileCheck %s --check-prefix=RESTORE

// OUTLINE-LABEL: func.func @outline_constant(
// OUTLINE: %0 = ascvf2.vf_group {
// OUTLINE: %1 = func.call @outline_constant_vf1(%arg0
// OUTLINE: ascvf2.yield %1

// RESTORE-LABEL: func.func @outline_constant(
// RESTORE: %0 = ascvf2.vf_group {
// RESTORE: %1 = arith.addf %arg0, %arg0
// RESTORE: %2 = asctile.muls %1, %cst
// RESTORE: ascvf2.yield %2
func.func @outline_constant(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %cst = arith.constant 2.000000e+00 : f32
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.muls %0, %cst : tensor<128xf32, #asctile.local<UB>>
  return %1 : tensor<128xf32, #asctile.local<UB>>
}

// OUTLINE-LABEL: func.func @outline_multiple_groups(
// OUTLINE: func.call @outline_multiple_groups_vf1(
// OUTLINE: func.call @outline_multiple_groups_vf2(

// OUTLINE-LABEL: func.func private @outline_constant_vf1(
// OUTLINE-SAME: %arg0: tensor<128xf32, #asctile.local<UB>>
// OUTLINE-SAME: attributes {ascvf2.vector_func}
// OUTLINE: %cst = arith.constant 2.000000e+00 : f32
// OUTLINE: %0 = arith.addf %arg0, %arg0
// OUTLINE: %1 = asctile.muls %0, %cst
// OUTLINE: return %1

// OUTLINE-DAG: func.func private @outline_multiple_groups_vf2(
// OUTLINE-DAG: func.func private @outline_multiple_groups_vf1(
func.func @outline_multiple_groups(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> (tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>) {
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %1 = arith.subf %0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %separator = asctile.power %1, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %2 = arith.mulf %separator, %separator : tensor<128xf32, #asctile.local<UB>>
  %3 = arith.divf %2, %separator : tensor<128xf32, #asctile.local<UB>>
  return %1, %3 : tensor<128xf32, #asctile.local<UB>>, tensor<128xf32, #asctile.local<UB>>
}
