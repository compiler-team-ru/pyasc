// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg %s | FileCheck %s

// CHECK-LABEL: func.func @convert_for_edge(
func.func @convert_for_edge(%arg0: index, %arg1: index, %arg2: index) -> vector<4xf32> attributes {ascvf2.vector_func} {
  // CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: %0 = ascendc.reg_tensor : <f32>
  // CHECK-NEXT: %1 = ascvf2.simple_mask_reg f32 static 4 : !ascendc.mask_reg
  // CHECK-NEXT: ascendc.duplicate_reg %0, %cst, %1 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  // CHECK-NEXT: %2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %0) -> (!ascendc.reg_tensor<f32>) {
  // CHECK-NEXT: %4 = builtin.unrealized_conversion_cast %arg4 : !ascendc.reg_tensor<f32> to vector<4xf32>
  // CHECK-NEXT: %5 = builtin.unrealized_conversion_cast %4 : vector<4xf32> to !ascendc.reg_tensor<f32>
  // CHECK-NEXT: %6 = builtin.unrealized_conversion_cast %5 : !ascendc.reg_tensor<f32> to vector<4xf32>
  // CHECK-NEXT: scf.yield %5 : !ascendc.reg_tensor<f32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %2 : !ascendc.reg_tensor<f32> to vector<4xf32>
  // CHECK-NEXT: return %3 : vector<4xf32>
  // CHECK-NEXT: }
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %0 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst) -> (vector<4xf32>) {
    %1 = builtin.unrealized_conversion_cast %arg4 : vector<4xf32> to !ascendc.reg_tensor<f32>
    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.reg_tensor<f32> to vector<4xf32>
    scf.yield %2 : vector<4xf32>
  }
  return %0 : vector<4xf32>
}
