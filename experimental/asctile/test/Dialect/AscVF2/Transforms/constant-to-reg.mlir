// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg %s | FileCheck %s

// CHECK-LABEL: func.func @duplicate_float_constant(
func.func @duplicate_float_constant() -> vector<64xf32> attributes {ascvf2.vector_func} {
  // CHECK-NEXT: %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: %0 = ascendc.reg_tensor : <f32>
  // CHECK-NEXT: %1 = ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
  // CHECK-NEXT: ascendc.duplicate_reg %0, %cst, %1 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  // CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %0 : !ascendc.reg_tensor<f32> to vector<64xf32>
  // CHECK-NEXT: return %2 : vector<64xf32>
  // CHECK-NEXT: }
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  return %cst : vector<64xf32>
}

// CHECK-LABEL: func.func @duplicate_integer_constant(
func.func @duplicate_integer_constant() -> vector<64xi32> attributes {ascvf2.vector_func} {
  // CHECK-NEXT: %c-2147483648_i32 = arith.constant -2147483648 : i32
  // CHECK-NEXT: %0 = ascendc.reg_tensor : <i32>
  // CHECK-NEXT: %1 = ascvf2.simple_mask_reg i32 static 64 : !ascendc.mask_reg
  // CHECK-NEXT: ascendc.duplicate_reg %0, %c-2147483648_i32, %1 : !ascendc.reg_tensor<i32>, i32, !ascendc.mask_reg
  // CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %0 : !ascendc.reg_tensor<i32> to vector<64xi32>
  // CHECK-NEXT: return %2 : vector<64xi32>
  // CHECK-NEXT: }
  %cst = arith.constant dense<-2147483648> : vector<64xi32>
  return %cst : vector<64xi32>
}
