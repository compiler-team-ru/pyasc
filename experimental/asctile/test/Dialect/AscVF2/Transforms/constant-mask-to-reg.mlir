// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg --cse %s | FileCheck %s

// CHECK-LABEL: func.func @lower_constant_mask_all
// CHECK: ascvf2.simple_mask_reg f32 static 64
// CHECK: ascendc.add_reg
func.func @lower_constant_mask_all(%arg0: vector<64xf32>) -> vector<64xf32> attributes {ascvf2.vector_func} {
  %mask = vector.constant_mask [64] : vector<64xi1>
  %result = vector.mask %mask { arith.addf %arg0, %arg0 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_constant_mask_vl_patterns
// CHECK: ascvf2.simple_mask_reg f32 static 1
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 2
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 3
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 4
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 8
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 16
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 32
// CHECK: ascendc.add_reg
// CHECK: ascvf2.simple_mask_reg f32 static 64
// CHECK: ascendc.add_reg
func.func @lower_constant_mask_vl_patterns(%arg0: vector<64xf32>) -> vector<64xf32> attributes {ascvf2.vector_func} {
  %mask1 = vector.constant_mask [1] : vector<64xi1>
  %result1 = vector.mask %mask1 { arith.addf %arg0, %arg0 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask2 = vector.constant_mask [2] : vector<64xi1>
  %result2 = vector.mask %mask2 { arith.addf %result1, %result1 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask3 = vector.constant_mask [3] : vector<64xi1>
  %result3 = vector.mask %mask3 { arith.addf %result2, %result2 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask4 = vector.constant_mask [4] : vector<64xi1>
  %result4 = vector.mask %mask4 { arith.addf %result3, %result3 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask8 = vector.constant_mask [8] : vector<64xi1>
  %result8 = vector.mask %mask8 { arith.addf %result4, %result4 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask16 = vector.constant_mask [16] : vector<64xi1>
  %result16 = vector.mask %mask16 { arith.addf %result8, %result8 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask32 = vector.constant_mask [32] : vector<64xi1>
  %result32 = vector.mask %mask32 { arith.addf %result16, %result16 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  %mask64 = vector.constant_mask [64] : vector<64xi1>
  %result64 = vector.mask %mask64 { arith.addf %result32, %result32 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  return %result64 : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_constant_mask_update_mask
// CHECK: ascvf2.simple_mask_reg f32 static 5
// CHECK: ascendc.add_reg
func.func @lower_constant_mask_update_mask(%arg0: vector<64xf32>) -> vector<64xf32> attributes {ascvf2.vector_func} {
  %mask = vector.constant_mask [5] : vector<64xi1>
  %result = vector.mask %mask { arith.addf %arg0, %arg0 : vector<64xf32> } : vector<64xi1> -> vector<64xf32>
  return %result : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_unmasked_full_mask_fallback
// CHECK: ascvf2.simple_mask_reg f32 static 63
// CHECK: ascendc.add_reg
func.func @lower_unmasked_full_mask_fallback(%arg0: vector<63xf32>) -> vector<63xf32> attributes {ascvf2.vector_func} {
  %result = arith.addf %arg0, %arg0 : vector<63xf32>
  return %result : vector<63xf32>
}
