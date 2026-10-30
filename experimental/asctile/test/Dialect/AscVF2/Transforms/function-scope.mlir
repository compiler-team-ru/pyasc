// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg %s | FileCheck %s

// CHECK-LABEL: func.func @skip_unmarked_function(
// CHECK-NEXT: %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK-NEXT: return %cst : vector<64xf32>
// CHECK-NEXT: }
func.func @skip_unmarked_function() -> vector<64xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  return %cst : vector<64xf32>
}

// CHECK-LABEL: func.func @lower_marked_function(
// CHECK: ascendc.reg_tensor
// CHECK: ascvf2.simple_mask_reg f32 static 64
// CHECK: ascendc.duplicate_reg
// CHECK: builtin.unrealized_conversion_cast
func.func @lower_marked_function() -> vector<64xf32> attributes {ascvf2.vector_func} {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  return %cst : vector<64xf32>
}
