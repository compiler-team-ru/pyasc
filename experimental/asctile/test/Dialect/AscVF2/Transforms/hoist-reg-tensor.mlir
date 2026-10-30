// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-hoist-reg-tensor %s | FileCheck %s

func.func @hoist_reg_tensors() attributes {ascvf2.vector_func} {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  %first = ascendc.reg_tensor : <f32>
  %second = ascendc.reg_tensor : <f32>
  ascendc.abs_reg %first, %second, %mask :
    !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %index = %c0 to %c8 step %c1 {
    %third = ascendc.reg_tensor : <f32>
    ascendc.abs_reg %third, %first, %mask :
      !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }

  return
}

// CHECK-LABEL: func.func @hoist_reg_tensors(
// CHECK-NEXT: %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT: %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT: %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT: %3 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT: ascendc.abs_reg %0, %1, %3 :
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c8 = arith.constant 8 : index
// CHECK: scf.for
// CHECK: ascendc.abs_reg %2, %0, %3 :
// CHECK: return
