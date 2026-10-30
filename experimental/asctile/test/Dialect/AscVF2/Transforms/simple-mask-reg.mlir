// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-expand-simple-mask-reg %s | FileCheck %s
// RUN: asctile-opt --cse -ascvf2-expand-simple-mask-reg %s | FileCheck %s --check-prefix=CSE

func.func @static_patterns(%arg0: !ascendc.reg_tensor<f32>, %arg1: !ascendc.reg_tensor<f32>) attributes {ascvf2.vector_func} {
  %mask64 = ascvf2.simple_mask_reg f32 static 64 : !ascendc.mask_reg
  %mask32 = ascvf2.simple_mask_reg f32 static 32 : !ascendc.mask_reg
  %mask5 = ascvf2.simple_mask_reg f32 static 5 : !ascendc.mask_reg
  %dst64 = ascendc.reg_tensor : <f32>
  %dst32 = ascendc.reg_tensor : <f32>
  %dst5 = ascendc.reg_tensor : <f32>
  ascendc.add_reg %dst64, %arg0, %arg1, %mask64 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.add_reg %dst32, %arg0, %arg1, %mask32 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.add_reg %dst5, %arg0, %arg1, %mask5 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL: func.func @static_patterns
// CHECK: %[[COUNT5:.*]] = arith.constant 5 : index
// CHECK: %[[MASK64:.*]] = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK: %[[MASK32:.*]] = ascendc.create_mask f32, VL32 : !ascendc.mask_reg
// CHECK: %[[VAR5:.*]] = emitasc.variable %[[COUNT5]] : index, memref<1xindex>
// CHECK: %[[MASK5:.*]] = ascendc.update_mask f32, %[[VAR5]] : memref<1xindex>
// CHECK: ascendc.add_reg {{.*}}, %[[MASK64]]
// CHECK: ascendc.add_reg {{.*}}, %[[MASK32]]
// CHECK: ascendc.add_reg {{.*}}, %[[MASK5]]
// CHECK-NOT: ascvf2.simple_mask_reg

func.func @dynamic_mask(%arg0: !ascendc.reg_tensor<f32>, %arg1: !ascendc.reg_tensor<f32>, %count: index) attributes {ascvf2.vector_func} {
  %mask = ascvf2.simple_mask_reg f32 dynamic %count : !ascendc.mask_reg
  %dst = ascendc.reg_tensor : <f32>
  ascendc.add_reg %dst, %arg0, %arg1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL: func.func @dynamic_mask
// CHECK: %[[VAR:.*]] = emitasc.variable %arg2 : index, memref<1xindex>
// CHECK: %[[MASK:.*]] = ascendc.update_mask f32, %[[VAR]] : memref<1xindex>
// CHECK: ascendc.add_reg {{.*}}, %[[MASK]]
// CHECK-NOT: ascvf2.simple_mask_reg

func.func @cse_masks(%arg0: !ascendc.reg_tensor<f32>, %arg1: !ascendc.reg_tensor<f32>) attributes {ascvf2.vector_func} {
  %mask0 = ascvf2.simple_mask_reg f32 static 4 : !ascendc.mask_reg
  %mask1 = ascvf2.simple_mask_reg f32 static 4 : !ascendc.mask_reg
  %dst0 = ascendc.reg_tensor : <f32>
  %dst1 = ascendc.reg_tensor : <f32>
  ascendc.add_reg %dst0, %arg0, %arg1, %mask0 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.add_reg %dst1, %arg0, %arg1, %mask1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CSE-LABEL: func.func @cse_masks
// CSE: %[[MASK:.*]] = ascendc.create_mask f32, VL4 : !ascendc.mask_reg
// CSE: ascendc.add_reg {{.*}}, %[[MASK]]
// CSE: ascendc.add_reg {{.*}}, %[[MASK]]
// CSE-NOT: ascendc.create_mask
