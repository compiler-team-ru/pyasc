// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascendc-reuse-tensor-allocation %s | FileCheck %s

// CHECK-LABEL: func.func @reuse_group_border_tensors
// CHECK: %[[TENSOR:.*]] = ascendc.local_tensor_auto
// CHECK: %[[INPUT:.*]] = ascendc.reinterpret_cast %[[TENSOR]]
// CHECK: ascvf2.vf_group ins(%[[INPUT]] : !ascendc.local_tensor<8xf32>) outs(%[[TENSOR]] : !ascendc.local_tensor<8xf32>)
// CHECK: ascendc.data_copy_vld_reg
// CHECK: ascendc.data_copy_vst_reg
// CHECK-NOT: ascendc.local_tensor_auto
func.func @reuse_group_border_tensors() {
  %c0 = arith.constant 0 : index
  %input = ascendc.local_tensor_auto veccalc() : <8xf32>
  %output = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascvf2.vf_group ins(%input : !ascendc.local_tensor<8xf32>) outs(%output : !ascendc.local_tensor<8xf32>) {
    %value = ascendc.reg_tensor : <f32>
    %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    %inputAddress = ascendc.local_tensor.get_phy_addr_v2 %input
      : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    %inputPointer = emitasc.ptr_offset %inputAddress[%c0]
      : memref<8xf32, 26>, memref<8xf32, 26>
    ascendc.data_copy_vld_reg %value, %inputPointer {dist = 0 : i32}
      : !ascendc.reg_tensor<f32>, memref<8xf32, 26>
    %outputAddress = ascendc.local_tensor.get_phy_addr_v2 %output
      : !ascendc.local_tensor<8xf32>, memref<8xf32, 26>
    %outputPointer = emitasc.ptr_offset %outputAddress[%c0]
      : memref<8xf32, 26>, memref<8xf32, 26>
    ascendc.data_copy_vst_reg %outputPointer, %value, %mask {dist = 16 : i32}
      : memref<8xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    ascvf2.yield
  }
  return
}
