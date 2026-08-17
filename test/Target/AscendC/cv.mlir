// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void lower_if_aic(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2) {
// CHECK:        AscendC::LocalTensor<float> v4;
// CHECK-NEXT:   if ASCEND_IS_AIC {
// CHECK-NEXT:     AscendC::LocalTensor<float> v5{AscendC::TPosition::CO1, 0, 1024};
// CHECK-NEXT:     AscendC::Mmad(v5, v1, v2, v3);
// CHECK-NEXT:     v4 = v5;
// CHECK-NEXT:   };
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @lower_if_aic(%arg0: !ascendc.local_tensor<16x16xf32>, %arg1: !ascendc.local_tensor<16x16xf32>) {
  %c16_i32 = arith.constant 16 : i32
  %0 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
  ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    ascendc.mmad %1, %arg0, %arg1, %0 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.mmad_params
    ascendc.yield %1 : !ascendc.local_tensor<16x16xf32>
  }
  return
}

// CHECK-LABEL:void lower_if_aiv(AscendC::LocalTensor<float> v1) {
// CHECK:        AscendC::LocalTensor<float> v2;
// CHECK-NEXT:   if ASCEND_IS_AIV {
// CHECK-NEXT:     AscendC::LocalTensor<float> v3{AscendC::TPosition::VECCALC, 0, 128};
// CHECK-NEXT:     AscendC::Relu(v3, v1, c32_i64);
// CHECK-NEXT:     v2 = v3;
// CHECK-NEXT:   };
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @lower_if_aiv(%arg0: !ascendc.local_tensor<32xf32>) {
  %c32_i64 = arith.constant 32 : i64
  ascendc.if_aiv(%arg0 : !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<16x16xf32> {
    %0 = ascendc.local_tensor_v3 veccalc, 0, 128 : !ascendc.local_tensor<32xf32>
    ascendc.relu_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
    ascendc.yield %0 : !ascendc.local_tensor<32xf32>
  }
  return
}
