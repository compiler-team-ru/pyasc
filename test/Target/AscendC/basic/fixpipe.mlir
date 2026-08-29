// Copyright (c) 2025 ISE Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_fixpipe(AscendC::GlobalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<uint64_t> v3) {
// CHECK-NEXT:   constexpr int64_t c16_i64 = 16;
// CHECK-NEXT:   constexpr int64_t c0_i64 = 0;
// CHECK-NEXT:   constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:   constexpr int64_t c1_i64 = 1;
// CHECK-NEXT:   constexpr bool c0_i1 = false;
// CHECK-NEXT:   constexpr int8_t c0_i8 = 0;
// CHECK-NEXT:   AscendC::FixpipeParamsV220 v4{c16_i64, c16_i64, c0_i64, c0_i64, c0_i32, c0_i64, c1_i64, c0_i64, c0_i64, c0_i1, c0_i8, c0_i1};
// CHECK-NEXT:   AscendC::FixpipeConfig v5{static_cast<AscendC::CO2Layout>(c1_i64)};
// CHECK-NEXT:   AscendC::Fixpipe<float, float, v5>(v1, v2, v4);
// CHECK-NEXT:   AscendC::Fixpipe<float, float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_fixpipe(%dst: !ascendc.global_tensor<f32>, %src: !ascendc.local_tensor<f32>, %cbuf: !ascendc.local_tensor<ui64>)
{
  %c16_i64 = arith.constant 16 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c0_i1 = arith.constant false
  %c0_i8 = arith.constant 0 : i8
  %fixpipe_params = ascendc.construct !ascendc.fixpipe_params_v220(%c16_i64, %c16_i64, %c0_i64, %c0_i64, %c0_i32, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c0_i1, %c0_i8, %c0_i1) : i64, i64, i64, i64, i32, i64, i64, i64, i64, i1, i8, i1
  %fixpipe_config = ascendc.construct !ascendc.fixpipe_config(%c1_i64) [!ascendc.co2_layout] :i64
  ascendc.fixpipe %dst, %src, %fixpipe_params, %fixpipe_config : !ascendc.global_tensor<f32>, !ascendc.local_tensor<f32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
  ascendc.fixpipe_with_workspace %dst, %src, %cbuf, %fixpipe_params, %fixpipe_config : !ascendc.global_tensor<f32>, !ascendc.local_tensor<f32>, !ascendc.local_tensor<ui64>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
  return
}

// CHECK-LABEL: void emit_set_fixpipe_pre_quant_flag() {
// CHECK-NEXT:    constexpr int64_t c11_i64 = 11;
// CHECK-NEXT:    AscendC::SetFixpipePreQuantFlag(c11_i64);
// CHECK-NEXT:    return;
// CHECK-NEXT: }
func.func @emit_set_fixpipe_pre_quant_flag() {
  %config = arith.constant 11 : i64
  ascendc.set_fixpipe_pre_quant_flag %config : i64
  return
}
