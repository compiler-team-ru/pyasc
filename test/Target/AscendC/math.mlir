// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK:void emit_unary_ops(half v1) {
// CHECK-NEXT:  half v2 = (v1 > static_cast<half>(0)) ? v1 : -v1;
// CHECK-NEXT:  half v3 = AscendC::Cos(v2);
// CHECK-NEXT:  half v4 = AscendC::Log2(v2);
// CHECK-NEXT:  half v5 = AscendC::Sin(v2);
// CHECK-NEXT:  half v6 = AscendC::Ceil(v2);
// CHECK-NEXT:  half v7 = AscendC::Floor(v2);
// CHECK-NEXT:  half v8 = AscendC::Exp(v2 * Log(2));
// CHECK-NEXT:  half v9 = AscendC::Rsqrt(v2);
// CHECK-NEXT:  half v10 = AscendC::Round(v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_unary_ops(%arg0 : f16) {
  %0 = math.absf %arg0 : f16
  %1 = math.cos %0 : f16
  %2 = math.log2 %0 : f16
  %3 = math.sin %0 : f16
  %4 = math.ceil %0 : f16
  %5 = math.floor %0 : f16
  %6 = math.exp2 %0 : f16
  %7 = math.rsqrt %0 : f16
  %8 = math.round %0 : f16
  return
}

// CHECK:void emit_binary_ops(float v1, float v2) {
// CHECK-NEXT:  float v3 = ((v1 < 0 && v2 > 0) || (v1 > 0 && v2 < 0)) ? -v1 : v1;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_binary_ops(%arg0: f32, %arg1: f32) {
  %0 = math.copysign %arg0, %arg1 : f32
  return
}
//CHECK:void emit_ternary_ops(float v1, float v2, float v3) {
// CHECK-NEXT:  float v4 = v1 * v2 + v3;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_ternary_ops(%arg0: f32, %arg1: f32, %arg2: f32) {
  %0 = math.fma %arg0, %arg1, %arg2 : f32
  return
}

// CHECK-LABEL:void emit_unary_math_scalar(float v1) {
// CHECK-NEXT:  float v2 = sqrt(v1);
// CHECK-NEXT:  float v3 = AscendC::Exp(v1);
// CHECK-NEXT:  float v4 = AscendC::Log(v1);
// CHECK-NEXT:  float v5 = AscendC::Erf(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_unary_math_scalar(%arg0: f32) {
  %0 = math.sqrt %arg0 : f32
  %1 = math.exp %arg0 : f32
  %2 = math.log %arg0 : f32
  %3 = math.erf %arg0 : f32
  return
}

// CHECK-LABEL:void emit_atan2(float v1, float v2) {
// CHECK-NEXT:  float v3 = AscendC::Atan2(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_atan2(%arg0: f32, %arg1: f32) {
  %0 = math.atan2 %arg0, %arg1 : f32
  return
}
