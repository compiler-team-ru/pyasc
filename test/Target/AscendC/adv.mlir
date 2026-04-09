// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_softmax(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, AscendC::LocalTensor<half> v3, AscendC::LocalTensor<half> v4, SoftMaxTiling v5, AscendC::SoftMaxShapeInfo v6) {
// CHECK-NEXT:  AscendC::SimpleSoftMax<half, 0, 0, 0>(v1, v2, v3, v4, v5, v6);
// CHECK-NEXT:  AscendC::SoftMax<half, 1, 1>(v1, v2, v3, v4, v5, v6);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_softmax(%arg0: !ascendc.local_tensor<*xf16>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.local_tensor<*xf16>, %arg4: !ascendc.softmax_tiling, %arg5: !ascendc.softmax_shape_info) {
  ascendc.simple_softmax %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
  ascendc.softmax %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {basicBlock, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>, reuseSource} : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
  return
}

// CHECK-LABEL:void emit_unary_math_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, bool v5) {
// CHECK-NEXT:   AscendC::Acosh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Acos<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Asinh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Asin<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Atanh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Atan<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Ceil<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Cosh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Cos<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Digamma<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Erfc<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Erf<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Floor<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Frac<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Lgamma<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Log<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Round<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Sign<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Sinh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Sin<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Tanh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Tan<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Trunc<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_unary_math_ops(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %calCount : i32, %isReuseSource : i1) {
  ascendc.acosh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.acos %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.asinh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.asin %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.atanh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.atan %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.ceil %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.cosh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.cos %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.digamma %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.erfc %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.erf %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.floor %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.frac %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.lgamma %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.log %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.round %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.sign %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.sinh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.sin %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.tanh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.tan %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.trunc %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  return
}

// CHECK-LABEL:void emit_unary_math_ops_test_params(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, bool v5) {
// CHECK-NEXT:   AscendC::Acosh<float, v5>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Acosh<float, v5>(v1, v2);
// CHECK-NEXT:   AscendC::Acosh<float, v5>(v1, v2, v3);
// CHECK-NEXT:   AscendC::Acosh<float, v5>(v1, v2, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_unary_math_ops_test_params(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %calCount : i32, %isReuseSource : i1) {
  ascendc.acosh %dst, %src, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.acosh %dst, %src, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i1
  ascendc.acosh %dst, %src, %sharedTmpBuffer, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i1
  ascendc.acosh %dst, %src, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  return
}

// CHECK-LABEL:void emit_binary_math_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, int32_t v5, bool v6) {
// CHECK-NEXT:   AscendC::Power<float, v6>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   AscendC::Xor<float, v6>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_binary_math_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %calCount : i32, %isReuseSource : i1) {
  ascendc.power %dst, %src0, %src1, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.xor   %dst, %src0, %src1, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  return
}

// CHECK-LABEL:void emit_binary_math_ops_test_params(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, int32_t v5, bool v6) {
// CHECK-NEXT:   AscendC::Power<float, v6>(v1, v2, v3);
// CHECK-NEXT:   AscendC::Power<float, v6>(v1, v2, v3, v5);
// CHECK-NEXT:   AscendC::Power<float, v6>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Power<float, v6>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_binary_math_ops_test_params(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %calCount : i32, %isReuseSource : i1) {
  ascendc.power %dst, %src0, %src1, %isReuseSource                              {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i1
  ascendc.power %dst, %src0, %src1, %calCount, %isReuseSource                   {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.power %dst, %src0, %src1, %sharedTmpBuffer, %isReuseSource            {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i1
  ascendc.power %dst, %src0, %src1, %sharedTmpBuffer, %calCount, %isReuseSource {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i1
  return
}

// CHECK-LABEL:void emit_clamp(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, int32_t v5, bool v6) {
// CHECK-NEXT:   AscendC::ClampMax<float, v6>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   AscendC::ClampMax<float, v6>(v1, v2, v4, v5);
// CHECK-NEXT:   AscendC::ClampMin<float, v6>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   AscendC::ClampMin<float, v6>(v1, v2, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_clamp(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %scalar : i32, %calCount : i32, %isReuseSource : i1) {
  ascendc.clamp_max %dst, %src, %sharedTmpBuffer, %scalar, %calCount, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i1
  ascendc.clamp_max %dst, %src, %scalar, %calCount, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i1
  ascendc.clamp_min %dst, %src, %sharedTmpBuffer, %scalar, %calCount, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i1
  ascendc.clamp_min %dst, %src, %scalar, %calCount, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i1
  return
}

// CHECK-LABEL:void emit_cumsum(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, int32_t v5, int32_t v6, int32_t v7) {
// CHECK-NEXT:   AscendC::CumSum(v1, v2, v3, v4, AscendC::CumSumInfo(v5, v6, v7));
// CHECK-NEXT:   AscendC::CumSum(v1, v2, v3, AscendC::CumSumInfo(v5, v6, v7));
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_cumsum(%dst: !ascendc.local_tensor<1024xf32>, %lastRaw: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %lastAxis : i32, %reuseSource : i32, %outputLastRow : i32) {
  ascendc.cumsum %dst, %lastRaw, %src, %sharedTmpBuffer, %lastAxis, %reuseSource, %outputLastRow :  !ascendc.local_tensor<1024xf32>,  !ascendc.local_tensor<1024xf32>,  !ascendc.local_tensor<1024xf32>,  !ascendc.local_tensor<1024xf32>, i32, i32, i32
  ascendc.cumsum %dst, %lastRaw, %src, %lastAxis, %reuseSource, %outputLastRow :  !ascendc.local_tensor<1024xf32>,  !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_axpy(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, AscendC::LocalTensor<float> v4, int32_t v5, bool v6) {
// CHECK-NEXT:   AscendC::Axpy<float, float, v6>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   AscendC::Axpy<float, float, v6>(v1, v2, v3, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_axpy(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %scalar : i32, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %calCount : i32, %isReuseSource : i1) {
  ascendc.axpy %dst, %src, %scalar, %sharedTmpBuffer, %calCount, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, !ascendc.local_tensor<1024xf32>, i32, i1
  ascendc.axpy %dst, %src, %scalar, %calCount, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i1
  return
}

// CHECK-LABEL:void emit_exp(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, uint8_t v4, AscendC::LocalTensor<float> v5, bool v6) {
// CHECK-NEXT:   AscendC::Exp<float, v4, v6>(v1, v2, v5, v3);
// CHECK-NEXT:   AscendC::Exp<float, v4, v6>(v1, v2, v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_exp(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %calCount : i32, %taylorExpandLevel: ui8, %sharedTmpBuffer : !ascendc.local_tensor<1024xf32>, %isReuseSource : i1) {
  ascendc.exp %dst, %src, %calCount, %taylorExpandLevel, %sharedTmpBuffer, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui8, !ascendc.local_tensor<1024xf32>, i1
  ascendc.exp %dst, %src, %calCount, %taylorExpandLevel, %isReuseSource : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui8, i1
  return
}

// CHECK-LABEL:void emit_cube_tiling_operator_equal(TCubeTiling* v1, TCubeTiling* v2) {
// CHECK-NEXT:  constexpr uint32_t c0_idx = 0;
// CHECK-NEXT:  TCubeTiling v3 = v2[c0_idx];
// CHECK-NEXT:  constexpr uint32_t c0_idx = 0;
// CHECK-NEXT:  v1[c0_idx] = v3;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_cube_tiling_operator_equal(%arg0: memref<?x!ascendc.cube_tiling>, %arg1: memref<?x!ascendc.cube_tiling>) {
  %c0 = arith.constant 0 : index
  %0 = memref.load %arg1[%c0] : memref<?x!ascendc.cube_tiling>
  %c0_0 = arith.constant 0 : index
  memref.store %0, %arg0[%c0_0] : memref<?x!ascendc.cube_tiling>
  return
}

// CHECK-LABEL: void emit_concat(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 128;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   constexpr uint32_t v4 = 256;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   AscendC::LocalTensor<half> v5 = AscendC::LocalTensor<half>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v6 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v7 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v4);
// CHECK-NEXT:   AscendC::Concat(v5, v6, v7, c4_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_concat(%arg0: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = "emitc.constant"() <{value = 128 : ui32}> : () -> ui32
  %1 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %2 = "emitc.constant"() <{value = 256 : ui32}> : () -> ui32
  %c4_i32 = arith.constant 4 : i32
  %3 = ascendc.local_tensor_v2 vecout, %1, %0 : !ascendc.local_tensor<*xf16>
  %4 = ascendc.local_tensor_v2 vecin, %1, %0 : !ascendc.local_tensor<*xf16>
  %5 = ascendc.local_tensor_v2 vecin, %1, %2 : !ascendc.local_tensor<*xf16>
  ascendc.concat %3, %4, %5, %c4_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
  return
}

// CHECK-LABEL: void emit_extract(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 128;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   constexpr uint32_t v4 = 256;
// CHECK-NEXT:   constexpr int32_t c8_i32 = 8;
// CHECK-NEXT:   AscendC::LocalTensor<half> v5 = AscendC::LocalTensor<half>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<uint32_t> v6 = AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v7 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v4);
// CHECK-NEXT:   AscendC::Extract(v5, v6, v7, c8_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_extract(%arg0: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = "emitc.constant"() <{value = 128 : ui32}> : () -> ui32
  %1 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %2 = "emitc.constant"() <{value = 256 : ui32}> : () -> ui32
  %c8_i32 = arith.constant 8 : i32
  %3 = ascendc.local_tensor_v2 vecout, %1, %0 : !ascendc.local_tensor<*xf16>
  %4 = ascendc.local_tensor_v2 vecout, %1, %0 : !ascendc.local_tensor<*xui32>
  %5 = ascendc.local_tensor_v2 vecin, %1, %2 : !ascendc.local_tensor<*xf16>
  ascendc.extract %3, %4, %5, %c8_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xui32>, !ascendc.local_tensor<*xf16>, i32
  return
}
