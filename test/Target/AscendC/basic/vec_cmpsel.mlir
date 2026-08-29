// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_compare(AscendC::LocalTensor<int8_t> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, int8_t v5, AscendC::BinaryRepeatParams v6, uint64_t v7) {
// CHECK-NEXT:  AscendC::Compare(v1, v2, v3, AscendC::CMPMODE::LT, v4, v5, v6);
// CHECK-NEXT:  uint64_t v1_mask_list[] = {v7, v7};
// CHECK-NEXT:  AscendC::Compare(v1, v2, v3, AscendC::CMPMODE::LT, v1_mask_list, v5, v6);
// CHECK-NEXT:  AscendC::Compare(v1, v2, v3, AscendC::CMPMODE::LT, v4);
// CHECK-NEXT:  AscendC::Compare(v2, v3, AscendC::CMPMODE::LT, v4, v6);
// CHECK-NEXT:  uint64_t v2_mask_list[] = {v7, v7};
// CHECK-NEXT:  AscendC::Compare(v2, v3, AscendC::CMPMODE::LT, v2_mask_list, v6);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_compare(%arg0: !ascendc.local_tensor<i8>, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.local_tensor<*xf32>, %c1_i32 : i32, %c2_i8 : i8, %params: !ascendc.binary_repeat_params, %c3_ui64 : ui64) {
  "ascendc.compare_l0"(%arg0, %arg1, %arg2, %c1_i32, %c2_i8, %params){cmpMode = 0} : (!ascendc.local_tensor<i8>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32, i8, !ascendc.binary_repeat_params) -> ()
  "ascendc.compare_l1"(%arg0, %arg1, %arg2, %c3_ui64, %c3_ui64, %c2_i8, %params){cmpMode = 0} : (!ascendc.local_tensor<i8>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, ui64, ui64, i8, !ascendc.binary_repeat_params) -> ()
  "ascendc.compare_l2"(%arg0, %arg1, %arg2, %c1_i32){cmpMode = 0} : (!ascendc.local_tensor<i8>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32) -> ()
  "ascendc.compare_r_l0"(%arg1, %arg2, %c1_i32, %params){cmpMode = 0} : (!ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32, !ascendc.binary_repeat_params) -> ()
  "ascendc.compare_r_l1"(%arg1, %arg2, %c3_ui64, %c3_ui64, %params){cmpMode = 0} : (!ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, ui64, ui64, !ascendc.binary_repeat_params) -> ()
  return
}

// CHECK-LABEL:void emit_compare_scalar(AscendC::LocalTensor<int8_t> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4, int8_t v5, AscendC::UnaryRepeatParams v6, uint64_t v7) {
// CHECK-NEXT:  AscendC::CompareScalar(v1, v2, v3, AscendC::CMPMODE::LT, v4, v5, v6);
// CHECK-NEXT:  uint64_t v1_mask_list[] = {v7, v7};
// CHECK-NEXT:  AscendC::CompareScalar(v1, v2, v3, AscendC::CMPMODE::LT, v1_mask_list, v5, v6);
// CHECK-NEXT:  AscendC::CompareScalar(v1, v2, v3, AscendC::CMPMODE::LT, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_compare_scalar(%arg0: !ascendc.local_tensor<i8>, %arg1: !ascendc.local_tensor<*xf32>, %scalar: i32, %c1_i32 : i32, %c2_i8 : i8, %params: !ascendc.unary_repeat_params, %c3_ui64 : ui64) {
  "ascendc.compare_scalar_l0"(%arg0, %arg1, %scalar, %c1_i32, %c2_i8, %params){cmpMode = 0} : (!ascendc.local_tensor<i8>, !ascendc.local_tensor<*xf32>, i32, i32, i8, !ascendc.unary_repeat_params) -> ()
  "ascendc.compare_scalar_l1"(%arg0, %arg1, %scalar, %c3_ui64, %c3_ui64, %c2_i8, %params){cmpMode = 0} : (!ascendc.local_tensor<i8>, !ascendc.local_tensor<*xf32>, i32, ui64, ui64, i8, !ascendc.unary_repeat_params) -> ()
  "ascendc.compare_scalar_l2"(%arg0, %arg1, %scalar, %c1_i32){cmpMode = 0} : (!ascendc.local_tensor<i8>, !ascendc.local_tensor<*xf32>, i32, i32) -> ()
  return
}

// CHECK-LABEL:void emit_get_cmp_mask(AscendC::LocalTensor<half> v1) {
// CHECK-NEXT:  AscendC::GetCmpMask(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_cmp_mask(%dst : !ascendc.local_tensor<*xf16>) {
  ascendc.get_cmp_mask %dst : !ascendc.local_tensor<*xf16>
  return
}

// CHECK-LABEL:void emit_set_cmp_mask(AscendC::LocalTensor<half> v1) {
// CHECK-NEXT:  AscendC::SetCmpMask(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_cmp_mask(%src : !ascendc.local_tensor<*xf16>) {
  ascendc.set_cmp_mask %src : !ascendc.local_tensor<*xf16>
  return
}

// CHECK-LABEL:void emit_select_l2_op(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<uint16_t> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, uint32_t v5, float v6) {
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_CMPMASK_SPR, v5);
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v6, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_select_l2_op(
  %dst : !ascendc.local_tensor<1024xf32>,
  %cond : !ascendc.local_tensor<32xui16>,
  %src0 : !ascendc.local_tensor<1024xf32>,
  %src1 : !ascendc.local_tensor<1024xf32>,
  %c0_ui32 : ui32,
  %f0 : f32
) {
  ascendc.select_l2 %dst, %cond, %src0, %src1, %c0_ui32 {selMode = 0 : i32}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui32
  ascendc.select_scalar_l2 %dst, %cond, %src0, %f0, %c0_ui32 {selMode = 1 : i32}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, f32, ui32
  return
}

// CHECK-LABEL:void emit_select_slice_scalar_op(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<uint16_t> v2, AscendC::LocalTensor<float> v3, float v4, uint8_t v5, AscendC::BinaryRepeatParams v6, uint64_t v7) {
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v7, v7};
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, v1_mask_list0, v5, v6);
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, v7, v5, v6);
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v5, v6);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_select_slice_scalar_op(
  %dst : !ascendc.local_tensor<1024xf32>,
  %cond : !ascendc.local_tensor<32xui16>,
  %src0 : !ascendc.local_tensor<1024xf32>,
  %f0 : f32,
  %c0_ui8 : ui8,
  %params : !ascendc.binary_repeat_params,
  %m0_ui64 : ui64
) {
  ascendc.select_scalar_l1 %dst, %cond, %src0, %f0, %m0_ui64, %m0_ui64, %c0_ui8, %params {selMode = 1 : i32, isSetMask}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, f32, ui64, ui64, ui8, !ascendc.binary_repeat_params
  ascendc.select_scalar_l0 %dst, %cond, %src0, %f0, %m0_ui64, %c0_ui8, %params {selMode = 1 : i32, isSetMask}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, f32, ui64, ui8, !ascendc.binary_repeat_params
  ascendc.select_scalar_regmask %dst, %cond, %src0, %c0_ui8, %params
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, ui8, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL:void emit_select_slice_tensor_op(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<uint16_t> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, uint8_t v5, AscendC::BinaryRepeatParams v6, uint64_t v7) {
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v7, v7};
// CHECK-NEXT:  AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_CMPMASK_SPR, v1_mask_list0, v5, v6);
// CHECK-NEXT:  AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_CMPMASK_SPR, v7, v5, v6);
// CHECK-NEXT:  AscendC::Select(v1, v3, v4, v5, v6);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_select_slice_tensor_op(
  %dst : !ascendc.local_tensor<1024xf32>,
  %cond : !ascendc.local_tensor<32xui16>,
  %src0 : !ascendc.local_tensor<1024xf32>,
  %src1 : !ascendc.local_tensor<1024xf32>,
  %c0_ui8 : ui8,
  %params : !ascendc.binary_repeat_params,
  %m0_ui64 : ui64
) {
  ascendc.select_l1 %dst, %cond, %src0, %src1, %m0_ui64, %m0_ui64, %c0_ui8, %params {selMode = 0 : i32, isSetMask}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, ui8, !ascendc.binary_repeat_params
  ascendc.select_l0 %dst, %cond, %src0, %src1, %m0_ui64, %c0_ui8, %params {selMode = 0 : i32, isSetMask}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui16>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui8, !ascendc.binary_repeat_params
  ascendc.select_regmask %dst, %src0, %src1, %c0_ui8, %params {selMode = 0 : i32}
    : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui8, !ascendc.binary_repeat_params
  return
}
