// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_data_copy_vld_micro(AscendC::MicroAPI::RegTensor<float> v1, float* v2) {
// CHECK-NEXT:  AscendC::MicroAPI::DataCopy(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_data_copy_vld_micro(%dstReg: !ascendc.reg_tensor<f32>, %src: memref<?xf32>) {
  ascendc.data_copy_vld_micro %dstReg, %src : !ascendc.reg_tensor<f32>, memref<?xf32>
  return
}

// CHECK-LABEL:void emit_data_copy_vst_micro(float* v1, AscendC::MicroAPI::RegTensor<float> v2, AscendC::MicroAPI::MaskReg v3) {
// CHECK-NEXT:  AscendC::MicroAPI::DataCopy(v1, v2, v3);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_data_copy_vst_micro(%dst: memref<?xf32>, %srcReg: !ascendc.reg_tensor<f32>, %maskReg: !ascendc.mask_reg) {
  ascendc.data_copy_vst_micro %dst, %srcReg, %maskReg : memref<?xf32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_update_mask(int32_t* v1) {
// CHECK:       AscendC::MicroAPI::MaskReg v2 = AscendC::MicroAPI::UpdateMask<float>(*v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_update_mask(%count: memref<?xi32>) {
  %mask = ascendc.update_mask f32, %count : memref<?xi32>
  return
}

// CHECK-LABEL:void emit_reg_tensor() {
// CHECK-NEXT:  AscendC::MicroAPI::RegTensor<float> v1;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_reg_tensor() {
  %reg = ascendc.reg_tensor : !ascendc.reg_tensor<f32>
  return
}

// CHECK-LABEL:void emit_duplicate_scalar_micro(AscendC::MicroAPI::RegTensor<float> v1, float v2) {
// CHECK-NEXT:  Duplicate(v1, v2);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_duplicate_scalar_micro(%dstReg: !ascendc.reg_tensor<f32>, %scalar: f32) {
  ascendc.duplicate %dstReg, %scalar : !ascendc.reg_tensor<f32>, f32
  return
}

// CHECK-LABEL:void emit_get_vec_len() {
// CHECK-NEXT:  int32_t v1 = AscendC::GetVecLen();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_vec_len() {
  %len = ascendc.get_vec_len : i32
  return
}

// CHECK-LABEL:void emit_local_mem_bar() {
// CHECK-NEXT:  AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_local_mem_bar() {
  ascendc.local_mem_bar VEC_STORE, VEC_LOAD
  return
}

// CHECK-LABEL:void emit_select_micro(AscendC::MicroAPI::RegTensor<float> v1, AscendC::MicroAPI::RegTensor<float> v2, AscendC::MicroAPI::RegTensor<float> v3, AscendC::MicroAPI::MaskReg v4) {
// CHECK-NEXT:  AscendC::MicroAPI::Select(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_select_micro(%dstReg: !ascendc.reg_tensor<f32>, %src0Reg: !ascendc.reg_tensor<f32>, %src1Reg: !ascendc.reg_tensor<f32>, %maskReg: !ascendc.mask_reg) {
  ascendc.select_micro %dstReg, %src0Reg, %src1Reg, %maskReg : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_create_mask() {
// CHECK-NEXT:  AscendC::MicroAPI::MaskReg v1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_create_mask() {
  %mask = ascendc.create_mask f32, ALL : !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_binary_micro(AscendC::MicroAPI::RegTensor<float> v1, AscendC::MicroAPI::RegTensor<float> v2, AscendC::MicroAPI::RegTensor<float> v3, AscendC::MicroAPI::MaskReg v4) {
// CHECK-NEXT:  AscendC::MicroAPI::Add(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::And(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Div(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::FusedAbsSub(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::FusedExpSub(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::FusedMulDstAdd(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Sub(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Max(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Min(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Mul(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::MulAddDst(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Or(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Prelu(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Xor(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_binary_micro(%dst: !ascendc.reg_tensor<f32>, %src0: !ascendc.reg_tensor<f32>, %src1: !ascendc.reg_tensor<f32>, %mask: !ascendc.mask_reg) {
  ascendc.add_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.and_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.div_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.fused_abs_sub_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.fused_exp_sub_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.fused_mul_dst_add_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.sub_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.max_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.min_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.mul_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.mul_add_dst_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.or_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.prelu_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.xor_micro %dst, %src0, %src1, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_unary_micro(AscendC::MicroAPI::RegTensor<float> v1, AscendC::MicroAPI::RegTensor<float> v2, AscendC::MicroAPI::MaskReg v3) {
// CHECK-NEXT:  AscendC::MicroAPI::Abs(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Exp(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Ln(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Log(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Log10(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::MaskNot(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Neg(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Not(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Relu(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Sqrt(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::ReduceMax(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::ReduceMin(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::ReduceSum(v1, v2, v3);
// CHECK-NEXT:  AscendC::MicroAPI::Duplicate(v1, v2, v3);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_unary_micro(%dst: !ascendc.reg_tensor<f32>, %src: !ascendc.reg_tensor<f32>, %mask: !ascendc.mask_reg) {
  ascendc.abs_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.exp_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.ln_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.log_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.log10_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.mask_not_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.neg_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.not_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.relu_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.sqrt_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_max_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_min_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.reduce_sum_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  ascendc.duplicate_micro %dst, %src, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  return
}

// CHECK-LABEL:void emit_vecscalar_micro(AscendC::MicroAPI::RegTensor<float> v1, AscendC::MicroAPI::RegTensor<float> v2, float v3, AscendC::MicroAPI::MaskReg v4) {
// CHECK-NEXT:  AscendC::MicroAPI::Adds(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Muls(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Maxs(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::Mins(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::LeakyRelu(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::ShiftLefts(v1, v2, v3, v4);
// CHECK-NEXT:  AscendC::MicroAPI::ShiftRights(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_vecscalar_micro(%dst: !ascendc.reg_tensor<f32>, %src: !ascendc.reg_tensor<f32>, %scalar: f32, %mask: !ascendc.mask_reg) {
  ascendc.adds_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.muls_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.maxs_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.mins_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.leaky_relu_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.shift_lefts_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  ascendc.shift_rights_micro %dst, %src, %scalar, %mask : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
  return
}
