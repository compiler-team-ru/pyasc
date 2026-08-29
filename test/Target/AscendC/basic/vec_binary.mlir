// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_vector_binary_l0_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, AscendC::BinaryRepeatParams v5) {
// CHECK-NEXT:   AscendC::AddDeqRelu<float, float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::AddRelu<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::And<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Div<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::FusedMulAdd<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::FusedMulAddRelu<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Max<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Min<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Mul<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::MulAddDst<float, float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::MulCast<float, float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Or<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Sub<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::SubRelu<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l0_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>, %c1_i32 : i32, %params: !ascendc.binary_repeat_params) {
  ascendc.add_deq_relu_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.add_relu_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.and_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.div_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.fused_mul_add_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.fused_mul_add_relu_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.max_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.min_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.mul_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.mul_add_dst_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.mul_cast_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.or_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.sub_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.sub_relu_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_binary_l1_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, AscendC::BinaryRepeatParams v5, uint64_t v6, uint64_t v7) {
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::AddDeqRelu<float, float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::AddRelu<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::And<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Div<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::FusedMulAdd<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::FusedMulAddRelu<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Max<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Min<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Mul<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::MulAddDst<float, float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::MulCast<float, float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Or<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Sub<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::SubRelu<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l1_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>, %c1_i32 : i32, %params: !ascendc.binary_repeat_params, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.add_deq_relu_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.add_relu_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.and_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.div_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.fused_mul_add_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.fused_mul_add_relu_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.max_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.min_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.mul_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.mul_add_dst_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.mul_cast_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.or_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.sub_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.sub_relu_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_binary_l2_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4) {
// CHECK-NEXT:   AscendC::Add(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::AddDeqRelu(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::AddRelu(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::And(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Div(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::FusedAbsSub(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::FusedExpSub(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::FusedMulAdd(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::FusedMulAddRelu(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Max(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Min(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Mul(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::MulAddDst(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::MulCast(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Or(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Prelu(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Sub(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::SubRelu(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l2_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1 : !ascendc.local_tensor<1024xf32>, %calCount_i32 : i32) {
  ascendc.add_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.add_deq_relu_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.add_relu_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.and_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.div_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.fused_abs_sub_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.fused_exp_sub_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.fused_mul_add_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.fused_mul_add_relu_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.max_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.min_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.mul_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.mul_add_dst_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.mul_cast_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.or_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.prelu_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.sub_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.sub_relu_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  return
}

// CHECK-LABEL:void emit_vector_binary_l3_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3) {
// CHECK-NEXT:   v1 = v2.operator+(v3);
// CHECK-NEXT:   v1 = v2.operator/(v3);
// CHECK-NEXT:   v1 = v2.operator*(v3);
// CHECK-NEXT:   v1 = v2.operator-(v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l3_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>) {
  ascendc.add_l3 %dst, %src0, %src1 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>
  ascendc.div_l3 %dst, %src0, %src1 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>
  ascendc.mul_l3 %dst, %src0, %src1 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>
  ascendc.sub_l3 %dst, %src0, %src1 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>
  return
}

// CHECK-LABEL:void emit_bilinear_interpolation_l0(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, int64_t v5, int8_t v6, int8_t v7, int16_t v8, int16_t v9, int8_t v10, AscendC::LocalTensor<float> v11, uint64_t v12, uint64_t v13) {
// CHECK-NEXT:  AscendC::BilinearInterpolation(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v12, v13};
// CHECK-NEXT:  AscendC::BilinearInterpolation(v1, v2, v3, v4, v1_mask_list0, v6, v7, v8, v9, v10, v11);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_bilinear_interpolation_l0(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.local_tensor<*xf32>, %arg3: !ascendc.local_tensor<*xf32>, %arg4: i64, %arg5: i8, %arg6: i8, %arg7: i16, %arg8: i16, %arg9: i8, %arg10: !ascendc.local_tensor<*xf32>, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.bilinearInterpolation_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i64, i8, i8, i16, i16, i8, !ascendc.local_tensor<*xf32>
  ascendc.bilinearInterpolation_l1 %arg0, %arg1, %arg2, %arg3, %maskArray1_0, %maskArray1_1, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, ui64, ui64, i8, i8, i16, i16, i8, !ascendc.local_tensor<*xf32>
  return
}
