// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s
 
// CHECK-LABEL: void emit_get_sort_len(uint32_t v1) {
// CHECK-NEXT:  AscendC::GetSortLen<float>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_get_sort_len(%arg0: ui32) {
 	%0 = ascendc.get_sort_len %arg0 : ui32
 	return
}

// CHECK-LABEL: void emit_get_sort_offset(uint32_t v1) {
// CHECK-NEXT:  AscendC::GetSortOffset<float>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_get_sort_offset(%arg0: ui32) {
  %0 = ascendc.get_sort_offset %arg0 : ui32
 	return
}

// CHECK-LABEL:void emit_mrg_sort(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, AscendC::LocalTensor<float> v5, uint16_t v6, uint16_t v7, uint16_t v8, uint16_t v9, uint32_t v10, uint32_t v11, uint32_t v12, uint32_t v13, int16_t v14, int32_t v15) {
// CHECK-NEXT:   AscendC::MrgSortSrcList<float> v16{v2, v3, v4, v5};
// CHECK-NEXT:   uint16_t v1_element_count_list_0[] = {v6, v7, v8, v9};
// CHECK-NEXT:   uint32_t v1_sorted_num_0[] = {v10, v11, v12, v13};
// CHECK-NEXT:   AscendC::MrgSort<float, 0>(v1, v16, v1_element_count_list_0, v1_sorted_num_0, v14, v15);
// CHECK-NEXT:   constexpr int8_t c1_i8 = 1;
// CHECK-NEXT:   uint16_t v1_element_count_list_1[] = {v6, v7, v8, v9};
// CHECK-NEXT:   uint32_t v1_sorted_num_1[] = {v10, v11, v12, v13};
// CHECK-NEXT:   AscendC::MrgSort<float, 1>(v1, v16, v1_element_count_list_1, v1_sorted_num_1, v14, v15);
// CHECK-NEXT:   constexpr uint16_t v17 = 16;
// CHECK-NEXT:   constexpr bool c0_i1 = false;
// CHECK-NEXT:   constexpr int16_t c15_i16 = 15;
// CHECK-NEXT:   constexpr int16_t c1_i16 = 1;
// CHECK-NEXT:   constexpr uint32_t c0_idx = 0;
// CHECK-NEXT:   constexpr uint32_t c1_idx = 1;
// CHECK-NEXT:   constexpr uint32_t c2_idx = 2;
// CHECK-NEXT:   constexpr uint32_t c3_idx = 3;
// CHECK-NEXT:   uint16_t v18[4];
// CHECK-NEXT:   v18[c0_idx] = v17;
// CHECK-NEXT:   v18[c1_idx] = v17;
// CHECK-NEXT:   v18[c2_idx] = v17;
// CHECK-NEXT:   v18[c3_idx] = v17;
// CHECK-NEXT:   AscendC::MrgSort4Info v19{v18, c0_i1, c15_i16, c1_i16};
// CHECK-NEXT:   AscendC::MrgSort(v1, v16, v19);
// CHECK-NEXT:   uint16_t v20;
// CHECK-NEXT:   uint16_t v21;
// CHECK-NEXT:   uint16_t v22;
// CHECK-NEXT:   uint16_t v23;
// CHECK-NEXT:   AscendC::GetMrgSortResult(v20, v21, v22, v23);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_mrg_sort(%dst: !ascendc.local_tensor<1024xf32>,
                                    %src1: !ascendc.local_tensor<512xf32>,
                                    %src2: !ascendc.local_tensor<512xf32>,
                                    %src3: !ascendc.local_tensor<512xf32>,
                                    %src4: !ascendc.local_tensor<512xf32>,
                                    %elem_count1: ui16, %elem_count2: ui16, %elem_count3: ui16, %elem_count4: ui16,
                                    %sorted_num1: ui32, %sorted_num2: ui32, %sorted_num3: ui32, %sorted_num4: ui32,
                                    %valid_bit: i16, %repeat_time: i32) {
  %mrg_sort_src_list = ascendc.construct !ascendc.mrg_sort_src_list<f32>(%src1, %src2, %src3, %src4) [!ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>] : !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>
  ascendc.mrg_sort %dst, %mrg_sort_src_list, %elem_count1, %elem_count2, %elem_count3, %elem_count4,
                  %sorted_num1, %sorted_num2, %sorted_num3, %sorted_num4, %valid_bit, %repeat_time
                  {operandSegmentSizes = array<i32: 1, 1, 4, 4, 1, 1>}:
                  !ascendc.local_tensor<1024xf32>, !ascendc.mrg_sort_src_list<f32>, ui16, ui16, ui16, ui16,
                  ui32, ui32, ui32, ui32, i16, i32
  %is_exhausted_suspension = arith.constant 1 : i8
  ascendc.mrg_sort %dst, %mrg_sort_src_list, %elem_count1, %elem_count2, %elem_count3, %elem_count4,
                  %sorted_num1, %sorted_num2, %sorted_num3, %sorted_num4, %valid_bit, %repeat_time
                  {operandSegmentSizes = array<i32: 1, 1, 4, 4, 1, 1>, isExhaustedSuspension}:
                  !ascendc.local_tensor<1024xf32>, !ascendc.mrg_sort_src_list<f32>, ui16, ui16, ui16, ui16,
                  ui32, ui32, ui32, ui32, i16, i32
  %c16_ui16 = "emitc.constant"() <{value = 16 : ui16}> : () -> ui16
  %c0_i1 = arith.constant 0 : i1
  %c15 = arith.constant 15 : i16
  %c1 = arith.constant 1 : i16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %alloca = memref.alloca() : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c0_idx] : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c1_idx] : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c2_idx] : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c3_idx] : memref<4xui16>
  %mrg_sort4_info = ascendc.construct !ascendc.mrg_sort4_info(%alloca, %c0_i1, %c15, %c1) [memref<4xui16>, i1, i16, i16] : memref<4xui16>, i1, i16, i16
  ascendc.mrg_sort_with_info %dst, %mrg_sort_src_list, %mrg_sort4_info : !ascendc.local_tensor<1024xf32>, !ascendc.mrg_sort_src_list<f32>, !ascendc.mrg_sort4_info
  %mrg1, %mrg2 , %mrg3, %mrg4 = ascendc.get_mrg_sort_result  : ui16, ui16, ui16, ui16
  return
}

// CHECK-LABEL:void emit_mrg_sort4(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, AscendC::LocalTensor<float> v5) {
// CHECK-NEXT:   AscendC::MrgSortSrcList<float> v6{v2, v3, v4, v5};
// CHECK-NEXT:   constexpr uint16_t v7 = 16;
// CHECK-NEXT:   constexpr bool c0_i1 = false;
// CHECK-NEXT:   constexpr int16_t c15_i16 = 15;
// CHECK-NEXT:   constexpr int16_t c1_i16 = 1;
// CHECK-NEXT:   constexpr uint32_t c0_idx = 0;
// CHECK-NEXT:   constexpr uint32_t c1_idx = 1;
// CHECK-NEXT:   constexpr uint32_t c2_idx = 2;
// CHECK-NEXT:   constexpr uint32_t c3_idx = 3;
// CHECK-NEXT:   uint16_t v8[4];
// CHECK-NEXT:   v8[c0_idx] = v7;
// CHECK-NEXT:   v8[c1_idx] = v7;
// CHECK-NEXT:   v8[c2_idx] = v7;
// CHECK-NEXT:   v8[c3_idx] = v7;
// CHECK-NEXT:   AscendC::MrgSort4Info v9{v8, c0_i1, c15_i16, c1_i16};
// CHECK-NEXT:   AscendC::MrgSort4(v1, v6, v9);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_mrg_sort4(%dst: !ascendc.local_tensor<1024xf32>,
                             %src1: !ascendc.local_tensor<512xf32>,
                             %src2: !ascendc.local_tensor<512xf32>,
                             %src3: !ascendc.local_tensor<512xf32>,
                             %src4: !ascendc.local_tensor<512xf32>) {
  %mrg_sort_src_list = ascendc.construct !ascendc.mrg_sort_src_list<f32>(%src1, %src2, %src3, %src4) 
  [!ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>] : !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>
  %c16_ui16 = "emitc.constant"() <{value = 16 : ui16}> : () -> ui16
  %c0_i1 = arith.constant 0 : i1
  %c15 = arith.constant 15 : i16
  %c1 = arith.constant 1 : i16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %alloca = memref.alloca() : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c0_idx] : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c1_idx] : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c2_idx] : memref<4xui16>
  memref.store %c16_ui16, %alloca[%c3_idx] : memref<4xui16>
  %mrg_sort4_info = ascendc.construct !ascendc.mrg_sort4_info(%alloca, %c0_i1, %c15, %c1) [memref<4xui16>, i1, i16, i16] : memref<4xui16>, i1, i16, i16
  ascendc.mrg_sort4 %dst, %mrg_sort_src_list, %mrg_sort4_info : !ascendc.local_tensor<1024xf32>, !ascendc.mrg_sort_src_list<f32>, !ascendc.mrg_sort4_info
  return
}

// CHECK-LABEL: void emit_proposal_concat(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 256;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   constexpr int32_t c2_i32 = 2;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   AscendC::LocalTensor<half> v4 = AscendC::LocalTensor<half>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v5 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v2);
// CHECK-NEXT:   AscendC::ProposalConcat(v4, v5, c2_i32, c4_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }

func.func @emit_proposal_concat(%arg0: memref<?xui64, 22>) {
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %c256 = "emitc.constant"() <{value = 256 : ui32}> : () -> ui32
  %c0 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %dst = ascendc.local_tensor_v2 vecout, %c0, %c256 : !ascendc.local_tensor<*xf16>
  %src = ascendc.local_tensor_v2 vecin, %c0, %c256 : !ascendc.local_tensor<*xf16>
  ascendc.proposal_concat %dst, %src, %c2_i32, %c4_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32
  return
}

// CHECK-LABEL: void emit_proposal_extract(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 256;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   constexpr int32_t c2_i32 = 2;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   AscendC::LocalTensor<half> v4 = AscendC::LocalTensor<half>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v5 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v2);
// CHECK-NEXT:   AscendC::ProposalExtract(v4, v5, c2_i32, c4_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }

func.func @emit_proposal_extract(%arg0: memref<?xui64, 22>) {
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %c256 = "emitc.constant"() <{value = 256 : ui32}> : () -> ui32
  %c0 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %dst = ascendc.local_tensor_v2 vecout, %c0, %c256 : !ascendc.local_tensor<*xf16>
  %src = ascendc.local_tensor_v2 vecin, %c0, %c256 : !ascendc.local_tensor<*xf16>
  ascendc.proposal_extract %dst, %src, %c2_i32, %c4_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32
  return
}

// CHECK-LABEL:void emit_rp_sort16(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3) {
// CHECK-NEXT:   AscendC::RpSort16(v1, v2, v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_rp_sort16(%dst: !ascendc.local_tensor<512xf32>,
                         %src: !ascendc.local_tensor<512xf32>,
                         %repeat_time: i32) {
  ascendc.rp_sort16 %dst, %src, %repeat_time : !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xf32>, i32
  return
}

// CHECK-LABEL:void emit_sort(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<uint32_t> v3, AscendC::LocalTensor<float> v4, int32_t v5) {
// CHECK-NEXT:   AscendC::Sort<float, 0>(v1, v2, v3, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_sort(%dst: !ascendc.local_tensor<1024xf32>,
                    %concat: !ascendc.local_tensor<512xf32>,
                    %index: !ascendc.local_tensor<512xui32>,
                    %tmp: !ascendc.local_tensor<512xf32>,
                    %repeat_time: i32) {
  ascendc.sort %dst, %concat, %index, %tmp, %repeat_time : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xui32>, !ascendc.local_tensor<512xf32>, i32
  return
}

// CHECK-LABEL:void emit_sort32(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<uint32_t> v3, int32_t v4) {
// CHECK-NEXT:   AscendC::Sort32(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_sort32(%dst: !ascendc.local_tensor<1024xf32>,
                      %src0: !ascendc.local_tensor<512xf32>,
                      %src1: !ascendc.local_tensor<512xui32>,
                      %repeat_time: i32) {
  ascendc.sort32 %dst, %src0, %src1, %repeat_time : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<512xf32>, !ascendc.local_tensor<512xui32>, i32
  return
}

