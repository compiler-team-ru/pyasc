// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_transpose(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<uint8_t> v3, AscendC::TransposeParamsExt v4) {
// CHECK-NEXT:   AscendC::Transpose(v1, v2);
// CHECK-NEXT:   AscendC::Transpose(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_transpose(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %tmp: !ascendc.local_tensor<1024xui8>, %params: !ascendc.transpose_params_ext) {
  ascendc.transpose %dst, %src : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>
  ascendc.transpose_ext %dst, %src, %tmp, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xui8>, !ascendc.transpose_params_ext
  return
}

// CHECK-LABEL:void emit_trans_data_to_5hd(AscendC::LocalTensor<float> v1, AscendC::TransDataTo5HDParams v2, uint32_t v3) {
// CHECK-NEXT:   constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:   AscendC::LocalTensor<uint64_t> v4;
// CHECK-NEXT:   AscendC::LocalTensor<uint64_t> v5;
// CHECK-NEXT:   AscendC::LocalTensor<float> v6 = v1[c0_i32]
// CHECK-NEXT:   uint64_t v7 = v6.GetPhyAddr(v3);
// CHECK-NEXT:   v4.SetValue(v3, v7);
// CHECK-NEXT:   v5.SetValue(v3, v7);
// CHECK-NEXT:   AscendC::TransDataTo5HD<float>(v4, v5, v2);
// CHECK-NEXT:   AscendC::LocalTensor<float> v6_list[] = {v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6};
// CHECK-NEXT:   AscendC::LocalTensor<float> v6_list[] = {v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6, v6};
// CHECK-NEXT:   AscendC::TransDataTo5HD<float>(v6_list, v6_list, v2);
// CHECK-NEXT:   uint64_t v2_dst_list[] = {v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7};
// CHECK-NEXT:   uint64_t v2_src_list[] = {v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7, v7};
// CHECK-NEXT:   AscendC::TransDataTo5HD<float>(v2_dst_list, v2_src_list, v2);
// CHECK: uint64_t [[ADDR1:.*]]=[[VAR1:.*]].GetPhyAddr()
// CHECK: uint64_t [[ADDR2:.*]]=[[VAR2:.*]].GetPhyAddr()
// CHECK: uint64_t [[LIST1:.*]][] = {[[ADDR1]]+0, [[ADDR1]]+1, [[ADDR1]]+2, [[ADDR1]]+3, [[ADDR1]]+4, [[ADDR1]]+5, [[ADDR1]]+6, [[ADDR1]]+7, [[ADDR1]]+8, [[ADDR1]]+9, [[ADDR1]]+10, [[ADDR1]]+11, [[ADDR1]]+12, [[ADDR1]]+13, [[ADDR1]]+14, [[ADDR1]]+15};
// CHECK: uint64_t [[LIST2:.*]][] = {[[ADDR2]]+0, [[ADDR2]]+1, [[ADDR2]]+2, [[ADDR2]]+3, [[ADDR2]]+4, [[ADDR2]]+5, [[ADDR2]]+6, [[ADDR2]]+7, [[ADDR2]]+8, [[ADDR2]]+9, [[ADDR2]]+10, [[ADDR2]]+11, [[ADDR2]]+12, [[ADDR2]]+13, [[ADDR2]]+14, [[ADDR2]]+15};
// CHECK: AscendC::TransDataTo5HD<uint64_t>([[LIST2]], [[LIST1]], [[PARAM:.*]]);
// CHECK:   return;
// CHECK: }
func.func @emit_trans_data_to_5hd(%tensor: !ascendc.local_tensor<1024xf32>, %params: !ascendc.trans_data_to_5hd_params, %arg: ui32) {
  %idx = arith.constant 0 : i32
  %dst = ascendc.local_tensor : !ascendc.local_tensor<1024xui64>
  %src = ascendc.local_tensor : !ascendc.local_tensor<1024xui64>
  %subidx = ascendc.local_tensor.subindex %tensor[%idx] : !ascendc.local_tensor<1024xf32>, i32, !ascendc.local_tensor<1024xf32>
  %addr = ascendc.local_tensor.get_phy_addr %subidx, %arg : !ascendc.local_tensor<1024xf32>, ui64, ui32
  ascendc.local_tensor.set_value %dst, %arg, %addr : !ascendc.local_tensor<1024xui64>, ui32, ui64
  ascendc.local_tensor.set_value %src, %arg, %addr : !ascendc.local_tensor<1024xui64>, ui32, ui64
  ascendc.trans_data_to_5hd %dst, %src, %params : !ascendc.local_tensor<1024xui64>, !ascendc.local_tensor<1024xui64>, !ascendc.trans_data_to_5hd_params
  ascendc.trans_data_to_5hd_tensor_list %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %subidx, %params {operandSegmentSizes = array<i32: 16, 16, 1>} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.trans_data_to_5hd_params
  ascendc.trans_data_to_5hd_uint_list %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %addr, %params {operandSegmentSizes = array<i32: 16, 16, 1>} : ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, ui64, !ascendc.trans_data_to_5hd_params
  ascendc.trans_data_to_5hd_tensor %dst, %src, %params {dstOffsets = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>, srcOffsets = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>} : !ascendc.local_tensor<1024xui64>, !ascendc.local_tensor<1024xui64>, !ascendc.trans_data_to_5hd_params
  return
}