// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-refine-cube-position %s | FileCheck %s

// CHECK-LABEL: func.func @refine_a1_position(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.mmad_params, %arg4: !ascendc.load_data_2d_params_v2, %arg5: !ascendc.data_copy_params) {
// CHECK-NEXT:  %0 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() input : <256x64xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() input output : <256x192xf16>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %0, %arg2 : !ascendc.local_tensor<256x192xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %arg4 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x192xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @refine_a1_position(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.mmad_params, %arg4: !ascendc.load_data_2d_params_v2, %arg5: !ascendc.data_copy_params) {
  %0 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
  %1 = ascendc.local_tensor_auto a2() input : <256x64xf16>
  %2 = ascendc.local_tensor_auto a1() input output : <256x192xf16>
  ascendc.data_copy_l2 %2, %0, %arg2 : !ascendc.local_tensor<256x192xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
  ascendc.load_data_l0_v2 %1, %2, %arg4 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x192xf16>, !ascendc.load_data_2d_params_v2
  return
}

// CHECK-LABEL: func.func @refine_b1_position(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.mmad_params, %arg4: !ascendc.load_data_2d_params_v2, %arg5: !ascendc.data_copy_params) {
// CHECK-NEXT:  %0 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <64x256xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto b1() input output : <256x192xf16>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %0, %arg2 : !ascendc.local_tensor<256x192xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  ascendc.load_data_l0_v2 %1, %2, %arg4 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x192xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @refine_b1_position(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.mmad_params, %arg4: !ascendc.load_data_2d_params_v2, %arg5: !ascendc.data_copy_params) {
  %0 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
  %1 = ascendc.local_tensor_auto b2() input : <64x256xf16>
  %2 = ascendc.local_tensor_auto a1() input output : <256x192xf16>
  ascendc.data_copy_l2 %2, %0, %arg2 : !ascendc.local_tensor<256x192xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
  ascendc.load_data_l0_v2 %1, %2, %arg4 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x192xf16>, !ascendc.load_data_2d_params_v2
  return
}

// CHECK-LABEL: func.func @refine_c1_position(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.mmad_params, %arg4: !ascendc.load_data_2d_params_v2, %arg5: !ascendc.data_copy_params) {
// CHECK-NEXT:  %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:  %0 = ascendc.global_tensor : !ascendc.global_tensor<4608xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto c2() input : <256xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto c1() input output : <256xf16>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %0, %c256_i32 : !ascendc.local_tensor<256xf16>, !ascendc.global_tensor<4608xf16>, i32
// CHECK-NEXT:  ascendc.data_copy_l0 %1, %2, %arg5 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @refine_c1_position(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.mmad_params, %arg4: !ascendc.load_data_2d_params_v2, %arg5: !ascendc.data_copy_params) {
  %c256_i32 = arith.constant 256 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<4608xf16>
  %1 = ascendc.local_tensor_auto c2() input : <256xf32>
  %2 = ascendc.local_tensor_auto a1() input output : <256xf16>
  ascendc.data_copy_l2 %2, %0, %c256_i32 : !ascendc.local_tensor<256xf16>, !ascendc.global_tensor<4608xf16>, i32
  ascendc.data_copy_l0 %1, %2, %arg5 : !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf16>, !ascendc.data_copy_params
  return
}
