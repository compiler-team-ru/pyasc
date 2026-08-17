// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-fill-asc-operands -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @test_binary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i32) {
// CHECK:       ascendc.add_l2 %arg0, %arg1, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_binary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i32) {
  ascendc.add_l2 %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
  return
}

// CHECK-LABEL: func.func @test_unary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i32) {
// CHECK:       ascendc.abs_l2 %arg0, %arg1, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_unary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i32) {
  ascendc.abs_l2 %arg0, %arg1, %arg2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
  return
}

// CHECK-LABEL: func.func @test_duplicate_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
// CHECK:       ascendc.duplicate_l2 %arg0, %arg1, %c64_i64 : !ascendc.local_tensor<64xf32>, f32, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_duplicate_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
  ascendc.duplicate_l2 %arg0, %arg1, %arg2 : !ascendc.local_tensor<64xf32>, f32, i32
  return
}

// CHECK-LABEL: func.func @test_vecscalar_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i32) {
// CHECK:       ascendc.adds_l2 %arg0, %arg1, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_vecscalar_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i32) {
  ascendc.adds_l2 %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i32
  return
}

// CHECK-LABEL: func.func @test_l2_cal_count_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
// CHECK-NEXT:  ascendc.duplicate_l2 %arg0, %arg1, %arg2 {asc.cal_count_set} : !ascendc.local_tensor<64xf32>, f32, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_l2_cal_count_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
  ascendc.duplicate_l2 %arg0, %arg1, %arg2 {asc.cal_count_set} : !ascendc.local_tensor<64xf32>, f32, i32
  return
}

// CHECK-LABEL: func.func @test_unary_l0(%arg0: !ascendc.local_tensor<64xf16>, %arg1: !ascendc.local_tensor<64xf16>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.abs_l0 %arg0, %arg1, %c128_i64, %c1_i64, %0 {isSetMask} : !ascendc.local_tensor<64xf16>, !ascendc.local_tensor<64xf16>, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_unary_l0(%arg0: !ascendc.local_tensor<64xf16>, %arg1: !ascendc.local_tensor<64xf16>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
  ascendc.abs_l0 %arg0, %arg1, %arg2, %arg3, %arg4 : !ascendc.local_tensor<64xf16>, !ascendc.local_tensor<64xf16>, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_binary_l0(%arg0: !ascendc.local_tensor<256xbf16>, %arg1: !ascendc.local_tensor<256xbf16>, %arg2: !ascendc.local_tensor<256xbf16>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.binary_repeat_params(%c1_i64, %c1_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64, i64, i64
// CHECK-NEXT:  ascendc.add_l0 %arg0, %arg1, %arg2, %c128_i64, %c2_i64, %0 {isSetMask} : !ascendc.local_tensor<256xbf16>, !ascendc.local_tensor<256xbf16>, !ascendc.local_tensor<256xbf16>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_binary_l0(%arg0: !ascendc.local_tensor<256xbf16>, %arg1: !ascendc.local_tensor<256xbf16>, %arg2: !ascendc.local_tensor<256xbf16>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
  ascendc.add_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !ascendc.local_tensor<256xbf16>, !ascendc.local_tensor<256xbf16>, !ascendc.local_tensor<256xbf16>, i64, i64, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_cast_l0(%arg0: !ascendc.local_tensor<192xf32>, %arg1: !ascendc.local_tensor<192xi32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.cast_l0 %arg0, %arg1, %c64_i64, %c3_i64, %0 {isSetMask, roundMode = 0 : i32} : !ascendc.local_tensor<192xf32>, !ascendc.local_tensor<192xi32>, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_cast_l0(%arg0: !ascendc.local_tensor<192xf32>, %arg1: !ascendc.local_tensor<192xi32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
  ascendc.cast_l0 %arg0, %arg1, %arg2, %arg3, %arg4 {roundMode = 0 : i32} : !ascendc.local_tensor<192xf32>, !ascendc.local_tensor<192xi32>, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_compare_scalar_l0(%arg0: !ascendc.local_tensor<2xui8>, %arg1: !ascendc.local_tensor<256xf64>, %arg2: f64, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.compare_scalar_l0 %arg0, %arg1, %arg2, %c32_i64, %c8_i64, %0 {cmpMode = 0 : i64, isSetMask} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<256xf64>, f64, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_compare_scalar_l0(%arg0: !ascendc.local_tensor<2xui8>, %arg1: !ascendc.local_tensor<256xf64>, %arg2: f64, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
  ascendc.compare_scalar_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {cmpMode = 0 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<256xf64>, f64, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_compare_l0(%arg0: !ascendc.local_tensor<32xui8>, %arg1: !ascendc.local_tensor<1024xi8>, %arg2: !ascendc.local_tensor<1024xi8>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.binary_repeat_params(%c1_i64, %c1_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64, i64, i64
// CHECK-NEXT:  ascendc.compare_l0 %arg0, %arg1, %arg2, %c256_i64, %c4_i64, %0 {cmpMode = 1 : i64, isSetMask} : !ascendc.local_tensor<32xui8>, !ascendc.local_tensor<1024xi8>, !ascendc.local_tensor<1024xi8>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_compare_l0(%arg0: !ascendc.local_tensor<32xui8>, %arg1: !ascendc.local_tensor<1024xi8>, %arg2: !ascendc.local_tensor<1024xi8>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
  ascendc.compare_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {cmpMode = 1 : i64} : !ascendc.local_tensor<32xui8>, !ascendc.local_tensor<1024xi8>, !ascendc.local_tensor<1024xi8>, i64, i64, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_select_l0(%arg0: !ascendc.local_tensor<640xi16>, %arg1: !ascendc.local_tensor<80xui8>, %arg2: !ascendc.local_tensor<640xi16>, %arg3: !ascendc.local_tensor<640xi16>, %arg4: i64, %arg5: i64, %arg6: !ascendc.binary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.binary_repeat_params(%c1_i64, %c1_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64, i64, i64
// CHECK-NEXT:  ascendc.select_l0 %arg0, %arg1, %arg2, %arg3, %c128_i64, %c5_i64, %0 {isSetMask, selMode = 2 : i32} : !ascendc.local_tensor<640xi16>, !ascendc.local_tensor<80xui8>, !ascendc.local_tensor<640xi16>, !ascendc.local_tensor<640xi16>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_select_l0(%arg0: !ascendc.local_tensor<640xi16>, %arg1: !ascendc.local_tensor<80xui8>, %arg2: !ascendc.local_tensor<640xi16>, %arg3: !ascendc.local_tensor<640xi16>, %arg4: i64, %arg5: i64, %arg6: !ascendc.binary_repeat_params) {
  ascendc.select_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {selMode = 2 : i32} : !ascendc.local_tensor<640xi16>, !ascendc.local_tensor<80xui8>, !ascendc.local_tensor<640xi16>, !ascendc.local_tensor<640xi16>, i64, i64, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_duplicate_l0(%arg0: !ascendc.local_tensor<384xi32>, %arg1: i32, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
// CHECK:       ascendc.duplicate_l0 %arg0, %arg1, %c64_i64, %c6_i64, %arg4, %arg5 {isSetMask} : !ascendc.local_tensor<384xi32>, i32, i64, i64, i64, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_duplicate_l0(%arg0: !ascendc.local_tensor<384xi32>, %arg1: i32, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
  ascendc.duplicate_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !ascendc.local_tensor<384xi32>, i32, i64, i64, i64, i64
  return
}

// CHECK-LABEL: func.func @test_vecscalar_l0(%arg0: !ascendc.local_tensor<224xi64>, %arg1: !ascendc.local_tensor<224xi64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.adds_l0 %arg0, %arg1, %arg2, %c32_i64, %c7_i64, %0 {isSetMask} : !ascendc.local_tensor<224xi64>, !ascendc.local_tensor<224xi64>, i64, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_vecscalar_l0(%arg0: !ascendc.local_tensor<224xi64>, %arg1: !ascendc.local_tensor<224xi64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
  ascendc.adds_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !ascendc.local_tensor<224xi64>, !ascendc.local_tensor<224xi64>, i64, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_l0_mask_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.abs_l0 %arg0, %arg1, %arg2, %c1_i64, %0 {asc.mask_set, isSetMask} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_l0_mask_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
  ascendc.abs_l0 %arg0, %arg1, %arg2, %arg3, %arg4 {asc.mask_set} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.unary_repeat_params
  return
}
