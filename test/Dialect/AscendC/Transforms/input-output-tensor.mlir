// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-input-output-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @input_output_tensor_ub(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: i32) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg1, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %2, %arg3 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %3, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @input_output_tensor_ub(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: !ascendc.global_tensor<*xf32>, %arg3: i32) {
    %0 = ascendc.local_tensor_auto vecin() : <64xf32>
    ascendc.data_copy_l2 %0, %arg0, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %1, %arg1, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %2 = ascendc.local_tensor_auto vecout() : <64xf32>
    ascendc.data_copy_l2 %arg2, %2, %arg3 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
    %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %2, %3, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @scf_for_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:  %2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:    scf.yield %arg2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @scf_for_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %tensor : !ascendc.local_tensor<64xf32>
    }
    ascendc.data_copy_l2 %arg0, %result, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    return
}

// CHECK-LABEL: func.func @scf_if_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>, %arg1: i1) {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:  %2 = scf.if %arg1 -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:    scf.yield %0 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %0 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @scf_if_yield_tensor_used_in_copy(%arg0: !ascendc.global_tensor<*xf32>, %cond: i1) {
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.if %cond -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %0 : !ascendc.local_tensor<64xf32>
    } else {
        scf.yield %0 : !ascendc.local_tensor<64xf32>
    }
    ascendc.data_copy_l2 %arg0, %result, %c64_i64 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    return
}

// CHECK-LABEL: func.func @subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto vecin() input output : <128xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %1, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = ascendc.local_tensor_auto vecin() : <128xf32>
    ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
    ascendc.data_copy_l2 %arg1, %1, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @copy_to_l0_mark_input_output(%arg0: !ascendc.global_tensor<*xf16>, %arg1: !ascendc.load_data_2d_params) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto b1() input output : <64xf16>
// CHECK-NEXT:  ascendc.load_data_g2l %0, %arg0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.global_tensor<*xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <64xf16>
// CHECK-NEXT:  ascendc.load_data_g2l %1, %0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.local_tensor<64xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @copy_to_l0_mark_input_output(%arg0: !ascendc.global_tensor<*xf16>, %arg1: !ascendc.load_data_2d_params) {
    %0 = ascendc.local_tensor_auto b1() : <64xf16>
    ascendc.load_data_g2l %0, %arg0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.global_tensor<*xf16>, !ascendc.load_data_2d_params
    %1 = ascendc.local_tensor_auto b2() : <64xf16>
    ascendc.load_data_g2l %1, %0, %arg1 : !ascendc.local_tensor<64xf16>, !ascendc.local_tensor<64xf16>, !ascendc.load_data_2d_params
    return
}

// CHECK-LABEL: func.func @fixpipe_mark_input_output(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto a1() output : <64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto co1() input : <64xf32>
// CHECK-NEXT:  ascendc.fixpipe %1, %0, %arg0, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @fixpipe_mark_input_output(%arg0: !ascendc.fixpipe_params_v220, %arg1: !ascendc.fixpipe_config) {
    %0 = ascendc.local_tensor_auto a1() : <64xf32>
    %1 = ascendc.local_tensor_auto co1() : <64xf32>
    ascendc.fixpipe %1, %0, %arg0, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    return
}

// CHECK-LABEL: func.func @reinterpret_and_subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto vecin() input output : <128xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<64xf32> to !ascendc.local_tensor<64xf16>
// CHECK-NEXT:  %3 = ascendc.local_tensor.subindex %2[%arg3] : !ascendc.local_tensor<64xf16>, i32, !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<32xf16> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %4, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<32xf32>, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reinterpret_and_subindex_chain(%arg0: !ascendc.global_tensor<*xf32>, %arg1: !ascendc.global_tensor<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = ascendc.local_tensor_auto vecin() : <128xf32>
    ascendc.data_copy_l2 %0, %arg0, %arg2 : !ascendc.local_tensor<128xf32>, !ascendc.global_tensor<*xf32>, i32
    %cast1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<128xf32> to !ascendc.local_tensor<64xf16>
    %sub = ascendc.local_tensor.subindex %0[%arg3] : !ascendc.local_tensor<128xf32>, i32, !ascendc.local_tensor<64xf32>
    %cast2 = ascendc.reinterpret_cast %sub : !ascendc.local_tensor<64xf32> to !ascendc.local_tensor<64xf16>
    %sub2 = ascendc.local_tensor.subindex %cast2[%arg3] : !ascendc.local_tensor<64xf16>, i32, !ascendc.local_tensor<32xf16>
    %cast3 = ascendc.reinterpret_cast %sub2 : !ascendc.local_tensor<32xf16> to !ascendc.local_tensor<32xf32>
    ascendc.data_copy_l2 %arg1, %cast3, %arg4 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<32xf32>, i32
    return
}
