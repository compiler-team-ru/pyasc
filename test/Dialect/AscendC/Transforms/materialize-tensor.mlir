// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-materialize-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @materialize_tensor_static
// CHECK:      %0 = ascendc.pipe
// CHECK-NEXT: %1 = ascendc.queue : <vecin, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %0, %1, %c1_i32, %c256_i64 : !ascendc.queue<vecin, 1>, i32, i64
// CHECK-NEXT: %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<64xf32>
// CHECK:      %3 = ascendc.pipe
// CHECK-NEXT: %4 = ascendc.queue : <vecin, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %3, %4, %c1_i32, %c512_i64 : !ascendc.queue<vecin, 1>, i32, i64
// CHECK-NEXT: %5 = ascendc.que_bind.alloc_tensor %4 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<4x32xf32>
// CHECK:      %6 = ascendc.pipe
// CHECK-NEXT: %7 = ascendc.queue : <vecout, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %6, %7, %c1_i32, %c256_i64 : !ascendc.queue<vecout, 1>, i32, i64
// CHECK-NEXT: %8 = ascendc.que_bind.alloc_tensor %7 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<64xf32>
// CHECK:      %9 = ascendc.pipe
// CHECK-NEXT: %10 = ascendc.queue : <vecout, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %9, %10, %c1_i32, %c512_i64 : !ascendc.queue<vecout, 1>, i32, i64
// CHECK-NEXT: %11 = ascendc.que_bind.alloc_tensor %10 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<4x32xf32>
// CHECK:      ascendc.que_bind.free_tensor %10, %11 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<4x32xf32>
// CHECK-NEXT: ascendc.que_bind.free_tensor %7, %8 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<64xf32>
// CHECK-NEXT: ascendc.que_bind.free_tensor %4, %5 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<4x32xf32>
// CHECK-NEXT: ascendc.que_bind.free_tensor %1, %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return %2, %5, %8, %11 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4x32xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4x32xf32>
func.func @materialize_tensor_static(%arg0 : !ascendc.global_tensor<*xf32>, %arg1 : !ascendc.global_tensor<*xf32>, %arg2 : !ascendc.global_tensor<*xf32>, %arg3 : !ascendc.global_tensor<*xf32>) -> (!ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4x32xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4x32xf32>) {
    %c4_i64 = arith.constant 4 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
    %1 = ascendc.local_tensor_auto vecin(%c4_i64, %c32_i64) input : <4x32xf32>
    %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
    %3 = ascendc.local_tensor_auto vecout(%c4_i64, %c32_i64) output : <4x32xf32>
    return %0, %1, %2, %3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4x32xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4x32xf32>
}

// CHECK-LABEL: func.func @materialize_tensor_dynamic
// CHECK:      %0 = arith.muli %arg3, %c16_i64 : i64
// CHECK-NEXT: %1 = ascendc.pipe
// CHECK-NEXT: %2 = ascendc.queue : <vecin, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %1, %2, %c1_i32, %0 : !ascendc.queue<vecin, 1>, i32, i64
// CHECK-NEXT: %3 = ascendc.que_bind.alloc_tensor %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK:      %4 = arith.muli %arg3, %c16_i64 : i64
// CHECK-NEXT: %5 = ascendc.pipe
// CHECK-NEXT: %6 = ascendc.queue : <vecout, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %5, %6, %c1_i32, %4 : !ascendc.queue<vecout, 1>, i32, i64
// CHECK-NEXT: %7 = ascendc.que_bind.alloc_tensor %6 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
// CHECK:      ascendc.que_bind.free_tensor %6, %7 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT: ascendc.que_bind.free_tensor %2, %3 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT: return %3, %7 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
func.func @materialize_tensor_dynamic(%arg0 : !ascendc.global_tensor<*xf32>, %arg1 : !ascendc.global_tensor<*xf32>, %arg2 : !ascendc.global_tensor<*xf32>, %arg3 : i64) -> (!ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>) {
    %c4_i64 = arith.constant 4 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = ascendc.local_tensor_auto vecin(%c4_i64, %arg3) input : <*xf32>
    %1 = ascendc.local_tensor_auto vecout(%c4_i64, %arg3) output : <*xf32>
    return %0, %1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
}

// CHECK-LABEL: func.func @materialize_tensor_vec_calc
// CHECK:      %0 = ascendc.pipe
// CHECK-NEXT: %1 = ascendc.tbuf : <veccalc>
// CHECK-NEXT: ascendc.pipe.init_buffer %0, %1, %c256_i64 : !ascendc.tbuf<veccalc>, i64
// CHECK-NEXT: %2 = ascendc.tbuf.get_tensor %1 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return %2 : !ascendc.local_tensor<64xf32>
func.func @materialize_tensor_vec_calc() -> (!ascendc.local_tensor<64xf32>) {
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    return %0 : !ascendc.local_tensor<64xf32>
}
