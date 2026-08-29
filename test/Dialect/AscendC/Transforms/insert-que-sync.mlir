// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-insert-que-sync %s | FileCheck %s

// CHECK-LABEL: func.func @get_set_value
// CHECK:       %0 = ascendc.pipe
// CHECK-NEXT:  %1 = ascendc.pipe.fetch_event_id %0, v_s : i8
// CHECK-NEXT:  ascendc.set_flag v_s, %1 : i8
// CHECK-NEXT:  ascendc.wait_flag v_s, %1 : i8
// CHECK-NEXT:  %2 = ascendc.local_tensor.get_value %arg0, %arg1 : !ascendc.local_tensor<64xf32>, ui32, f32
// CHECK-NEXT:  %3 = ascendc.pipe
// CHECK-NEXT:  %4 = ascendc.pipe.fetch_event_id %3, s_v : i8
// CHECK-NEXT:  ascendc.set_flag s_v, %4 : i8
// CHECK-NEXT:  ascendc.wait_flag s_v, %4 : i8
// CHECK-NEXT:  %5 = ascendc.pipe
// CHECK-NEXT:  %6 = ascendc.pipe.fetch_event_id %5, v_s : i8
// CHECK-NEXT:  ascendc.set_flag v_s, %6 : i8
// CHECK-NEXT:  ascendc.wait_flag v_s, %6 : i8
// CHECK-NEXT:  ascendc.local_tensor.set_value %arg0, %arg1, %2 : !ascendc.local_tensor<64xf32>, ui32, f32
// CHECK-NEXT:  %7 = ascendc.pipe
// CHECK-NEXT:  %8 = ascendc.pipe.fetch_event_id %7, s_v : i8
// CHECK-NEXT:  ascendc.set_flag s_v, %8 : i8
// CHECK-NEXT:  ascendc.wait_flag s_v, %8 : i8
func.func @get_set_value(%arg0: !ascendc.local_tensor<64xf32>, %arg1: ui32) {
    %0 = ascendc.local_tensor.get_value %arg0, %arg1 : !ascendc.local_tensor<64xf32>, ui32, f32
    ascendc.local_tensor.set_value %arg0, %arg1, %0 : !ascendc.local_tensor<64xf32>, ui32, f32
    return
}

// CHECK-LABEL: func.func @get_set_value_for
// CHECK:         %0 = ascendc.pipe
// CHECK-NEXT:    %1 = ascendc.pipe.fetch_event_id %0, v_s : i8
// CHECK-NEXT:    ascendc.set_flag v_s, %1 : i8
// CHECK-NEXT:    ascendc.wait_flag v_s, %1 : i8
// CHECK-NEXT:    %2 = ascendc.global_tensor.get_value %arg0, %arg2 : !ascendc.global_tensor<64xf32>, i32, f32
// CHECK-NEXT:    %3 = ascendc.pipe
// CHECK-NEXT:    %4 = ascendc.pipe.fetch_event_id %3, s_v : i8
// CHECK-NEXT:    ascendc.set_flag s_v, %4 : i8
// CHECK-NEXT:    ascendc.wait_flag s_v, %4 : i8
// CHECK-NEXT:    %5 = ascendc.pipe
// CHECK-NEXT:    %6 = ascendc.pipe.fetch_event_id %5, v_s : i8
// CHECK-NEXT:    ascendc.set_flag v_s, %6 : i8
// CHECK-NEXT:    ascendc.wait_flag v_s, %6 : i8
// CHECK-NEXT:    scf.for %arg5 = %arg2 to %arg3 step %arg2  : i32 {
// CHECK-NEXT:      ascendc.global_tensor.set_value %arg0, %arg4, %2 : !ascendc.global_tensor<64xf32>, ui64, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %7 = ascendc.pipe
// CHECK-NEXT:    %8 = ascendc.pipe.fetch_event_id %7, s_v : i8
// CHECK-NEXT:    ascendc.set_flag s_v, %8 : i8
// CHECK-NEXT:    ascendc.wait_flag s_v, %8 : i8
func.func @get_set_value_for(%arg0: !ascendc.global_tensor<64xf32>, %arg1: ui32, %arg2: i32, %arg3: i32, %arg4: ui64) {
    %0 = ascendc.global_tensor.get_value %arg0, %arg2 : !ascendc.global_tensor<64xf32>, i32, f32
    scf.for %arg5 = %arg2 to %arg3 step %arg2 : i32 {
        ascendc.global_tensor.set_value %arg0, %arg4, %0 : !ascendc.global_tensor<64xf32>, ui64, f32
    }
    return
}

// CHECK-LABEL: func.func @insert_pipe_v_pipe_all
// CHECK:       ascendc.duplicate_l2 %arg0, %arg1, %arg2 : !ascendc.local_tensor<64xf32>, f32, i32
// CHECK-NEXT:  ascendc.pipe_barrier pipe_v
// CHECK-NEXT:  ascendc.add_l2 %arg3, %arg0, %arg0, %arg2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
// CHECK-NEXT:  ascendc.pipe_barrier pipe_all
// CHECK-NEXT:  return
func.func @insert_pipe_v_pipe_all(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32, %arg3: !ascendc.local_tensor<64xf32>) {
    ascendc.duplicate_l2 %arg0, %arg1, %arg2 : !ascendc.local_tensor<64xf32>, f32, i32
    ascendc.add_l2 %arg3, %arg0, %arg0, %arg2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @insert_enqueue_dequeue
// CHECK:       ascendc.data_copy_l2 %2, %arg0, %arg2 : !ascendc.local_tensor<32xf16>, !ascendc.global_tensor<*xf16>, i32
// CHECK-NEXT:  ascendc.que_bind.enque_tensor %1, %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
// CHECK:       %5 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  ascendc.add_l2 %4, %5, %5, %arg2 : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, i32
// CHECK-NEXT:  ascendc.que_bind.enque_tensor %3, %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>
// CHECK-NEXT:  %6 = ascendc.que_bind.deque_tensor %3 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>
func.func @insert_enqueue_dequeue(%arg0: !ascendc.global_tensor<*xf16>, %arg1: !ascendc.global_tensor<*xf16>, %arg2: i32, %arg3: i64) {
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vecin, 1>
    ascendc.pipe.init_queue %0, %1, %arg2, %arg3 : !ascendc.queue<vecin, 1>, i32, i64
    %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
    ascendc.data_copy_l2 %2, %arg0, %arg2 : !ascendc.local_tensor<32xf16>, !ascendc.global_tensor<*xf16>, i32
    %3 = ascendc.queue : <vecout, 1>
    ascendc.pipe.init_queue %0, %3, %arg2, %arg3 : !ascendc.queue<vecout, 1>, i32, i64
    %4 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>
    ascendc.add_l2 %4, %2, %2, %arg2 : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, i32
    ascendc.data_copy_l2 %arg1, %4, %arg2 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<32xf16>, i32
    ascendc.que_bind.free_tensor %3, %4 : !ascendc.queue<vecout, 1>, !ascendc.local_tensor<32xf16>
    ascendc.que_bind.free_tensor %1, %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
    return
}
