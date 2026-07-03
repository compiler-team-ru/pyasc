// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-verify-sync -split-input-file -verify-diagnostics %s

// expected-note@below {{tensor declared here}}
func.func @warning_if_absent_alloc_tensor(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %global_tensor : !ascendc.global_tensor<*xf16>, %other_tensor: !ascendc.local_tensor<*xf16>) {
  // expected-warning@below {{FreeTensor: there is no corresponding call to AllocTensor}}
  ascendc.que_bind.free_tensor %que_bind, %other_tensor : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

func.func @warning_if_tensor_free_twice(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %global_tensor : !ascendc.global_tensor<*xf16>) {
  // expected-note@below {{tensor declared here}}
  %tensor1 = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  // expected-warning@below {{FreeTensor: tensor memory was freed before its last use}}
  ascendc.que_bind.free_tensor %que_bind, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

func.func @warning_if_tensor_not_free(%que_bind : !ascendc.que_bind<gm, vecin, 1>) {
  // expected-warning@below {{AllocTensor: there is no corresponding call to FreeTensor for this tensor}}
  %tensor1 = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

// expected-note@below {{queue declared here}}
func.func @warning_if_deque_more_than_enque(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %other_tensor: !ascendc.local_tensor<*xf16>) {
  ascendc.que_bind.enque_tensor %que_bind, %other_tensor : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor2 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  // expected-warning@below {{DeQue: there is no corresponding call to EnQue}}
  %tensor3 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

// expected-note@below {{queue declared here}}
func.func @warning_if_enque_more_than_deque(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %other_tensor1: !ascendc.local_tensor<*xf16>, %other_tensor2: !ascendc.local_tensor<*xf16>) {
  ascendc.que_bind.enque_tensor %que_bind, %other_tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  // expected-warning@below {{EnQue: there is no corresponding call to DeQue for this tensor}}
  ascendc.que_bind.enque_tensor %que_bind, %other_tensor2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor1 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

func.func @warning_if_exist_op_between_enque_and_deque(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %global_tensor : !ascendc.global_tensor<*xf16>, %other_local_tensor: !ascendc.local_tensor<*xf16>, %c_i32 : i32) {
  // expected-note@below {{tensor declared here}}
  %tensor1 = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.enque_tensor %que_bind, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  // expected-warning@below {{unexpected use of tensor between EnQue and DeQue}}
  ascendc.data_copy_l2 %tensor1, %global_tensor, %c_i32 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, i32
  %tensor2 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind, %tensor2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

func.func @no_warning_for_correct_alloc_free(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %global_tensor : !ascendc.global_tensor<*xf16>) {
  %tensor1 = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor2 = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind, %tensor2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

func.func @no_warning_for_correct_enque_deque(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %local_tensor : !ascendc.local_tensor<*xf16>, %global_tensor : !ascendc.global_tensor<*xf16>, %c_i32 : i32) {
  ascendc.que_bind.enque_tensor %que_bind, %local_tensor : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor1 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.data_copy_l2 %local_tensor, %global_tensor, %c_i32 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, i32
  ascendc.data_copy_l2 %tensor1, %global_tensor, %c_i32 : !ascendc.local_tensor<*xf16>, !ascendc.global_tensor<*xf16>, i32
  ascendc.que_bind.enque_tensor %que_bind, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor2 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

func.func @no_warning_for_correct_two_que_binds(%que_bind1 : !ascendc.que_bind<gm, vecin, 1>, %que_bind2 : !ascendc.que_bind<gm, vecin, 1>, %global_tensor : !ascendc.global_tensor<*xf16>, %other_local_tensor: !ascendc.local_tensor<*xf16>, %c_i32 : i32) {
  %tensor1 = ascendc.que_bind.alloc_tensor %que_bind1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor2 = ascendc.que_bind.alloc_tensor %que_bind2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.enque_tensor %que_bind1, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.enque_tensor %que_bind2, %tensor2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor3 = ascendc.que_bind.deque_tensor %que_bind1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor4 = ascendc.que_bind.deque_tensor %que_bind2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind1, %tensor3 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind2, %tensor4 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}

// -----

// The queue's pending op list holds an EnQue op ahead of the matching AllocTensor,
// so FreeTensor must skip the non-alloc op (dyn_cast returns null) before finding its alloc.
// expected-note@below {{queue declared here}}
func.func @free_skips_non_alloc_op_in_queue(%que_bind : !ascendc.que_bind<gm, vecin, 1>, %other_tensor: !ascendc.local_tensor<*xf16>) {
  // expected-warning@below {{EnQue: there is no corresponding call to DeQue for this tensor}}
  ascendc.que_bind.enque_tensor %que_bind, %other_tensor : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  %tensor1 = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  ascendc.que_bind.free_tensor %que_bind, %tensor1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf16>
  return
}
