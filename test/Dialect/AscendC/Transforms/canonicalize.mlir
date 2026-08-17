// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @erase_unused_ops() {
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @erase_unused_ops() {
  %0 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %1 = ascendc.local_tensor : !ascendc.local_tensor<777xf16>
  return
}

// CHECK-LABEL: func.func @fold_global_tensor_subindex(%arg0: !ascendc.global_tensor<*xf32>, %arg1: i32)
// CHECK-NEXT:  %c777_i32 = arith.constant 777 : i32
// CHECK-NEXT:  %0 = ascendc.global_tensor.subindex %arg0[%c777_i32] : !ascendc.global_tensor<*xf32>, i32, !ascendc.global_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.global_tensor.subindex %arg0[%arg1] : !ascendc.global_tensor<*xf32>, i32, !ascendc.global_tensor<*xf32>
// CHECK-NEXT:  return %arg0, %0, %1 : !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>
// CHECK-NEXT:}
func.func @fold_global_tensor_subindex(%arg0: !ascendc.global_tensor<*xf32>, %arg1: i32)
    -> (!ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c777_i32 = arith.constant 777 : i32
  %0 = ascendc.global_tensor.subindex %arg0[%c0_i32] : !ascendc.global_tensor<*xf32>, i32, !ascendc.global_tensor<*xf32>
  %1 = ascendc.global_tensor.subindex %arg0[%c777_i32] : !ascendc.global_tensor<*xf32>, i32, !ascendc.global_tensor<*xf32>
  %2 = ascendc.global_tensor.subindex %arg0[%arg1] : !ascendc.global_tensor<*xf32>, i32, !ascendc.global_tensor<*xf32>
  return %0, %1, %2 : !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>, !ascendc.global_tensor<*xf32>
}

// CHECK-LABEL: func.func @fold_local_tensor_subindex(%arg0: !ascendc.local_tensor<*xf32>, %arg1: i32)
// CHECK-NEXT:  %c777_i32 = arith.constant 777 : i32
// CHECK-NEXT:  %0 = ascendc.local_tensor.subindex %arg0[%c777_i32] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %arg0[%arg1] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:  return %arg0, %0, %1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:}
func.func @fold_local_tensor_subindex(%arg0: !ascendc.local_tensor<*xf32>, %arg1: i32)
  -> (!ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c777_i32 = arith.constant 777 : i32
  %0 = ascendc.local_tensor.subindex %arg0[%c0_i32] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
  %1 = ascendc.local_tensor.subindex %arg0[%c777_i32] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
  %2 = ascendc.local_tensor.subindex %arg0[%arg1] : !ascendc.local_tensor<*xf32>, i32, !ascendc.local_tensor<*xf32>
  return %0, %1, %2 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>
}

// CHECK-LABEL: func.func @fold_local_tensor_reinterpret_cast(%arg0: !ascendc.local_tensor<*xf32>)
// CHECK-NEXT:  %0 = ascendc.reinterpret_cast %arg0 : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %arg0 : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<777xi32>
// CHECK-NEXT:  return %0, %arg0, %1 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<777xi32>
// CHECK-NEXT:}
func.func @fold_local_tensor_reinterpret_cast(%arg0: !ascendc.local_tensor<*xf32>)
  -> (!ascendc.local_tensor<777xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<777xi32>) {
  %0 = ascendc.reinterpret_cast %arg0 : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<777xf32>
  %1 = ascendc.reinterpret_cast %arg0 : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<*xf32>
  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<777xf32> to !ascendc.local_tensor<777xi32>
  return %0, %1, %2 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<777xi32>
}
