// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @if_aic_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aic_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aic(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aiv_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aiv_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aic_erase_empty_group_no_operands() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aic_erase_empty_group_no_operands() {
  ascendc.if_aic {
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @if_aiv_erase_empty_group_no_operands() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aiv_erase_empty_group_no_operands() {
  ascendc.if_aiv {
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @if_aic_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aic_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aiv_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aiv_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aiv(%arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aic_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aic_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
  %0 = ascendc.if_aic(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return
}

// CHECK-LABEL: func.func @if_aiv_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aiv_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return
}

// CHECK-LABEL: func.func @if_aic_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aic_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
  %0:2 = ascendc.if_aic(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
  }
  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aiv_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aiv_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
  %0:2 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
  }
  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
}
