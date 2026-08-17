// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-detect-bias-load %s | FileCheck %s

// CHECK-LABEL: func.func @mark_bias_basic(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]]] {asctile.is_bias} : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]]] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<BT>>
// CHECK-NEXT: return %[[COPY]] : tensor<128xf16, #asctile.local<BT>>
// CHECK-NEXT:}
func.func @mark_bias_basic(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<BT>> {
  %c0_i32 = arith.constant 0 : i32
  %load = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<BT>>
  return %copy : tensor<128xf16, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @no_mark_non_bt_copy(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]]] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]]] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<UB>>
// CHECK-NEXT: return %[[COPY]] : tensor<128xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @no_mark_non_bt_copy(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<UB>> {
  %c0_i32 = arith.constant 0 : i32
  %load = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<UB>>
  return %copy : tensor<128xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @mark_bias_with_other_use(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<128xf16, #asctile.global>, %[[TENSOR2:.*]]: tensor<128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]]] {asctile.is_bias} : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY_BT:.*]] = asctile.copy %[[LOAD]][%[[C0]]] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<BT>>
// CHECK-NEXT: asctile.store %[[LOAD]], %[[TENSOR2]][%[[C0]]] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.global>
// CHECK-NEXT: return %[[COPY_BT]] : tensor<128xf16, #asctile.local<BT>>
// CHECK-NEXT:}
func.func @mark_bias_with_other_use(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<BT>> {
  %c0_i32 = arith.constant 0 : i32
  %load = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %copy_bt = asctile.copy %load[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<BT>>
  asctile.store %load, %arg1[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.global>
  return %copy_bt : tensor<128xf16, #asctile.local<BT>>
}
