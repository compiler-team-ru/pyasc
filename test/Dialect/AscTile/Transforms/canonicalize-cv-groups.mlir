// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @cube_erase_empty_group(%arg0: tensor<128xf16, #asctile.local<L1>>) -> tensor<128xf16, #asctile.local<L1>> {
// CHECK-NEXT: return %arg0 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @cube_erase_empty_group(%arg0: tensor<128xf16, #asctile.local<L1>>) -> tensor<128xf16, #asctile.local<L1>> {
  %0 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
    asctile.yield %arg0 : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %0 : tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @vector_erase_empty_group(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @vector_erase_empty_group(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    asctile.yield %arg0 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @cube_erase_empty_group_no_operands() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @cube_erase_empty_group_no_operands() {
  asctile.cube_group {
    asctile.yield
  }
  return
}

// CHECK-LABEL: func.func @vector_erase_empty_group_no_operands() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @vector_erase_empty_group_no_operands() {
  asctile.vector_group {
    asctile.yield
  }
  return
}

// CHECK-LABEL: func.func @cube_erase_unused_operands(%arg0: tensor<128xf16, #asctile.local<L1>>, %arg1: tensor<128xf16, #asctile.local<L1>>) -> tensor<128xf16, #asctile.local<L0A>> {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT:   %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:   asctile.yield %1 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %0 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @cube_erase_unused_operands(%arg0: tensor<128xf16, #asctile.local<L1>>, %arg1: tensor<128xf16, #asctile.local<L1>>) -> tensor<128xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.cube_group(%arg0, %arg1 : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>) {
    %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
    asctile.yield %1 : tensor<128xf16, #asctile.local<L0A>>
  } : tensor<128xf16, #asctile.local<L0A>>
  return %0 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @vector_erase_unused_operands(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:   %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @vector_erase_unused_operands(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.vector_group(%arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @cube_erase_unused_result(%arg0: tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @cube_erase_unused_result(%arg0: tensor<128xf16, #asctile.local<L1>>) {
  %0 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
    asctile.yield %arg0 : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return
}

// CHECK-LABEL: func.func @vector_erase_unused_result(%arg0: tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @vector_erase_unused_result(%arg0: tensor<32xf32, #asctile.local<UB>>) {
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    asctile.yield %arg0 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return
}

// CHECK-LABEL: func.func @cube_keep_used_result(%arg0: tensor<128xf16, #asctile.local<L1>>) -> tensor<128xf16, #asctile.local<L0A>> {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT:   %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:   asctile.yield %1 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %0 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @cube_keep_used_result(%arg0: tensor<128xf16, #asctile.local<L1>>) -> tensor<128xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
    %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
    asctile.yield %1 : tensor<128xf16, #asctile.local<L0A>>
  } : tensor<128xf16, #asctile.local<L0A>>
  return %0 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @vector_keep_used_result(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:   %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @vector_keep_used_result(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @cube_used_and_forwarded_arg(%arg0: tensor<128xf16, #asctile.local<L1>>, %arg1: tensor<128xf16, #asctile.local<L1>>) -> (tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT:   %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:   asctile.yield %1 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %0, %arg1 : tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:}
func.func @cube_used_and_forwarded_arg(%arg0: tensor<128xf16, #asctile.local<L1>>, %arg1: tensor<128xf16, #asctile.local<L1>>) -> (tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
    %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
    asctile.yield %1, %arg1 : tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
  return %0#0, %0#1 : tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @vector_used_and_forwarded_arg(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT: %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:   %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0, %arg1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @vector_used_and_forwarded_arg(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
  %0:2 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1, %arg1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0#0, %0#1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @cube_used_and_forwarded_op(%arg0: tensor<128xf16, #asctile.local<L1>>) -> (tensor<128xf16, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0C>>) {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = asctile.accumulator : tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT: %1 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT:   %2 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:   asctile.yield %2 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %1, %0 : tensor<128xf16, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @cube_used_and_forwarded_op(%arg0: tensor<128xf16, #asctile.local<L1>>) -> (tensor<128xf16, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  %acc = asctile.accumulator : tensor<16x16xf32, #asctile.local<L0C>>
  %0:2 = asctile.cube_group(%arg0 : tensor<128xf16, #asctile.local<L1>>) {
    %1 = asctile.copy %arg0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
    asctile.yield %1, %acc : tensor<128xf16, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0C>>
  } : tensor<128xf16, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0C>>
  return %0#0, %0#1 : tensor<128xf16, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @vector_used_and_forwarded_op(%arg0: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT: %cst = arith.constant dense<1.000000e+00> : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:   %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0, %cst : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @vector_used_and_forwarded_op(%arg0: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32, #asctile.local<UB>>
  %0:2 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
    %1 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1, %cst : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0#0, %0#1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
}
