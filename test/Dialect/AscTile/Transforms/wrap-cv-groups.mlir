// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-wrap-cv-groups %s | FileCheck %s

// CHECK-LABEL: func.func @cube_load_and_copy(%arg0: tensor<128xf16, #asctile.global>, %arg1: i32) -> tensor<128xf16, #asctile.local<L0A>> {
// CHECK-NEXT:  %0 = asctile.cube_group {
// CHECK-NEXT:    %2 = asctile.load %arg0[%arg1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    asctile.yield %2 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  %1 = asctile.cube_group(%0 : tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT:    %2 = asctile.copy %0[%arg1] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:    asctile.yield %2 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:  } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:  return %1 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @cube_load_and_copy(%arg0: tensor<128xf16, #asctile.global>, %arg1: i32) -> tensor<128xf16, #asctile.local<L0A>> {
  %0 = asctile.load %arg0[%arg1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %1 = asctile.copy %0[%arg1] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
  return %1 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @vector_ops(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32) {
// CHECK-NEXT:  %0 = asctile.vector_group {
// CHECK-NEXT:    %2 = asctile.load %arg0[%arg1] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.vector_group(%0 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    %2 = asctile.relu %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  asctile.vector_group(%1 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    asctile.store %1, %arg0[%arg1] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @vector_ops(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32) {
  %0 = asctile.load %arg0[%arg1] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  %1 = asctile.relu %0 : tensor<32xf32, #asctile.local<UB>>
  asctile.store %1, %arg0[%arg1] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @mixed_cube_and_vector(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>, %arg2: i32) -> tensor<128xf16, #asctile.local<L0A>> {
// CHECK-NEXT:  %0 = asctile.cube_group {
// CHECK-NEXT:    %4 = asctile.load %arg0[%arg2] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    asctile.yield %4 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  %1 = asctile.cube_group(%0 : tensor<128xf16, #asctile.local<L1>>) {
// CHECK-NEXT:    %4 = asctile.copy %0[%arg2] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:    asctile.yield %4 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:  } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:  %2 = asctile.vector_group {
// CHECK-NEXT:    %4 = asctile.load %arg1[%arg2] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %4 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.vector_group(%2 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    %4 = asctile.relu %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %4 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @mixed_cube_and_vector(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>, %arg2: i32) -> tensor<128xf16, #asctile.local<L0A>> {
  %0 = asctile.load %arg0[%arg2] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %1 = asctile.copy %0[%arg2] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
  %2 = asctile.load %arg1[%arg2] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  %3 = asctile.relu %2 : tensor<32xf32, #asctile.local<UB>>
  return %1 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @no_wrap_scalar_ops(
// CHECK-NOT:   asctile.{{.+}}_group
// CHECK:       arith.addi
// CHECK-NEXT:  asctile.vector_group
// CHECK-NEXT:  asctile.load
// CHECK-NEXT:  asctile.yield
// CHECH-NEXT:  }
// CHECH-NEXT:  return
func.func @no_wrap_scalar_ops(%arg0: tensor<32xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %c0, %c1 : i32
  %0 = asctile.load %arg0[%sum] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @matmul_cube_group(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<128xf16, #asctile.global>)
// CHECK: asctile.cube_group
// CHECK:   asctile.accumulator
// CHECK: asctile.cube_group
// CHECK:   asctile.load
// CHECK: asctile.cube_group
// CHECK:   asctile.copy
// CHECK: asctile.cube_group
// CHECK:   asctile.load
// CHECK: asctile.cube_group
// CHECK:   asctile.copy
// CHECK: asctile.cube_group
// CHECK:   asctile.matmul
func.func @matmul_cube_group(%arg0: tensor<128xf16, #asctile.global>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  %c0 = arith.constant 0 : i32
  %acc = asctile.accumulator : tensor<16x16xf32, #asctile.local<L0C>>
  %la = asctile.load %arg0[%c0, %c0] : tensor<128xf16, #asctile.global>, tensor<16x16xf16, #asctile.local<L1>>
  %ca = asctile.copy %la[%c0, %c0] : tensor<16x16xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0A>>
  %lb = asctile.load %arg0[%c0, %c0] : tensor<128xf16, #asctile.global>, tensor<16x16xf16, #asctile.local<L1>>
  %cb = asctile.copy %lb[%c0, %c0] : tensor<16x16xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0B>>
  %res = asctile.matmul %ca, %cb : tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>> -> tensor<16x16xf32, #asctile.local<L0C>>
  return %res : tensor<16x16xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @vector_arith_in_block(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.vector_group(%arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    %1 = arith.addf %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @vector_arith_in_block(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @store_not_wrapped(%arg0: tensor<32xf32, #asctile.global>) {
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %0 = asctile.vector_group {
// CHECK-NEXT:    %1 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  asctile.vector_group(%0 : tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:    asctile.store %0, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @store_not_wrapped(%arg0: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  asctile.store %0, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @matmul_acc_cube_group(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf16, #asctile.local<L0A>>, %arg2: tensor<16x16xf16, #asctile.local<L0B>>) {
// CHECK-NEXT:  asctile.cube_group(%arg0, %arg1, %arg2 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>) {
// CHECK-NEXT:    asctile.matmul_acc %arg0, %arg1, %arg2 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @matmul_acc_cube_group(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf16, #asctile.local<L0A>>, %arg2: tensor<16x16xf16, #asctile.local<L0B>>) {
  asctile.matmul_acc %arg0, %arg1, %arg2 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
  return
}
