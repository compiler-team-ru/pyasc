// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-outline-vf-group -canonicalize -inline -symbol-dce %s | FileCheck %s
// RUN: asctile-opt -ascvf2-outline-vf-group %s | asctile-opt

// CHECK-LABEL: func.func @fuse_float_chain(
func.func @fuse_float_chain(%arg0: tensor<1x1024xf32, #asctile.local<UB>>, %arg1: tensor<1x1024xf32, #asctile.local<UB>>) -> tensor<1x1024xf32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = arith.subf %arg0, %arg1
  // CHECK: %2 = arith.mulf %1, %arg0
  // CHECK: %3 = arith.addf %2, %arg1
  // CHECK: ascvf2.yield %3
  // CHECK: return %0
  %0 = arith.subf %arg0, %arg1 : tensor<1x1024xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<1x1024xf32, #asctile.local<UB>>
  %2 = arith.addf %1, %arg1 : tensor<1x1024xf32, #asctile.local<UB>>
  return %2 : tensor<1x1024xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_integer_chain(
func.func @fuse_integer_chain(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group
  // CHECK: %1 = arith.subi %arg0, %arg1
  // CHECK: %2 = arith.muli %1, %arg0
  // CHECK: ascvf2.yield %2
  %0 = arith.subi %arg0, %arg1 : tensor<32xi32, #asctile.local<UB>>
  %1 = arith.muli %0, %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_chain_with_external_intermediate_use(
func.func @fuse_chain_with_external_intermediate_use(%arg0: tensor<64xf32, #asctile.local<UB>>, %arg1: tensor<64xf32, #asctile.local<UB>>) -> (tensor<64xf32, #asctile.local<UB>>, tensor<64xf32, #asctile.local<UB>>) {
  // CHECK: %0:2 = ascvf2.vf_group
  // CHECK: %1 = arith.subf %arg0, %arg1
  // CHECK: %2 = arith.mulf %1, %arg0
  // CHECK: ascvf2.yield %1, %2
  // CHECK: return %0#0, %0#1
  %0 = arith.subf %arg0, %arg1 : tensor<64xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<64xf32, #asctile.local<UB>>
  return %0, %1 : tensor<64xf32, #asctile.local<UB>>, tensor<64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_diamond_use_def_chain(
func.func @fuse_diamond_use_def_chain(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group
  // CHECK: %1 = arith.subf %arg0, %arg1
  // CHECK: %2 = arith.mulf %1, %arg0
  // CHECK: %3 = arith.addf %1, %arg1
  // CHECK: %4 = arith.minimumf %2, %3
  // CHECK: ascvf2.yield %4
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = arith.mulf %0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %2 = arith.addf %0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %3 = arith.minimumf %1, %2 : tensor<128xf32, #asctile.local<UB>>
  return %3 : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_single_operation(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_single_operation(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// Independent consecutive operations can execute in parallel.
// CHECK-LABEL: func.func @fuse_independent_operations(
// CHECK: %0:2 = ascvf2.vf_group {
// CHECK: %1 = arith.addf %arg0, %arg0
// CHECK: %2 = arith.addf %arg1, %arg1
// CHECK: ascvf2.yield %1, %2
// CHECK: return %0#0, %0#1
func.func @fuse_independent_operations(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
  %0 = arith.addf %arg0, %arg0 : tensor<32xf32, #asctile.local<UB>>
  %1 = arith.addf %arg1, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %0, %1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_aligned_float_operation_scope(
func.func @fuse_aligned_float_operation_scope(%arg0: tensor<128xf32, #asctile.local<UB>>, %scalar: f32) -> tensor<128xf32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = arith.addf %arg0, %arg0
  // CHECK: %2 = math.log %1
  // CHECK: %3 = asctile.muls %2, %arg1
  // CHECK: %4 = asctile.relu %3
  // CHECK: ascvf2.yield %4
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %1 = math.log %0 : tensor<128xf32, #asctile.local<UB>>
  %2 = asctile.muls %1, %scalar : tensor<128xf32, #asctile.local<UB>>
  %3 = asctile.relu %2 : tensor<128xf32, #asctile.local<UB>>
  return %3 : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_aligned_integer_operation_scope(
func.func @fuse_aligned_integer_operation_scope(%arg0: tensor<128xi32, #asctile.local<UB>>, %shift: i32) -> tensor<128xi32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = math.absi %arg0
  // CHECK: %2 = asctile.shls %1, %arg1
  // CHECK: %3 = asctile.bitwise_not %2
  // CHECK: ascvf2.yield %3
  %0 = arith.andi %arg0, %arg0 : tensor<128xi32, #asctile.local<UB>>
  %1 = math.absi %0 : tensor<128xi32, #asctile.local<UB>>
  %2 = asctile.shls %1, %shift : tensor<128xi32, #asctile.local<UB>>
  %3 = asctile.bitwise_not %2 : tensor<128xi32, #asctile.local<UB>>
  return %3 : tensor<128xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_unsupported_math_operations(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_unsupported_math_operations(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg0 : tensor<128xf32, #asctile.local<UB>>
  %1 = math.ceil %0 : tensor<128xf32, #asctile.local<UB>>
  %2 = math.floor %1 : tensor<128xf32, #asctile.local<UB>>
  %3 = math.log2 %2 : tensor<128xf32, #asctile.local<UB>>
  %4 = math.log10 %3 : tensor<128xf32, #asctile.local<UB>>
  %5 = math.exp2 %4 : tensor<128xf32, #asctile.local<UB>>
  %6 = math.rsqrt %5 : tensor<128xf32, #asctile.local<UB>>
  %7 = math.round %6 : tensor<128xf32, #asctile.local<UB>>
  %8 = math.roundeven %7 : tensor<128xf32, #asctile.local<UB>>
  %9 = math.trunc %8 : tensor<128xf32, #asctile.local<UB>>
  return %9 : tensor<128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_unsupported_arith_operations(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_unsupported_arith_operations(%arg0: tensor<128xi32, #asctile.local<UB>>, %arg1: tensor<128xi32, #asctile.local<UB>>) -> tensor<128xi32, #asctile.local<UB>> {
  %0 = arith.addi %arg0, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %1 = arith.divui %0, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %2 = arith.remsi %1, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %3 = arith.remui %2, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %4 = arith.xori %3, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %5 = arith.shli %4, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %6 = arith.shrsi %5, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %7 = arith.shrui %6, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %8 = arith.maxui %7, %arg1 : tensor<128xi32, #asctile.local<UB>>
  %9 = arith.minui %8, %arg1 : tensor<128xi32, #asctile.local<UB>>
  return %9 : tensor<128xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_unsupported_asctile_operation(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_unsupported_asctile_operation(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<128xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.power %0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  return %1 : tensor<128xf32, #asctile.local<UB>>
}

// Only consecutive compatible operations are grouped.
// CHECK-LABEL: func.func @do_not_fuse_nonconsecutive_operations(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_nonconsecutive_operations(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
  %0 = arith.addf %arg0, %arg0 : tensor<32xf32, #asctile.local<UB>>
  %unused = arith.constant 0 : i32
  %1 = arith.addf %arg1, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %0, %1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_non_ub_tensors(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_non_ub_tensors(%arg0: tensor<32xf32, #asctile.local<L1>>, %arg1: tensor<32xf32, #asctile.local<L1>>) -> tensor<32xf32, #asctile.local<L1>> {
  %0 = arith.subf %arg0, %arg1 : tensor<32xf32, #asctile.local<L1>>
  %1 = arith.mulf %0, %arg0 : tensor<32xf32, #asctile.local<L1>>
  return %1 : tensor<32xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @fuse_elementwise_into_final_axis_sum(
func.func @fuse_elementwise_into_final_axis_sum(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = arith.subf %arg0, %arg1
  // CHECK: %2 = asctile.reduce <sum> %1 {dims = [0 : i32]}
  // CHECK: ascvf2.yield %2
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_elementwise_into_reduce_as_1d_tensor(
func.func @fuse_elementwise_into_reduce_as_1d_tensor(%arg0: tensor<1x128xf32, #asctile.local<UB>>, %arg1: tensor<1x128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = arith.subf %arg0, %arg1
  // CHECK: %2 = asctile.reduce_as_1d <sum> %1
  // CHECK: ascvf2.yield %2
  %0 = arith.subf %arg0, %arg1 : tensor<1x128xf32, #asctile.local<UB>>
  %1 = asctile.reduce_as_1d <sum> %0 : tensor<1x128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_reduce_as_1d_scalar(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_reduce_as_1d_scalar(%arg0: tensor<1x128xf32, #asctile.local<UB>>, %arg1: tensor<1x128xf32, #asctile.local<UB>>) -> f32 {
  %0 = arith.subf %arg0, %arg1 : tensor<1x128xf32, #asctile.local<UB>>
  %1 = asctile.reduce_as_1d <sum> %0 : tensor<1x128xf32, #asctile.local<UB>>, f32
  return %1 : f32
}

// CHECK-LABEL: func.func @do_not_fuse_unsupported_reduction_kind(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_unsupported_reduction_kind(%arg0: tensor<128xf32, #asctile.local<UB>>, %arg1: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <prod> %0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %1 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_non_final_axis_reduction(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_non_final_axis_reduction(%arg0: tensor<2x128xf32, #asctile.local<UB>>, %arg1: tensor<2x128xf32, #asctile.local<UB>>) -> tensor<1x128xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<2x128xf32, #asctile.local<UB>>
  %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<2x128xf32, #asctile.local<UB>>, tensor<1x128xf32, #asctile.local<UB>>
  return %1 : tensor<1x128xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @do_not_fuse_reduction_without_producer(
// CHECK-NOT: ascvf2.vf_group
func.func @do_not_fuse_reduction_without_producer(%arg0: tensor<128xf32, #asctile.local<UB>>) -> tensor<1xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <sum> %arg0 {dims = [0 : i32]} : tensor<128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  return %0 : tensor<1xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fuse_reduction_broadcast_prefix(
func.func @fuse_reduction_broadcast_prefix(%arg0: tensor<1x128xf32, #asctile.local<UB>>, %arg1: tensor<1x128xf32, #asctile.local<UB>>) -> tensor<1x128xf32, #asctile.local<UB>> {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = asctile.reduce_as_1d <max> %arg0
  // CHECK: %2 = asctile.broadcast %1
  // CHECK: %3 = arith.subf %arg0, %2
  // CHECK: %4 = math.exp %3
  // CHECK: %5 = asctile.reduce_as_1d <sum> %4
  // CHECK: %6 = asctile.broadcast %5
  // CHECK: %7 = arith.divf %4, %6
  // CHECK: ascvf2.yield %7
  %max = asctile.reduce_as_1d <max> %arg0 : tensor<1x128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  %broadcast = asctile.broadcast %max : tensor<1xf32, #asctile.local<UB>> to tensor<1x128xf32, #asctile.local<UB>>
  %difference = arith.subf %arg0, %broadcast : tensor<1x128xf32, #asctile.local<UB>>
  %exponential = math.exp %difference : tensor<1x128xf32, #asctile.local<UB>>
  %sum = asctile.reduce_as_1d <sum> %exponential : tensor<1x128xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
  %sumBroadcast = asctile.broadcast %sum : tensor<1xf32, #asctile.local<UB>> to tensor<1x128xf32, #asctile.local<UB>>
  %quotient = arith.divf %exponential, %sumBroadcast : tensor<1x128xf32, #asctile.local<UB>>
  return %quotient : tensor<1x128xf32, #asctile.local<UB>>
  // CHECK-NOT: ascvf2.vf_group
}
