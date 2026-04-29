// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-reuse-ub-allocation %s | FileCheck %s

// CHECK-LABEL: func.func @works_without_crash() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  return
func.func @works_without_crash() {
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_tensor_with_static_shape
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <333xi32>
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<333xi32> to !ascendc.local_tensor<333xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c1_i64 : !ascendc.local_tensor<333xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c1_i64 : !ascendc.local_tensor<333xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
func.func @reuse_tensor_with_static_shape(%arg0: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <333xi32>
  ascendc.data_copy_l2 %0, %arg0, %c1_i64 : !ascendc.local_tensor<333xi32>, !ascendc.global_tensor<?xi32>, i64
  %1 = ascendc.local_tensor_auto veccalc() : <333xi32>
  ascendc.data_copy_l2 %1, %arg0, %c1_i64 : !ascendc.local_tensor<333xi32>, !ascendc.global_tensor<?xi32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_with_different_tensor_types_and_shapes(
// CHECK:       scf.for %arg5 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <1000xi32>
// CHECK-NEXT:    %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<777xf32> to !ascendc.local_tensor<8xf16>
// CHECK-NEXT:    %3 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<777xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg0, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<777xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg2, %c8_i64 : !ascendc.local_tensor<8xf16>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg2, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf16>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg4, %c8_i64 : !ascendc.local_tensor<1000xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg4, %0, %c8_i64 : !ascendc.global_tensor<?xi32>, !ascendc.local_tensor<1000xi32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @reuse_with_different_tensor_types_and_shapes(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>, %arg3: !ascendc.global_tensor<?xf32>, %arg4: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  scf.for %arg5 = %c0 to %c32 step %c1 {
    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
    ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<777xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<777xf32>, i64
    %2 = ascendc.local_tensor_auto veccalc() : <8xf16>
    ascendc.data_copy_l2 %2, %arg2, %c8_i64 : !ascendc.local_tensor<8xf16>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg2, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf16>, i64
    %3 = ascendc.local_tensor_auto veccalc() : <1000xi32>
    ascendc.data_copy_l2 %3, %arg4, %c8_i64 : !ascendc.local_tensor<1000xi32>, !ascendc.global_tensor<?xi32>, i64
    ascendc.data_copy_l2 %arg4, %3, %c8_i64 : !ascendc.global_tensor<?xi32>, !ascendc.local_tensor<1000xi32>, i64
  }
  return
}

// CHECK-LABEL: func.func @noreuse_tensor_with_different_attributes(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() input : <8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() output : <8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
func.func @noreuse_tensor_with_different_attributes(%arg0: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %1 = ascendc.local_tensor_auto veccalc() input : <8xi32>
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %2 = ascendc.local_tensor_auto veccalc() output : <8xi32>
  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_tensor_because_op_does_not_create_new_tensor(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xindex>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
func.func @noreuse_tensor_because_op_does_not_create_new_tensor(%arg0: !ascendc.global_tensor<?xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xi32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xf32>
  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  %5 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xindex>
  ascendc.data_copy_l2 %4, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_two_tensors_in_data_copy(
// CHECK:       scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg2, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg2, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @reuse_two_tensors_in_data_copy(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c32 step %c1 {
    %alloca = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.data_copy_l2 %alloca, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg0, %alloca, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    %alloca_0 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.data_copy_l2 %alloca_0, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg1, %alloca_0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
    %alloca_1 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.data_copy_l2 %alloca_1, %arg2, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.data_copy_l2 %arg2, %alloca_1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}

// CHECK-LABEL: func.func @nofree_tmp_tensor_to_reuse(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
func.func @nofree_tmp_tensor_to_reuse() {
    %c1_i64 = arith.constant 1 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
    %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    return
}

// CHECK-LABEL: func.func @reuse_two_temporary_tensors(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %3, %4, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %2, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
func.func @reuse_two_temporary_tensors(%arg0: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  ascendc.add_l3 %4, %0, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  ascendc.data_copy_l2 %arg0, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_tensor_used_in_other_block(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %3, %4, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %2, %4, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
func.func @reuse_tensor_used_in_other_block(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg2, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_inside_loop(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
func.func @noreuse_inside_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.data_copy_l2 %1, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  ascendc.data_copy_l2 %arg2, %4, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_inside_condition(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @noreuse_inside_condition() {
  %c1_i64 = arith.constant 1 : i64
  %cond = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.if %cond {
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @noreuse_inside_while_loop(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  scf.while : () -> () {
// CHECK-NEXT:    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK:       } do {
// CHECK-NEXT:    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @noreuse_inside_while_loop() {
  %c1_i64 = arith.constant 1 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.while : () -> () {
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %true = arith.constant true
    scf.condition(%true)
  } do {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @reuse_only_in_one_region_inside_condition(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.add_l3 %2, %3, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @reuse_only_in_one_region_inside_condition() {
  %c1_i64 = arith.constant 1 : i64
  %cond = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  scf.if %cond {
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  return
}

// CHECK-LABEL: func.func @reuse_only_in_one_region_inside_while_loop(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.while : () -> () {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    scf.condition(%true)
// CHECK-NEXT:  } do {
// CHECK-NEXT:    ascendc.add_l3 %2, %3, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @reuse_only_in_one_region_inside_while_loop() {
  %c1_i64 = arith.constant 1 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  scf.while : () -> () {
    ascendc.add_l3 %3, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %cond = arith.constant true
    scf.condition(%cond)
  } do {
    ascendc.add_l3 %4, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @hoist_and_reuse_inside_loop(
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  scf.for %arg0 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %4 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func.func @hoist_and_reuse_inside_loop() {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1_i64 = arith.constant 1 : i64
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.for %arg0 = %c0 to %c32 step %c1 {
    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
      ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
      ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
    %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @hoist_and_reuse_inside_condition(
// CHECK:      %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %true = arith.constant true
// CHECK-NEXT: %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT: ascendc.add_l3 %4, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT: scf.if %true {
// CHECK-NEXT:   %6 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %1, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %6, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %6 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %0, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %6, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT: }
// CHECK-NEXT: ascendc.add_l3 %5, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
func.func @hoist_and_reuse_inside_condition() {
  %c1_i64 = arith.constant 1 : i64
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  scf.if %true {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @hoist_and_reuse_inside_while_loop(
// CHECK:      %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT: %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT: ascendc.add_l3 %4, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT: scf.while : () -> () {
// CHECK-NEXT:   %6 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %1, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %6, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   %true = arith.constant true
// CHECK-NEXT:   scf.condition(%true)
// CHECK-NEXT: } do {
// CHECK-NEXT:   %6 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %0, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   ascendc.add_l3 %6, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: ascendc.add_l3 %5, %3, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT: return
func.func @hoist_and_reuse_inside_while_loop() {
  %c1_i64 = arith.constant 1 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  scf.while : () -> () {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %true = arith.constant true
    scf.condition(%true)
  } do {
    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %4, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
    ascendc.add_l3 %5, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield
  }
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}
