// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-reuse-tensor-allocation %s | FileCheck %s

// CHECK-LABEL: func.func @works_without_crash() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @works_without_crash() {
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_different_tensor_types_and_shapes(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>, %arg3: !ascendc.global_tensor<?xf32>, %arg4: !ascendc.global_tensor<?xi32>) {
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  scf.for %arg5 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <1000xi32>
// CHECK-NEXT:    %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:    %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg1, %c8_i64 : !ascendc.local_tensor<777xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:    %3 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<1000xi32> to !ascendc.local_tensor<8xf16>
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg2, %c8_i64 : !ascendc.local_tensor<8xf16>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg2, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf16>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg4, %c8_i64 : !ascendc.local_tensor<1000xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:    ascendc.data_copy_l2 %arg4, %0, %c8_i64 : !ascendc.global_tensor<?xi32>, !ascendc.local_tensor<1000xi32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @reuse_tensor_with_different_attributes_no_loop(%arg0: !ascendc.global_tensor<?xi32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() output : <8xi32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_tensor_with_different_attributes_no_loop(%arg0: !ascendc.global_tensor<?xi32>) {
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

// CHECK-LABEL: func.func @noreuse_input_output_in_same_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() input : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() output : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.add_l3 %1, %0, %2 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_input_output_in_same_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() output : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.add_l3 %2, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}

// CHECK-LABEL: func.func @reuse_input_output_in_different_loops(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() output : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.add_l3 %0, %2, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:    ascendc.add_l3 %3, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.data_copy_l2 %arg1, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_input_output_in_different_loops(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() input : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() output : <8xf32>
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.add_l3 %3, %1, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %3, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  scf.for %arg3 = %c0 to %c32 step %c1 {
    ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
    ascendc.add_l3 %2, %2, %0 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    ascendc.data_copy_l2 %arg1, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
  }
  return
}

// CHECK-LABEL: func.func @noreuse_tensor_because_op_does_not_create_new_tensor(%arg0: !ascendc.global_tensor<?xi32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %2 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xi32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xindex>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xi32>, !ascendc.global_tensor<?xi32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @reuse_two_temporary_tensors(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %1, %4, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %0, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @reuse_tensor_used_in_other_block(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>, %arg2: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg1, %c8_i64 : !ascendc.local_tensor<8xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.data_copy_l2 %arg2, %2, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<8xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @hoist_and_reuse_inside_loop() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  scf.for %arg0 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:      ascendc.add_l3 %4, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.add_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %2, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %3, %0, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @hoist_and_reuse_inside_condition() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %4, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %6 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %6, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %0, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @hoist_and_reuse_inside_while_loop() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  ascendc.add_l3 %4, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.while : () -> () {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %6 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %6, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    scf.condition(%true)
// CHECK-NEXT:  } do {
// CHECK-NEXT:    %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %5, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    ascendc.add_l3 %0, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %1, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
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

// CHECK-LABEL: func.func @reuse_with_users_in_then_and_else_regions() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_users_in_then_and_else_regions() {
  %c1_i64 = arith.constant 1 : i64
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.if %true {
    ascendc.add_l3 %1, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    ascendc.add_l3 %2, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %4, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_deeply_nested_blocks() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    scf.for %arg0 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.for %arg0 = %c0 to %c32 step %c1 {
// CHECK-NEXT:      ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_deeply_nested_blocks() {
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.if %true {
    scf.for %arg2 = %c0 to %c32 step %c1 {
      ascendc.add_l3 %1, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
  } else {
    scf.for %arg2 = %c0 to %c32 step %c1 {
      ascendc.add_l3 %2, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    }
  }
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %4, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_with_users_in_while_before_and_after() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.while : () -> () {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    scf.condition(%true)
// CHECK-NEXT:  } do {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:    scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_with_users_in_while_before_and_after() {
  %c1_i64 = arith.constant 1 : i64
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.while : () -> () {
    ascendc.add_l3 %1, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    %true = arith.constant true
    scf.condition(%true)
  } do {
    ascendc.add_l3 %2, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
    scf.yield
  }
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %4, %0, %3 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-NEXT:func.func @reuse_multiple_tensors_nested_blocks() {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c32 = arith.constant 32 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %true = arith.constant true
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<8xf32> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  scf.for %arg0 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    ascendc.add_l3 %4, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.if %true {
// CHECK-NEXT:    ascendc.add_l3 %5, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    ascendc.add_l3 %0, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.add_l3 %2, %3, %1 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_multiple_tensors_nested_blocks() {
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %0 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
  scf.for %arg2 = %c0 to %c32 step %c1 {
    ascendc.add_l3 %1, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  scf.if %true {
    ascendc.add_l3 %2, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  } else {
    ascendc.add_l3 %3, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  }
  %5 = ascendc.local_tensor_auto veccalc() : <8xf32>
  ascendc.add_l3 %5, %0, %4 : !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>
  return
}

// CHECK-LABEL: func.func @reuse_tensors_from_the_same_memory(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto a1() : <16x16xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_tensors_from_the_same_memory(%arg0: !ascendc.global_tensor<?xf32>) {
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto a1() : <16x16xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  %1 = ascendc.local_tensor_auto a1() : <16x16xf32>
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  return
}

// CHECK-LABEL: func.func @noreuse_different_positions(%arg0: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto b2() : <16x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto a2() : <16x16xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @noreuse_different_positions(%arg0: !ascendc.global_tensor<?xf32>) {
  %c8_i64 = arith.constant 8 : i64
  %0 = ascendc.local_tensor_auto a2() : <16x16xf32>
  ascendc.data_copy_l2 %0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  %1 = ascendc.local_tensor_auto b2() : <16x16xf32>
  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %arg0, %1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<16x16xf32>, i64
  return
}

// CHECK-LABEL: func.func @reuse_matmul_pattern_no_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto co1() output : <64x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <16x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a2() input : <64x16xf32>
// CHECK-NEXT:  %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<64x16xf32> to !ascendc.local_tensor<64x16xf32>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %3, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %4, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.add_l3 %0, %3, %4 : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.data_copy_l2 %1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
// CHECK-NEXT:  ascendc.add_l3 %0, %2, %1 : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %arg1, %0, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<64x16xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @reuse_matmul_pattern_no_loop(%arg0: !ascendc.global_tensor<?xf32>, %arg1: !ascendc.global_tensor<?xf32>) {
  %c8_i64 = arith.constant 8 : i64
  %co1 = ascendc.local_tensor_auto co1() output : <64x16xf32>
  %a2_0 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 0 : i64} : <64x16xf32>
  %a2_1 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 1 : i64} : <64x16xf32>
  %b2_0 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 0 : i64} : <16x16xf32>
  %b2_1 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 1 : i64} : <16x16xf32>
  ascendc.data_copy_l2 %a2_0, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %b2_0, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.add_l3 %co1, %a2_0, %b2_0 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
  ascendc.data_copy_l2 %a2_1, %arg0, %c8_i64 : !ascendc.local_tensor<64x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.data_copy_l2 %b2_1, %arg0, %c8_i64 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?xf32>, i64
  ascendc.add_l3 %co1, %a2_1, %b2_1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<64x16xf32>, !ascendc.local_tensor<16x16xf32>
  ascendc.data_copy_l2 %arg1, %co1, %c8_i64 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<64x16xf32>, i64
  return
}

// CHECK-NEXT:func.func @matmul_v3_kernel(%arg0: !ascendc.mmad_params, %arg1: !ascendc.load_data_2d_params_v2, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.fixpipe_params_v220, %arg4: !ascendc.fixpipe_config) {
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto co1() output : <256x256xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto b2() input : <64x256xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto b2() input : <64x256xf16>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto b1() input output : <256x256xf16>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto b1() input output : <256x256xf16>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a2() input : <256x64xf16>
// CHECK-NEXT:  %6 = ascendc.local_tensor_auto a2() input : <256x64xf16>
// CHECK-NEXT:  %7 = ascendc.local_tensor_auto a1() input output : <256x256xf16>
// CHECK-NEXT:  %8 = ascendc.local_tensor_auto a1() input output : <256x256xf16>
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %9 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
// CHECK-NEXT:  %10 = ascendc.global_tensor : !ascendc.global_tensor<7680x4608xf16>
// CHECK-NEXT:  %11 = ascendc.global_tensor : !ascendc.global_tensor<1152x4608xf16>
// CHECK-NEXT:  %12 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<64x256xf16> to !ascendc.local_tensor<64x256xf16>
// CHECK-NEXT:  %13 = ascendc.reinterpret_cast %6 : !ascendc.local_tensor<256x64xf16> to !ascendc.local_tensor<256x64xf16>
// CHECK-NEXT:  %14 = ascendc.reinterpret_cast %1 : !ascendc.local_tensor<64x256xf16> to !ascendc.local_tensor<64x256xf16>
// CHECK-NEXT:  %15 = ascendc.reinterpret_cast %5 : !ascendc.local_tensor<256x64xf16> to !ascendc.local_tensor<256x64xf16>
// CHECK-NEXT:  scf.for %arg5 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    scf.for %arg6 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      ascendc.data_copy_l2 %7, %9, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      ascendc.data_copy_l2 %3, %10, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:        ascendc.load_data_l0_v2 %5, %7, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %1, %3, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %5, %1, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:        ascendc.load_data_l0_v2 %6, %7, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %2, %3, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %6, %2, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:      }
// CHECK-NEXT:      ascendc.data_copy_l2 %8, %9, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      ascendc.data_copy_l2 %4, %10, %arg2 : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
// CHECK-NEXT:        ascendc.load_data_l0_v2 %15, %8, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %14, %4, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %15, %14, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:        ascendc.load_data_l0_v2 %13, %8, %arg1 : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.load_data_l0_v2 %12, %4, %arg1 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:        ascendc.mmad %0, %13, %12, %arg0 : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.fixpipe %11, %0, %arg3, %arg4 : !ascendc.global_tensor<1152x4608xf16>, !ascendc.local_tensor<256x256xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @matmul_v3_kernel(%arg0: !ascendc.mmad_params, %arg1: !ascendc.load_data_2d_params_v2, %arg2: !ascendc.nd2nz_params, %arg3: !ascendc.fixpipe_params_v220, %arg4: !ascendc.fixpipe_config) {
  %c1_i32 = arith.constant 1 : i32
  %c256_i32 = arith.constant 256 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<7680x1152xf16>
  %1 = ascendc.global_tensor : !ascendc.global_tensor<7680x4608xf16>
  %2 = ascendc.global_tensor : !ascendc.global_tensor<1152x4608xf16>
  %3 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 1 : i64} : <64x256xf16>
  %4 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 1 : i64} : <256x64xf16>
  %5 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 0 : i64} : <64x256xf16>
  %6 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 0 : i64} : <256x64xf16>
  %7 = ascendc.local_tensor_auto b1() input output {asc.reuse_group = 1 : i64} : <256x256xf16>
  %8 = ascendc.local_tensor_auto a1() input output {asc.reuse_group = 1 : i64} : <256x256xf16>
  %9 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 1 : i64} : <64x256xf16>
  %10 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 1 : i64} : <256x64xf16>
  %11 = ascendc.local_tensor_auto b2() input {asc.reuse_group = 0 : i64} : <64x256xf16>
  %12 = ascendc.local_tensor_auto a2() input {asc.reuse_group = 0 : i64} : <256x64xf16>
  %13 = ascendc.local_tensor_auto b1() input output {asc.reuse_group = 0 : i64} : <256x256xf16>
  %14 = ascendc.local_tensor_auto a1() input output {asc.reuse_group = 0 : i64} : <256x256xf16>
  %15 = ascendc.local_tensor_auto co1() output : <256x256xf32>
  scf.for %arg5 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
    scf.for %arg6 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
      ascendc.data_copy_l2 %14, %0, %arg2 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
      ascendc.data_copy_l2 %13, %1, %arg2 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
        ascendc.load_data_l0_v2 %12, %14, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %11, %13, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %12, %11, %arg0 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
        ascendc.load_data_l0_v2 %10, %14, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %9, %13, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %10, %9, %arg0 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
      }
      ascendc.data_copy_l2 %8, %0, %arg2 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x1152xf16>, !ascendc.nd2nz_params
      ascendc.data_copy_l2 %7, %1, %arg2 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf16>, !ascendc.global_tensor<7680x4608xf16>, !ascendc.nd2nz_params
      scf.for %arg7 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
        ascendc.load_data_l0_v2 %6, %8, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %5, %7, %arg1 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %6, %5, %arg0 {asc.reuse_group = 0 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
        ascendc.load_data_l0_v2 %4, %8, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.load_data_l0_v2 %3, %7, %arg1 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<256x256xf16>, !ascendc.load_data_2d_params_v2
        ascendc.mmad %15, %4, %3, %arg0 {asc.reuse_group = 1 : i64} : !ascendc.local_tensor<256x256xf32>, !ascendc.local_tensor<256x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
      }
    }
    ascendc.fixpipe %2, %15, %arg3, %arg4 : !ascendc.global_tensor<1152x4608xf16>, !ascendc.local_tensor<256x256xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
  }
  return
}
