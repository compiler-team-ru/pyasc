// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-reuse-tensor-allocation -debug-only=ascendc-reuse-tensor-allocation --mlir-disable-threading %s 2>&1 | FileCheck %s

// CHECK: Lifetimes:
// CHECK: tensor: %2 = ascendc.local_tensor_auto veccalc() : <43008xui8>
// CHECK: beginLife: ascendc.reduce_sum %3, %4, %2, %c64_i32, %c64_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168x64xf32>, !ascendc.local_tensor<43008xui8>, i32, i32
// CHECK: endLife: ascendc.reduce_sum %3, %4, %2, %c64_i32, %c64_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168x64xf32>, !ascendc.local_tensor<43008xui8>, i32, i32

// CHECK: tensor: %4 = ascendc.local_tensor_auto veccalc() input : <168x64xf32>
// CHECK: beginLife: ascendc.duplicate_l2 %4, %cst, %c168_i64 : !ascendc.local_tensor<168x64xf32>, f32, i64
// CHECK: endLife: ascendc.reduce_sum %3, %4, %2, %c64_i32, %c64_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168x64xf32>, !ascendc.local_tensor<43008xui8>, i32, i32

// CHECK: tensor: %0 = ascendc.local_tensor_auto veccalc() : <168xf32>
// CHECK: beginLife: ascendc.duplicate_l2 %0, %cst, %c168_i64 : !ascendc.local_tensor<168xf32>, f32, i64
// CHECK: endLife: scf.yield

// CHECK: tensor: %5 = ascendc.local_tensor_auto veccalc() output : <168xf32>
// CHECK: beginLife: ascendc.data_copy_l2 %5, %8, %c168_i64 : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, i64
// CHECK: endLife: ascendc.data_copy_pad_l2_ext %10, %5, %11 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<168xf32>, !ascendc.data_copy_ext_params

// CHECK: tensor: %1 = ascendc.local_tensor_auto veccalc() : <168xf32>
// CHECK: beginLife: %8 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg7 = %0) -> (!ascendc.local_tensor<168xf32>)  : i32
// CHECK: endLife: ascendc.data_copy_l2 %5, %8, %c168_i64 : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, i64

// CHECK: tensor: %3 = ascendc.local_tensor_auto veccalc() : <168xf32>
// CHECK: beginLife: ascendc.reduce_sum %3, %4, %2, %c64_i32, %c64_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168x64xf32>, !ascendc.local_tensor<43008xui8>, i32, i32
// CHECK: endLife: ascendc.add_l2 %1, %arg7, %3, %c168_i64 : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, i64
func.func @reduce_sum_rows(%arg0: memref<*xf32, 22>, %arg1: memref<*xf32, 22>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %false = arith.constant false
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c168_i64 = arith.constant 168 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.local_tensor_auto veccalc() : <168xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <168xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <43008xui8>
  %3 = ascendc.local_tensor_auto veccalc() : <168xf32>
  %4 = ascendc.local_tensor_auto veccalc() input : <168x64xf32>
  %5 = ascendc.local_tensor_auto veccalc() output : <168xf32>
  ascendc.duplicate_l2 %0, %cst, %c168_i64 : !ascendc.local_tensor<168xf32>, f32, i64
  %6 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
  %7 = ascendc.global_tensor : !ascendc.global_tensor<?xf32>
  scf.for %arg5 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
    %8 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg7 = %0) -> (!ascendc.local_tensor<168xf32>)  : i32 {
      %12 = ascendc.global_tensor.subindex %6[%c0_i32] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
      scf.if %false {
        ascendc.duplicate_l2 %4, %cst, %c168_i64 : !ascendc.local_tensor<168x64xf32>, f32, i64
      }
      scf.if %false {
        %15 = ascendc.local_tensor.subindex %4[%c64_i32] : !ascendc.local_tensor<168x64xf32>, i32, !ascendc.local_tensor<168x64xf32>
        ascendc.duplicate_l2 %15, %cst, %c64_i32 {asc.cal_count_set} : !ascendc.local_tensor<168x64xf32>, f32, i32
      }
      %13 = ascendc.construct !ascendc.data_copy_ext_params(%c1_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
      %14 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %c0_i32, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
      ascendc.data_copy_pad_l0_ext %4, %12, %13, %14 : !ascendc.local_tensor<168x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
      ascendc.reduce_sum %3, %4, %2, %c64_i32, %c64_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168x64xf32>, !ascendc.local_tensor<43008xui8>, i32, i32
      ascendc.add_l2 %1, %arg7, %3, %c168_i64 : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, i64
      scf.yield %1 : !ascendc.local_tensor<168xf32>
    }
    ascendc.data_copy_l2 %5, %8, %c168_i64 : !ascendc.local_tensor<168xf32>, !ascendc.local_tensor<168xf32>, i64
    %9 = arith.muli %arg5, %c64_i32 : i32
    %10 = ascendc.global_tensor.subindex %7[%9] : !ascendc.global_tensor<?xf32>, i32, !ascendc.global_tensor<?xf32>
    %11 = ascendc.construct !ascendc.data_copy_ext_params(%c1_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
    ascendc.data_copy_pad_l2_ext %10, %5, %11 : !ascendc.global_tensor<?xf32>, !ascendc.local_tensor<168xf32>, !ascendc.data_copy_ext_params
  }
  return
}
