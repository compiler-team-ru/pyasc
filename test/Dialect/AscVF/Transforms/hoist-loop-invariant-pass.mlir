// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascvf-hoist-loop-invariant-pass %s | FileCheck %s

// CHECK-LABEL: func.func @test_scalar_code
// CHECK:      ascendc.reduce_max_reg %6, %7, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascvf.store %1[%c0], %6, %16 : !ascendc.local_tensor<1xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.duplicate_reg %11, %6, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.duplicate_reg %13, %cst, %15 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
// CHECK-NEXT: ascendc.div_reg %14, %11, %13, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascvf.vf_for %25 : index {
// CHECK-NEXT: ^bb0(%arg0: index):
// CHECK-NEXT:   %27 = arith.muli %arg0, %18 : index
// CHECK-NEXT:   %28 = ascendc.update_mask f32, %24 : memref<1xui32>
// CHECK-NEXT:   %29 = arith.muli %arg0, %18 : index
// CHECK-NEXT:   ascvf.store %3[%29], %14, %28 : !ascendc.local_tensor<1x4096xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: }
func.func @test_scalar_code() {
  %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
  %3 = ascendc.local_tensor_auto veccalc() : <1x4096xf32>
  %c4096_i64 = arith.constant 4096 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %4 = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  ascvf.vf_group %1, %3, %3, %c4096_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x4096xf32>, !ascendc.local_tensor<1x4096xf32>, i64 {
    ascvf.vec_scope {
      %c4 = arith.constant 4 : index
      %cst_0 = arith.constant 0xFF800000 : f32
      %c0 = arith.constant 0 : index
      %5 = ascendc.reg_tensor : <f32>
      %6 = ascendc.reg_tensor : <f32>
      %7 = ascendc.reg_tensor : <f32>
      %8 = ascendc.reg_tensor : <f32>
      %9 = ascendc.reg_tensor : <f32>
      %10 = ascendc.reg_tensor : <f32>
      %11 = ascendc.reg_tensor : <f32>
      %12 = ascendc.reg_tensor : <f32>
      %13 = ascendc.reg_tensor : <f32>
      %14 = ascendc.reg_tensor : <f32>
      %15 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
      %16 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
      ascendc.duplicate %7, %cst_0 : <f32>, f32
      %17 = ascendc.get_vec_len : index
      %18 = arith.divsi %17, %c4 : index
      %19 = arith.index_cast %c4096_i64 : i64 to index
      %20 = arith.divsi %19, %18 : index
      %21 = emitasc.variable %19 : index, memref<1xui32>
      %22 = arith.remsi %19, %18 : index
      %23 = arith.cmpi ne, %22, %c0 : index
      %24 = emitasc.variable %19 : index, memref<1xui32>
      %25 = arith.ceildivsi %19, %18 : index
      %26 = emitasc.variable %19 : index, memref<1xui32>
      ascvf.vf_for %20 : index {
      ^bb0(%arg0: index):
        %27 = arith.muli %arg0, %18 : index
        %28 = ascendc.update_mask f32, %21 : memref<1xui32>
        ascvf.load %5, %3[%27] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x4096xf32>, index
        ascendc.max_reg %7, %7, %5, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
      ascendc.reduce_max_reg %6, %7, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      ascvf.store %1[%c0], %6, %16 : !ascendc.local_tensor<1xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      ascendc.duplicate_reg %11, %6, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      ascvf.vf_for %25 : index {
      ^bb0(%arg0: index):
        %27 = arith.muli %arg0, %18 : index
        %28 = ascendc.update_mask f32, %24 : memref<1xui32>
        %29 = arith.muli %arg0, %18 : index
        ascendc.duplicate_reg %13, %cst, %15 : !ascendc.reg_tensor<f32>, f32, !ascendc.mask_reg
        ascendc.div_reg %14, %11, %13, %15 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %3[%29], %14, %28 : !ascendc.local_tensor<1x4096xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
    }
  } {operandSegmentSizes = array<i32: 2, 1, 1>}
  ascendc.data_copy_pad_l2_ext %0, %3, %4 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x4096xf32>, !ascendc.data_copy_ext_params
  return
}

