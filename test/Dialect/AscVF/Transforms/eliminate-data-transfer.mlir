// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascvf-eliminate-data-transfer %s | FileCheck %s

// CHECK-LABEL: func.func @dont_create_load_for_loaded_reg_tensor
// CHECK:      ascvf.vf_for %11 : index {
// CHECK-NEXT: ^bb0(%arg7: index):
// CHECK-NEXT:   %13 = arith.muli %arg7, %8 : index
// CHECK-NEXT:   %14 = ascendc.update_mask f32, %10 : memref<1xui32>
// CHECK-NEXT:   ascvf.load %0, %arg2[%13] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
// CHECK-NEXT:   ascvf.load %1, %arg3[%13] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
// CHECK-NEXT:   ascendc.add_reg %2, %0, %1, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   ascvf.store %arg4[%13], %2, %14 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   %15 = arith.muli %arg7, %8 : index
// CHECK-NEXT:   ascendc.add_reg %5, %2, %0, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   ascvf.store %arg5[%15], %5, %14 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: }
func.func @dont_create_load_for_loaded_reg_tensor(%arg0: !ascendc.global_tensor<?x?xf32>, %arg1: !ascendc.data_copy_ext_params, %arg2: !ascendc.local_tensor<1x1024xf32>, %arg3: !ascendc.local_tensor<1x1024xf32>, %arg4: !ascendc.local_tensor<1x1024xf32>, %arg5: !ascendc.local_tensor<1x1024xf32>, %arg6: i64) {
  ascvf.vf_group %arg4, %arg5, %arg3, %arg2, %arg6 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64 {
    ascvf.vec_scope {
      %c4 = arith.constant 4 : index
      %0 = ascendc.reg_tensor : <f32>
      %1 = ascendc.reg_tensor : <f32>
      %2 = ascendc.reg_tensor : <f32>
      %3 = ascendc.reg_tensor : <f32>
      %4 = ascendc.reg_tensor : <f32>
      %5 = ascendc.reg_tensor : <f32>
      %6 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
      %7 = ascendc.get_vec_len : index
      %8 = arith.divsi %7, %c4 : index
      %9 = arith.index_cast %arg6 : i64 to index
      %10 = emitasc.variable %9 : index, memref<1xui32>
      %11 = arith.ceildivsi %9, %8 : index
      %12 = emitasc.variable %9 : index, memref<1xui32>
      ascvf.vf_for %11 : index {
      ^bb0(%arg7: index):
        %13 = arith.muli %arg7, %8 : index
        %14 = ascendc.update_mask f32, %10 : memref<1xui32>
        ascvf.load %0, %arg2[%13] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascvf.load %1, %arg3[%13] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.add_reg %2, %0, %1, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg4[%13], %2, %14 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        %15 = arith.muli %arg7, %8 : index
        ascvf.load %3, %arg4[%15] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascvf.load %4, %arg2[%15] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.add_reg %5, %3, %4, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%15], %5, %14 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
    }
  } {operandSegmentSizes = array<i32: 2, 2, 1>}
  ascendc.data_copy_pad_l2_ext %arg0, %arg5, %arg1 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}

// CHECK-LABEL: func.func @dont_rewrite_memory
// CHECK:      ascvf.vf_for %11 : index {
// CHECK-NEXT: ^bb0(%arg7: index):
// CHECK-NEXT:   %14 = arith.muli %arg7, %8 : index
// CHECK-NEXT:   %15 = ascendc.update_mask f32, %10 : memref<1xui32>
// CHECK-NEXT:   ascvf.load %0, %arg5[%14] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
// CHECK-NEXT:   ascendc.exp_reg %1, %0, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   %16 = arith.muli %arg7, %8 : index
// CHECK-NEXT:   ascendc.exp_reg %3, %1, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   %17 = arith.muli %arg7, %8 : index
// CHECK-NEXT:   ascendc.exp_reg %5, %3, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   ascvf.store %arg5[%17], %5, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: }
func.func @dont_rewrite_memory(%arg0: !ascendc.global_tensor<?x?xf32>, %arg1: !ascendc.data_copy_ext_params, %arg2: !ascendc.local_tensor<1x1024xf32>, %arg3: !ascendc.local_tensor<1x1024xf32>, %arg4: !ascendc.local_tensor<1x1024xf32>, %arg5: !ascendc.local_tensor<1x1024xf32>, %arg6: i64) {
  ascvf.vf_group %arg5, %arg5, %arg6 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64 {
    ascvf.vec_scope {
      %c4 = arith.constant 4 : index
      %0 = ascendc.reg_tensor : <f32>
      %1 = ascendc.reg_tensor : <f32>
      %2 = ascendc.reg_tensor : <f32>
      %3 = ascendc.reg_tensor : <f32>
      %4 = ascendc.reg_tensor : <f32>
      %5 = ascendc.reg_tensor : <f32>
      %6 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
      %7 = ascendc.get_vec_len : index
      %8 = arith.divsi %7, %c4 : index
      %9 = arith.index_cast %arg6 : i64 to index
      %10 = emitasc.variable %9 : index, memref<1xui32>
      %11 = arith.ceildivsi %9, %8 : index
      %12 = emitasc.variable %9 : index, memref<1xui32>
      %13 = emitasc.variable %9 : index, memref<1xui32>
      ascvf.vf_for %11 : index {
      ^bb0(%arg7: index):
        %14 = arith.muli %arg7, %8 : index
        %15 = ascendc.update_mask f32, %10 : memref<1xui32>
        ascvf.load %0, %arg5[%14] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.exp_reg %1, %0, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%14], %1, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        %16 = arith.muli %arg7, %8 : index
        ascvf.load %2, %arg5[%16] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.exp_reg %3, %2, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%16], %3, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        %17 = arith.muli %arg7, %8 : index
        ascvf.load %4, %arg5[%17] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.exp_reg %5, %4, %6 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %arg5[%17], %5, %15 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
    }
  } {operandSegmentSizes = array<i32: 1, 1, 1>}
  ascendc.data_copy_pad_l2_ext %arg0, %arg5, %arg1 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}

// CHECK-LABEL: func.func @replace_identical_loads
// CHECK:      ascvf.vec_scope {
// CHECK-NEXT:   %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:   %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:   %4 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:   %5 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:   %6 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:   ascendc.duplicate %2, %cst : <f32>, f32
// CHECK-NEXT:   %7 = emitasc.variable %c1024 : index, memref<1xui32>
// CHECK-NEXT:   ascvf.vf_for %c16 : index {
// CHECK-NEXT:   ^bb0(%arg1: index):
// CHECK-NEXT:     %8 = arith.muli %arg1, %c64 : index
// CHECK-NEXT:     %9 = ascendc.update_mask f32, %7 : memref<1xui32>
// CHECK-NEXT:   }
// CHECK-NEXT:   ascvf.vf_for %c16 : index {
// CHECK-NEXT:   ^bb0(%arg1: index):
// CHECK-NEXT:     %8 = arith.muli %arg1, %c64 : index
// CHECK-NEXT:     %9 = ascendc.update_mask f32, %7 : memref<1xui32>
// CHECK-NEXT:     ascendc.mul_reg %4, %2, %2, %9 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:     ascvf.store %0[%8], %4, %9 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   }
// CHECK-NEXT:   ascvf.vf_for %c16 : index {
// CHECK-NEXT:   ^bb0(%arg1: index):
// CHECK-NEXT:     %8 = arith.muli %arg1, %c64 : index
// CHECK-NEXT:     %9 = ascendc.update_mask f32, %7 : memref<1xui32>
// CHECK-NEXT:     ascvf.load %5, %0[%8] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
// CHECK-NEXT:     ascendc.add_reg %6, %5, %5, %9 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:     ascvf.store %0[%8], %6, %9 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @replace_identical_loads(%arg0: !ascendc.global_tensor<?x?xf32>) {
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  %c64 = arith.constant 64 : index
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %1 = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  ascvf.vf_group %0, %0, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64 {
    ascvf.vec_scope {
      %2 = ascendc.reg_tensor : <f32>
      %3 = ascendc.reg_tensor : <f32>
      %4 = ascendc.reg_tensor : <f32>
      %5 = ascendc.reg_tensor : <f32>
      %6 = ascendc.reg_tensor : <f32>
      ascendc.duplicate %2, %cst : <f32>, f32
      %7 = emitasc.variable %c1024 : index, memref<1xui32>
      ascvf.vf_for %c16 : index {
      ^bb0(%arg2: index):
        %8 = arith.muli %arg2, %c64 : index
        %9 = ascendc.update_mask f32, %7 : memref<1xui32>
        ascvf.store %0[%8], %2, %9 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
      ascvf.vf_for %c16 : index {
      ^bb0(%arg2: index):
        %8 = arith.muli %arg2, %c64 : index
        %9 = ascendc.update_mask f32, %7 : memref<1xui32>
        ascvf.load %3, %0[%8] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.mul_reg %4, %3, %3, %9 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %0[%8], %4, %9 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
      ascvf.vf_for %c16 : index {
      ^bb0(%arg2: index):
        %8 = arith.muli %arg2, %c64 : index
        %9 = ascendc.update_mask f32, %7 : memref<1xui32>
        ascvf.load %5, %0[%8] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1x1024xf32>, index
        ascendc.add_reg %6, %5, %5, %9 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
        ascvf.store %0[%8], %6, %9 : !ascendc.local_tensor<1x1024xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
      }
    }
  } {operandSegmentSizes = array<i32: 1, 1, 1>}
  ascendc.data_copy_pad_l2_ext %arg0, %0, %1 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}
