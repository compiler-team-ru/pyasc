// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascvf-find-vf-group -ascvf-lower-to-reg -cse -ascvf-reorder-ops-in-vec-scope -ascvf-fuse-vf-for -ascvf-eliminate-data-transfer -ascvf-eliminate-common-mask -ascvf-materialize-load-store %s | FileCheck %s

// CHECK-LABEL: func.func @general_test(%arg0: !ascendc.que_bind<gm, vecin, 1>) {
// CHECK:   ascvf.vf_group %0, %2, %1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32 {
// CHECK:     %3 = ascendc.local_tensor.get_phy_addr_v2 %1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK:     %4 = ascendc.local_tensor.get_phy_addr_v2 %2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK:     %5 = ascendc.local_tensor.get_phy_addr_v2 %0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK:     ascvf.vec_scope {
// CHECK:       %c4 = arith.constant 4 : index
// CHECK:       %6 = ascendc.reg_tensor : <f32>
// CHECK:       %7 = ascendc.reg_tensor : <f32>
// CHECK:       %8 = ascendc.reg_tensor : <f32>
// CHECK:       %9 = ascendc.reg_tensor : <f32>
// CHECK:       %10 = ascendc.reg_tensor : <f32>
// CHECK:       %11 = ascendc.reg_tensor : <f32>
// CHECK:       %12 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK:       %13 = ascendc.get_vec_len : index
// CHECK:       %14 = arith.divsi %13, %c4 : index
// CHECK:       %15 = arith.index_cast %c256_i32 : i32 to index
// CHECK:       %16 = arith.ceildivsi %15, %14 : index
// CHECK:       %17 = emitasc.variable %15 : index, memref<1xui32>
// CHECK:       %18 = emitasc.variable %15 : index, memref<1xui32>
// CHECK:       ascvf.vf_for %16 : index {
// CHECK:       ^bb0(%arg1: index):
// CHECK:         %19 = arith.muli %arg1, %14 : index
// CHECK:         %20 = ascendc.update_mask f32, %17 : memref<1xui32>
// CHECK:         %21 = emitasc.ptr_offset %3[%19] : memref<f32, 26>, memref<f32, 26>
// CHECK:         ascendc.data_copy_vld_reg %[[INP:.+]], %21 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK:         %22 = emitasc.ptr_offset %4[%19] : memref<f32, 26>, memref<f32, 26>
// CHECK:         ascendc.data_copy_vld_reg %7, %22 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK:         ascendc.add_reg %[[ADD:.+]], %[[INP]], %7, %12 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:         %23 = arith.muli %arg1, %14 : index
// CHECK:         ascendc.mul_reg %[[MUL:.+]], %[[ADD]], %7, %12 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:         %24 = emitasc.ptr_offset %5[%23] : memref<f32, 26>, memref<f32, 26>
// CHECK:         ascendc.data_copy_vst_reg %24, %[[MUL]], %20 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       }
// CHECK:     }
// CHECK:   } {operandSegmentSizes = array<i32: 1, 2, 1>}
// CHECK: }
func.func @general_test(%que_bind: !ascendc.que_bind<gm, vecin, 1>) {
  %c256_i32 = arith.constant 256 : i32
  %dst = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  %src0 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  %src1 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %dst, %src0, %src1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %dst, %dst, %src1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.que_bind.enque_tensor %que_bind, %dst : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.que_bind.free_tensor %que_bind, %src0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.que_bind.free_tensor %que_bind, %src1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  return
}

// CHECK-LABEL: func.func @softmax_kernel
// CHECK: ascvf.vf_group
// CHECK:   ascvf.vec_scope {
// CHECK:     ascendc.duplicate {{[^:]*}}: <f32>, f32
// CHECK:     ascendc.duplicate {{[^:]*}}: <f32>, f32
// CHECK:     ascvf.vf_for
// CHECK:       ascendc.data_copy_vld_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK:       ascendc.max_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     }
// CHECK:     ascendc.reduce_max_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     ascendc.duplicate_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     ascvf.vf_for
// CHECK:       ascendc.data_copy_vld_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK:       ascendc.sub_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.exp_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.data_copy_vst_reg {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.add_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     }
// CHECK:     ascendc.reduce_sum_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     ascendc.duplicate_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     ascvf.vf_for
// CHECK:       ascendc.data_copy_vld_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK:       ascendc.div_reg {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.data_copy_vst_reg {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     }
// CHECK:   }
// CHECK: }
// CHECK: ascendc.data_copy_pad_l2_ext {{[^:]*}}: !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
func.func @softmax_kernel(%arg0: memref<*xf32, 22>) {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
  %4 = ascendc.local_tensor_auto veccalc() output : <1x1024xf32>
  %5 = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %6 = ascendc.local_tensor_auto veccalc() : <16xf32>
  %7 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %f = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %z = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %y = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %11 = ascendc.local_tensor_auto veccalc() : <16xf32>
  %12 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %x = ascendc.local_tensor_auto veccalc() input : <1x1024xf32>

  %ext_params = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  ascendc.reduce_max_l2 %12, %x, %11, %c1024_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<16xf32>, i64, i64
  ascendc.duplicate_l2 %y, %12, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1xf32>, i64
  ascendc.sub_l2 %z, %x, %y, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.exp_l2 %f, %z, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.reduce_sum_l2 %7, %f, %6, %c1024_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<16xf32>, i64
  ascendc.duplicate_l2 %5, %7, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1xf32>, i64
  ascendc.div_l2 %4, %f, %5, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.data_copy_pad_l2_ext %0, %4, %ext_params : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}
