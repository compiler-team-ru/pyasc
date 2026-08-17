// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-legalize-kernel-args %s | FileCheck %s -check-prefixes=CHECK,CHECK-DEFAULT
// RUN: ascir-opt -ascendc-legalize-kernel-args="set-ffts-addr=true" %s | FileCheck %s -check-prefixes=CHECK,CHECK-FFTS

// CHECK-LABEL: func.func @test_simple_kernel(
// CHECK-SAME:  %arg0: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
// CHECK-SAME:  %arg1: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
// CHECK-DEFAULT-NEXT: return
// CHECK-FFTS-SAME:    %arg2: memref<?xui64, 22> {emitasc.kernel_arg = #emitasc<kernel_arg ffts_addr>}
// CHECK-FFTS-NEXT:    ascendc.set_ffts_base_addr %arg2 : memref<?xui64, 22>
// CHECK-FFTS-NEXT:    return
func.func @test_simple_kernel(%arg0: memref<*xf32, 22>, %arg1: memref<*xf32, 22>) attributes {ascendc.aicore, ascendc.global} {
  return
}

// CHECK-LABEL: func.func @test_non_kernel(
// CHECK-NEXT: return %arg0 : i32
func.func @test_non_kernel(%arg0: i32) -> i32 {
  return %arg0 : i32
}

!matmul = !ascendc.matmul<gm, 0 : i32, f32, false, 0 : i32, gm, 0 : i32, f32, false, 0 : i32, gm, 0 : i32, f32, false, 0 : i32, gm, 0 : i32, f32, <do_norm = false, do_basic_block = false, do_multi_data_load = false, basic_m = 0 : i32, basic_n = 0 : i32, basic_k = 0 : i32, intrinsics_check = false, is_n_batch = false, en_vec_nd2nz = false, do_special_basic_block = false, do_mte2_preload = 0 : i32, single_core_m = 0 : i32, single_core_n = 0 : i32, single_core_k = 0 : i32, step_m = 0 : i32, step_n = 0 : i32, base_mn = 0 : i32, single_core_mn = 0 : i32, en_unit_flag = false, is_per_tensor = false, has_anti_quant_offset = false, do_ib_share_norm = false, do_special_mdl = false, enable_init = false, batch_mode = 0 : i32, enable_end = false, enable_get_tensor_c = false, enable_set_org_shape = false, enable_set_bias = false, enable_set_tail = false, enable_quant_vector = false, enable_set_define_data = false, iterate_mode = 0 : i32, enable_reuse = false, enable_ub_reuse = false, enable_l1_cache_ub = false, intra_block_part_sum = false, iterate_order = 0 : i32, schedule_type = 0 : i32, enable_double_cache = false, is_bias_batch = false, enable_static_pad_zeros = false, is_partial_output = false, enable_mix_dual_master = false, is_a2b2_shared = false, is_enable_channel_split = false, enable_kdim_reorder_load = false, is_co1_shared = false, shared_co1_buffer_size = 0 : i32, batch_out_mode = 0 : i32>>

// CHECK-LABEL: func.func @test_matmul_not_cube_only(
// CHECK-SAME:  %arg0: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
// CHECK-SAME:  %arg1: !ascendc.pipe {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
// CHECK-SAME:  %arg2: memref<1024xui8, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}
// CHECK:       %0 = ascendc.ascend_is_aic : i1
// CHECK-NEXT:  scf.if %0 {
// CHECK-NEXT:    %c3873_i64 = arith.constant 3873 : i64
// CHECK-NEXT:    ascendc.ffts_cross_core_sync %c3873_i64 {pipe = 5 : i32} : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  ascendc.regist_matmul_obj %arg1, %arg2, %arg3
func.func @test_matmul_not_cube_only(%arg0: memref<*xf32, 22>, %arg1: !ascendc.pipe, %arg2: memref<1024xui8, 22>, %arg3: !matmul) attributes {ascendc.aicore, ascendc.global} {
  ascendc.regist_matmul_obj %arg1, %arg2, %arg3 : !ascendc.pipe, memref<1024xui8, 22>, !matmul
  return
}
