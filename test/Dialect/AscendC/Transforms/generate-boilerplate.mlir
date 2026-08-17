// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s --ascendc-generate-boilerplate --split-input-file | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT: emitc.include "kernel_operator.h"
// CHECK-NEXT: func.func @test_empty() {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_empty() {
    return
  }
}

// -----

// CHECK-LABEL: module {
// CHECK-NEXT: emitc.include "kernel_operator.h"
// CHECK-NEXT: emitc.include "kernel_operator_list_tensor_intf.h"
// CHECK-NEXT: func.func @test_list_tensor() {
// CHECK-NEXT: %0 = ascendc.list_tensor_desc : !ascendc.list_tensor_desc
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_list_tensor() {
    %0 = ascendc.list_tensor_desc : !ascendc.list_tensor_desc
    return
  }
}

// -----

!matmul = !ascendc.matmul<gm, 0 : i32, f32, false, 0 : i32, gm, 0 : i32, f32, false, 0 : i32, gm, 0 : i32, f32, false, 0 : i32, gm, 0 : i32, f32, <do_norm = false, do_basic_block = false, do_multi_data_load = false, basic_m = 0 : i32, basic_n = 0 : i32, basic_k = 0 : i32, intrinsics_check = false, is_n_batch = false, en_vec_nd2nz = false, do_special_basic_block = false, do_mte2_preload = 0 : i32, single_core_m = 0 : i32, single_core_n = 0 : i32, single_core_k = 0 : i32, step_m = 0 : i32, step_n = 0 : i32, base_mn = 0 : i32, single_core_mn = 0 : i32, en_unit_flag = false, is_per_tensor = false, has_anti_quant_offset = false, do_ib_share_norm = false, do_special_mdl = false, enable_init = false, batch_mode = 0 : i32, enable_end = false, enable_get_tensor_c = false, enable_set_org_shape = false, enable_set_bias = false, enable_set_tail = false, enable_quant_vector = false, enable_set_define_data = false, iterate_mode = 0 : i32, enable_reuse = false, enable_ub_reuse = false, enable_l1_cache_ub = false, intra_block_part_sum = false, iterate_order = 0 : i32, schedule_type = 0 : i32, enable_double_cache = false, is_bias_batch = false, enable_static_pad_zeros = false, is_partial_output = false, enable_mix_dual_master = false, is_a2b2_shared = false, is_enable_channel_split = false, enable_kdim_reorder_load = false, is_co1_shared = false, shared_co1_buffer_size = 0 : i32, batch_out_mode = 0 : i32>>

// CHECK-LABEL: module {
// CHECK-NEXT: emitc.include "kernel_operator.h"
// CHECK-NEXT: emitc.include "lib/matmul_intf.h"
// CHECK-NEXT: func.func @test_matmul(%arg0: !ascendc.pipe, %arg1: memref<1024xui8, 22>, %arg2: !ascendc.matmul<{{.+}}>) {
// CHECK-NEXT: ascendc.regist_matmul_obj %arg0, %arg1, %arg2 : !ascendc.pipe, memref<1024xui8, 22>, !ascendc.matmul<{{.+}}>
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_matmul(%arg0: !ascendc.pipe, %arg1: memref<1024xui8, 22>, %arg2: !matmul) {
    ascendc.regist_matmul_obj %arg0, %arg1, %arg2 : !ascendc.pipe, memref<1024xui8, 22>, !matmul
    return
  }
}
