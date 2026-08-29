// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s | ascir-opt | FileCheck %s
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt | FileCheck %s

// CHECK-LABEL: test_local_tensor
func.func @test_local_tensor(
// CHECK-SAME: !ascendc.local_tensor<15xi32>
  %one_dim: !ascendc.local_tensor<15xi32>,
// CHECK-SAME: !ascendc.local_tensor<?xf32>
  %one_dim_dynamic: !ascendc.local_tensor<?xf32>,
// CHECK-SAME: !ascendc.local_tensor<1x23x45xf16>
  %many_dims: !ascendc.local_tensor<1x23x45xf16>,
// CHECK-SAME: !ascendc.local_tensor<58x?x78x900x?xi16>
  %many_dims_dynamic: !ascendc.local_tensor<58x?x78x900x?xi16>,
// CHECK-SAME: !ascendc.local_tensor<*xf16>
  %unranked: !ascendc.local_tensor<*xf16>
) {
  return
}

// CHECK-LABEL: test_global_tensor
func.func @test_global_tensor(
// CHECK-SAME: !ascendc.global_tensor<15xi32>
  %one_dim: !ascendc.global_tensor<15xi32>,
// CHECK-SAME: !ascendc.global_tensor<?xf32>
  %one_dim_dynamic: !ascendc.global_tensor<?xf32>,
// CHECK-SAME: !ascendc.global_tensor<1x23x45xf16>
  %many_dims: !ascendc.global_tensor<1x23x45xf16>,
// CHECK-SAME: !ascendc.global_tensor<58x?x78x900x?xi16>
  %many_dims_dynamic: !ascendc.global_tensor<58x?x78x900x?xi16>,
// CHECK-SAME: !ascendc.global_tensor<*xf16>
  %unranked: !ascendc.global_tensor<*xf16>
) {
  return
}

// CHECK-LABEL: test_types_with_position
func.func @test_types_with_position(
// CHECK-SAME: !ascendc.tbuf<gm>
  %buffer_gm: !ascendc.tbuf<gm>,
// CHECK-SAME: !ascendc.tbuf<vecin>
  %buffer_vecin: !ascendc.tbuf<vecin>,
// CHECK-SAME: !ascendc.que_bind<gm, veccalc, 2>
  %que_bind_gm: !ascendc.que_bind<gm, veccalc, 2>,
// CHECK-SAME: !ascendc.que_bind<veccalc, gm, 101>
  %que_bind_veccalc: !ascendc.que_bind<veccalc, gm, 101>,
// CHECK-SAME: !ascendc.queue<gm, 2>
  %queue_gm: !ascendc.queue<gm, 2>,
// CHECK-SAME: !ascendc.queue<vecin, 101>
  %queue_vecin: !ascendc.queue<vecin, 101>,
// CHECK-SAME: !ascendc.pipe
  %pipe: !ascendc.pipe
) {
  return
}

// 测试基础管道创建功能
// CHECK-LABEL: func @test_create_pipe
func.func @test_create_pipe() -> !ascendc.pipe {
  // CHECK: %[[PIPE:[0-9]+]] = ascendc.pipe
  %pipe = "ascendc.pipe"() : () -> !ascendc.pipe
  // CHECK: return %[[PIPE]] : !ascendc.pipe
  return %pipe : !ascendc.pipe
}

// CHECK-LABEL: func @test_create_buffer
func.func @test_create_buffer() -> !ascendc.tbuf<gm> {
  // CHECK: %[[BUF:[0-9]+]] = ascendc.tbuf{{.*}}<gm>
  %buf = "ascendc.tbuf"() : () -> !ascendc.tbuf<gm>
  // CHECK: return %[[BUF]] : !ascendc.tbuf<gm>
  return %buf : !ascendc.tbuf<gm>
}

// CHECK-LABEL: func @test_get_tensor_basic
func.func @test_get_tensor_basic(%arg0: !ascendc.tbuf<vecin>) -> !ascendc.local_tensor<?xf32> {
  %0 = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<?xf32>
  // CHECK: return %{{[0-9]+}} : !ascendc.local_tensor<?xf32>
  return %0 : !ascendc.local_tensor<?xf32>
}

// CHECK-LABEL: test_local_mem_allocator
func.func @test_local_mem_allocator(
// CHECK-SAME: !ascendc.local_mem_allocator<0>
  %allocator: !ascendc.local_mem_allocator<0>
) {
  return
}
