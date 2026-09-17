// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt %s --ascendc-compute-memory-consumption --split-input-file | FileCheck %s

// CHECK: asc.memory_consumed = {UB = 1536 : i64}
module {
  func.func @test_ub() {
    %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
    %1 = ascendc.local_tensor_v3 veccalc, 1024, 128 : !ascendc.local_tensor<32xf32>
    return
  }
}

// -----

// CHECK: asc.memory_consumed = {L0A = 256 : i64, L1 = 512 : i64, UB = 1024 : i64}
module {
  func.func @test_mixed() {
    %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
    %1 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<32xf32>
    %2 = ascendc.local_tensor_v3 a2, 0, 64 : !ascendc.local_tensor<16xf32>
    return
  }
}

// -----

// CHECK: asc.memory_consumed = {}
module {
  func.func @test_empty() {
    return
  }
}

// -----

// CHECK: asc.memory_consumed = {BT = 256 : i64, L0A = 512 : i64, L0B = 256 : i64, L0C = 512 : i64, L1 = 768 : i64}
module {
  func.func @test_tbuf_all_positions() {
    %pipe = ascendc.pipe
    %buf_a1 = ascendc.tbuf : !ascendc.tbuf<a1>
    %buf_b1 = ascendc.tbuf : !ascendc.tbuf<b1>
    %buf_a2 = ascendc.tbuf : !ascendc.tbuf<a2>
    %buf_b2 = ascendc.tbuf : !ascendc.tbuf<b2>
    %buf_co1 = ascendc.tbuf : !ascendc.tbuf<co1>
    %buf_c2 = ascendc.tbuf : !ascendc.tbuf<c2>
    %c512 = arith.constant 512 : i32
    %c256 = arith.constant 256 : i32
    ascendc.pipe.init_buffer %pipe, %buf_a1, %c512 : !ascendc.tbuf<a1>, i32
    ascendc.pipe.init_buffer %pipe, %buf_b1, %c256 : !ascendc.tbuf<b1>, i32
    ascendc.pipe.init_buffer %pipe, %buf_a2, %c512 : !ascendc.tbuf<a2>, i32
    ascendc.pipe.init_buffer %pipe, %buf_b2, %c256 : !ascendc.tbuf<b2>, i32
    ascendc.pipe.init_buffer %pipe, %buf_co1, %c512 : !ascendc.tbuf<co1>, i32
    ascendc.pipe.init_buffer %pipe, %buf_c2, %c256 : !ascendc.tbuf<c2>, i32
    return
  }
}

// -----

// CHECK: asc.memory_consumed = {BT = 256 : i64, L0A = 1024 : i64, L0B = 256 : i64, L0C = 1024 : i64, L1 = 1280 : i64}
module {
  func.func @test_queue_all_positions() {
    %pipe = ascendc.pipe
    %q_a1 = ascendc.queue : !ascendc.queue<a1, 1>
    %q_b1 = ascendc.queue : !ascendc.queue<b1, 1>
    %q_a2 = ascendc.queue : !ascendc.queue<a2, 1>
    %q_b2 = ascendc.queue : !ascendc.queue<b2, 1>
    %q_co1 = ascendc.queue : !ascendc.queue<co1, 1>
    %q_c2 = ascendc.queue : !ascendc.queue<c2, 1>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c512 = arith.constant 512 : i64
    %c256 = arith.constant 256 : i64
    ascendc.pipe.init_queue %pipe, %q_a1, %c2, %c512 : !ascendc.queue<a1, 1>, i32, i64
    ascendc.pipe.init_queue %pipe, %q_b1, %c1, %c256 : !ascendc.queue<b1, 1>, i32, i64
    ascendc.pipe.init_queue %pipe, %q_a2, %c2, %c512 : !ascendc.queue<a2, 1>, i32, i64
    ascendc.pipe.init_queue %pipe, %q_b2, %c1, %c256 : !ascendc.queue<b2, 1>, i32, i64
    ascendc.pipe.init_queue %pipe, %q_co1, %c2, %c512 : !ascendc.queue<co1, 1>, i32, i64
    ascendc.pipe.init_queue %pipe, %q_c2, %c1, %c256 : !ascendc.queue<c2, 1>, i32, i64
    return
  }
}

// -----

// CHECK: asc.memory_consumed = {UB = 512 : i64}
module {
  func.func @test_que_bind() {
    %pipe = ascendc.pipe
    %qbind = ascendc.que_bind : !ascendc.que_bind<veccalc, co1, 1>
    %c2 = arith.constant 2 : i32
    %c256 = arith.constant 256 : i64
    ascendc.pipe.init_queue %pipe, %qbind, %c2, %c256 : !ascendc.que_bind<veccalc, co1, 1>, i32, i64
    return
  }
}

// -----

// CHECK: asc.memory_consumed = {UB = 2048 : i64}
module {
  func.func @test_combined_queue_and_local_tensor() {
    %pipe = ascendc.pipe
    %buf = ascendc.tbuf : !ascendc.tbuf<veccalc>
    %queue = ascendc.queue : !ascendc.queue<veccalc, 1>
    %c1 = arith.constant 1 : i32
    %c512_32 = arith.constant 512 : i32
    %c512_64 = arith.constant 512 : i64
    %0 = ascendc.local_tensor_v3 veccalc, 0, 256 : !ascendc.local_tensor<64xf32>
    ascendc.pipe.init_queue %pipe, %queue, %c1, %c512_64 : !ascendc.queue<veccalc, 1>, i32, i64
    ascendc.pipe.init_buffer %pipe, %buf, %c512_32 : !ascendc.tbuf<veccalc>, i32
    return
  }
}
