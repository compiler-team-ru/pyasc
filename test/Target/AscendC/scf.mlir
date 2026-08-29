// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

func.func private @evaluate_condition(i32) -> i1
func.func private @evaluate_next_void() -> i32
func.func private @evaluate_next_i32(i32) -> i32
func.func private @evaluate_next_f32(f32) -> i32

// CHECK:void while_without_results(int32_t v1) {
// CHECK-NEXT:  int32_t v2 = v1;
// CHECK-NEXT:  while (true) {
// CHECK-NEXT:    bool v3 = evaluate_condition(v2);
// CHECK-NEXT:    if (!v3) {
// CHECK-NEXT:      break;
// CHECK-NEXT:    };
// CHECK-NEXT:    int32_t v4 = evaluate_next_void();
// CHECK-NEXT:    v2 = v4;
// CHECK-NEXT:  };
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @while_without_results(%arg0 : i32) {
  scf.while (%arg = %arg0) : (i32) -> () {
      %cond = func.call @evaluate_condition(%arg) : (i32) -> i1
      scf.condition(%cond)
  } do {
    %next = func.call @evaluate_next_void() : () -> i32
    scf.yield %next : i32
  }
  return
}

// CHECK:void while_with_results(int32_t v1, int32_t v2) {
// CHECK-NEXT:  int32_t v3;
// CHECK-NEXT:  int32_t v4;
// CHECK-NEXT:  int32_t v5 = v1;
// CHECK-NEXT:  int32_t v6 = v2;
// CHECK-NEXT:  while (true) {
// CHECK-NEXT:    bool v7 = evaluate_condition(v5);
// CHECK-NEXT:    if (!v7) {
// CHECK-NEXT:      v3 = v1;
// CHECK-NEXT:      v4 = v2;
// CHECK-NEXT:      break;
// CHECK-NEXT:    };
// CHECK-NEXT:    int32_t v8 = v1;
// CHECK-NEXT:    int32_t v9 = v2;
// CHECK-NEXT:    int32_t v10 = evaluate_next_i32(v8);
// CHECK-NEXT:    v5 = v10;
// CHECK-NEXT:    v6 = v9;
// CHECK-NEXT:  };
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @while_with_results(%arg0: i32, %arg1: i32) {
  %0:2 = scf.while (%arg = %arg0, %arg9 = %arg1) : (i32, i32) -> (i32, i32) {
      %cond = func.call @evaluate_condition(%arg) : (i32) -> i1
      scf.condition(%cond) %arg0, %arg1 : i32, i32
  } do {
  ^bb0(%arg2: i32, %arg3: i32):
    %next = func.call @evaluate_next_i32(%arg2) : (i32) -> i32
    scf.yield %next, %arg3 : i32, i32
  }
  return
}

// CHECK:void while_with_unmatching_edges(int32_t v1) {
// CHECK-NEXT:  float v2;
// CHECK-NEXT:  int32_t v3 = v1;
// CHECK-NEXT:  while (true) {
// CHECK-NEXT:    float [[FLOAT:.+]] = (float)1.230000020e+00;
// CHECK-NEXT:    bool v4 = evaluate_condition(v3);
// CHECK-NEXT:    if (!v4) {
// CHECK-NEXT:      v2 = [[FLOAT]];
// CHECK-NEXT:      break;
// CHECK-NEXT:    };
// CHECK-NEXT:    float v5 = [[FLOAT]];
// CHECK-NEXT:    int32_t v6 = evaluate_next_f32(v5);
// CHECK-NEXT:    v3 = v6;
// CHECK-NEXT:  };
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @while_with_unmatching_edges(%arg0: i32) {
  %0 = scf.while (%arg = %arg0) : (i32) -> (f32) {
      %1 = arith.constant 1.23 : f32
      %cond = func.call @evaluate_condition(%arg) : (i32) -> i1
      scf.condition(%cond) %1 : f32
  } do {
  ^bb0(%arg1: f32):
    %next = func.call @evaluate_next_f32(%arg1) : (f32) -> i32
    scf.yield %next : i32
  }
  return
}

// CHECK-LABEL: void index_switch_no_results(uint32_t v1) {
// CHECK-NEXT:  switch(v1) {
// CHECK-NEXT:  case 1: {
// CHECK-NEXT:    constexpr uint32_t c45_idx = 45;
// CHECK-NEXT:  } break;
// CHECK-NEXT:  case 2: {
// CHECK-NEXT:    constexpr uint32_t c67_idx = 67;
// CHECK-NEXT:  } break;
// CHECK-NEXT:  default: {
// CHECK-NEXT:    int32_t v2 = static_cast<int32_t>(v1);
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @index_switch_no_results(%arg0: index) {
  scf.index_switch %arg0
  case 1 {
    %0 = arith.constant 45 : index
    scf.yield
  }
  case 2 {
    %0 = arith.constant 67 : index
    scf.yield
  }
  default {
    %0 = arith.index_cast %arg0 : index to i32
    scf.yield
  }
  return
}

// CHECK-LABEL: void index_switch_one_result(uint32_t v1) {
// CHECK-NEXT:  int32_t v2;
// CHECK-NEXT:  switch(v1) {
// CHECK-NEXT:  case 1: {
// CHECK-NEXT:    constexpr int32_t c45_i32 = 45;
// CHECK-NEXT:    v2 = c45_i32;
// CHECK-NEXT:  } break;
// CHECK-NEXT:  case 2: {
// CHECK-NEXT:    constexpr int32_t c67_i32 = 67;
// CHECK-NEXT:    v2 = c67_i32;
// CHECK-NEXT:  } break;
// CHECK-NEXT:  default: {
// CHECK-NEXT:    int32_t v3 = static_cast<int32_t>(v1);
// CHECK-NEXT:    v2 = v3;
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @index_switch_one_result(%arg0: index) {
  scf.index_switch %arg0 -> i32
  case 1 {
    %0 = arith.constant 45 : i32
    scf.yield %0 : i32
  }
  case 2 {
    %0 = arith.constant 67 : i32
    scf.yield %0 : i32
  }
  default {
    %0 = arith.index_cast %arg0 : index to i32
    scf.yield %0 : i32
  }
  return
}

// CHECK-LABEL: void index_switch_multiple_results(uint32_t v1, int64_t v2) {
// CHECK-NEXT:  int32_t v3;
// CHECK-NEXT:  int64_t v4;
// CHECK-NEXT:  switch(v1) {
// CHECK-NEXT:  case 1: {
// CHECK-NEXT:    constexpr int32_t c45_i32 = 45;
// CHECK-NEXT:    constexpr int64_t c54_i64 = 54;
// CHECK-NEXT:    v3 = c45_i32;
// CHECK-NEXT:    v4 = c54_i64;
// CHECK-NEXT:  } break;
// CHECK-NEXT:  case 2: {
// CHECK-NEXT:    constexpr int64_t c67_i64 = 67;
// CHECK-NEXT:    int64_t v5 = v2 + c67_i64;
// CHECK-NEXT:    int32_t v6 = static_cast<int32_t>(v5);
// CHECK-NEXT:    v3 = v6;
// CHECK-NEXT:    v4 = v5;
// CHECK-NEXT:  } break;
// CHECK-NEXT:  default: {
// CHECK-NEXT:    int32_t v7 = static_cast<int32_t>(v1);
// CHECK-NEXT:    int64_t v8 = static_cast<int64_t>(v1);
// CHECK-NEXT:    v3 = v7;
// CHECK-NEXT:    v4 = v8;
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @index_switch_multiple_results(%arg0: index, %arg1: i64) {
  scf.index_switch %arg0 -> i32, i64
  case 1 {
    %0 = arith.constant 45 : i32
    %1 = arith.constant 54 : i64
    scf.yield %0, %1 : i32, i64
  }
  case 2 {
    %0 = arith.constant 67 : i64
    %1 = arith.addi %arg1, %0 : i64
    %2 = arith.trunci %1 : i64 to i32
    scf.yield %2, %1 : i32, i64
  }
  default {
    %0 = arith.index_cast %arg0 : index to i32
    %1 = arith.index_cast %arg0 : index to i64
    scf.yield %0, %1 : i32, i64
  }
  return
}

// CHECK-LABEL: void emit_for_op(int32_t v1) {
// CHECK-NEXT:  constexpr uint32_t c0_idx = 0;
// CHECK-NEXT:  constexpr uint32_t c10_idx = 10;
// CHECK-NEXT:  constexpr uint32_t c1_idx = 1;
// CHECK-NEXT:  int32_t v2;
// CHECK-NEXT:  int32_t v3 = v1;
// CHECK-NEXT:  for (uint32_t v4 = c0_idx; v4 < c10_idx; v4 += c1_idx) {
// CHECK-NEXT:    int32_t v5 = static_cast<int32_t>(v4);
// CHECK-NEXT:    int32_t v6 = v3 + v5;
// CHECK-NEXT:    v3 = v6;
// CHECK-NEXT:  }
// CHECK-NEXT:  v2 = v3;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_for_op(%init: i32) {
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %init) -> (i32) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %sum = arith.addi %iter, %iv_i32 : i32
    scf.yield %sum : i32
  }
  return
}

// CHECK-LABEL: void emit_if_op(int32_t v1, int32_t v2) {
// CHECK-NEXT:  bool v3 = v1 == v2;
// CHECK-NEXT:  int32_t v4;
// CHECK-NEXT:  if (v3) {
// CHECK-NEXT:    constexpr int32_t c1_i32 = 1;
// CHECK-NEXT:    v4 = c1_i32;
// CHECK-NEXT:  } else {
// CHECK-NEXT:    constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:    v4 = c0_i32;
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_if_op(%arg0: i32, %arg1: i32) {
  %cond = arith.cmpi eq, %arg0, %arg1 : i32
  %result = scf.if %cond -> (i32) {
    %c1 = arith.constant 1 : i32
    scf.yield %c1 : i32
  } else {
    %c0 = arith.constant 0 : i32
    scf.yield %c0 : i32
  }
  return
}
