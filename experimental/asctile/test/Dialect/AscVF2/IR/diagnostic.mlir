// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -split-input-file -verify-diagnostics %s

func.func @reject_nested_group(%arg0: vector<4xf32>) -> vector<4xf32> {
  // expected-error@+2 {{'ascvf2.vf_group' op cannot be nested in another ascvf2.vf_group}}
  %result = ascvf2.vf_group {
    %nested = ascvf2.vf_group {
      ascvf2.yield %arg0 : vector<4xf32>
    } : vector<4xf32>
    ascvf2.yield %nested : vector<4xf32>
  } : vector<4xf32>
  return %result : vector<4xf32>
}

// -----

func.func @reject_yield_count_mismatch(%arg0: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  // expected-error@+1 {{number of yield operands (1) must match number of results (2)}}
  %result:2 = ascvf2.vf_group {
    ascvf2.yield %arg0 : vector<4xf32>
  } : vector<4xf32>, vector<4xf32>
  return %result#0, %result#1 : vector<4xf32>, vector<4xf32>
}

// -----

func.func @reject_yield_type_mismatch(%arg0: vector<8xf32>) -> vector<4xf32> {
  // expected-error@+1 {{yield operand #0 type 'vector<8xf32>' does not match result type 'vector<4xf32>'}}
  %result = ascvf2.vf_group {
    ascvf2.yield %arg0 : vector<8xf32>
  } : vector<4xf32>
  return %result : vector<4xf32>
}

// -----

func.func @reject_non_ub_to_buffer_source(%arg0: tensor<4xf32, #asctile.local<L1>>) -> memref<4xf32, 26> {
  // expected-error@+1 {{source must be a UB local tensor}}
  %buffer = ascvf2.to_buffer %arg0 : tensor<4xf32, #asctile.local<L1>> to memref<4xf32, 26>
  return %buffer : memref<4xf32, 26>
}

// -----

func.func @reject_non_ub_to_tensor_source(%arg0: memref<4xf32>) -> tensor<4xf32, #asctile.local<UB>> {
  // expected-error@+1 {{source memory space must be UB}}
  %tensor = ascvf2.to_tensor %arg0 : memref<4xf32> to tensor<4xf32, #asctile.local<UB>>
  return %tensor : tensor<4xf32, #asctile.local<UB>>
}

// -----

func.func @reject_reduce_dimension(%arg0: vector<4xf32>) -> vector<1xf32> {
  // expected-error@+1 {{only reduction over vector dimension 0 is supported}}
  %reduced = "ascvf2.reduce"(%arg0) <{dims = [1 : i32], kind = #asctile.kind<sum>}> : (vector<4xf32>) -> vector<1xf32>
  return %reduced : vector<1xf32>
}

// -----

func.func @reject_simple_mask_both_counts(%arg0: index) -> !ascendc.mask_reg {
  // expected-error@+1 {{cannot specify both static and dynamic count}}
  %mask = ascvf2.simple_mask_reg f32 static 4 dynamic %arg0 : !ascendc.mask_reg
  return %mask : !ascendc.mask_reg
}
