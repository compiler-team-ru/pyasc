// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -asctile-prepare-cv-strategy -split-input-file -verify-diagnostics %s

func.func @reduce_active_axis(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{cannot reduce the active split axis 0}}
    %1 = asctile.reduce <sum> %0 {dims = [0 : i32]} : tensor<32x64xf32, #asctile.local<UB>>, tensor<1x64xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @reshape_active_axis(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{must preserve the active split axis}}
    %1 = asctile.reshape %0 : tensor<32x64xf32, #asctile.local<UB>> to tensor<1x2048xf32, #asctile.local<UB>>
  }
  return
}

// -----

func.func @unsupported_consumer(%arg0: tensor<32x64xf32, #asctile.local<L0C>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{cannot be used inside the CV strategy body}}
    %1 = asctile.reduce_as_1d <sum> %0 : tensor<32x64xf32, #asctile.local<UB>>, f32
  }
  return
}

// -----

func.func @conflicting_split_shape(%arg0: tensor<32x64xf32, #asctile.local<L0C>>, %arg1: tensor<32x64xf32, #asctile.local<UB>>) {
  %c0_i32 = arith.constant 0 : i32
  asctile.cv_strategy <split_by_m> {
    %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<32x64xf32, #asctile.local<L0C>>, tensor<32x64xf32, #asctile.local<UB>>
    // expected-error@+1 {{has conflicting CV split shape requests}}
    %1 = arith.addf %0, %arg1 {asctile.need_split = #asctile.split_mode<split_by_m>, asctile.split_shape = array<i64: 8, 64>} : tensor<32x64xf32, #asctile.local<UB>>
  }
  return
}
