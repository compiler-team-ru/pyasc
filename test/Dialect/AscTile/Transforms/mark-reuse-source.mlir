// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt --asctile-mark-reuse-source %s | FileCheck %s

// CHECK-LABEL: func.func public @test_mark
// CHECK: asctile.reduce <sum> %arg0
// CHECK-SAME: asctile.reuse_source
func.func public @test_mark(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %1 = asctile.reduce <sum> %arg0 {dims= [ 1 : i32]} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
  return %1 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @test_unmark
// CHECK: asctile.reduce <sum> %arg0 {dims = [1 : i32]} :
func.func public @test_unmark(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>) {
  %1 = asctile.reduce <sum> %arg0 {dims= [ 1 : i32]} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
  return %1, %arg0 : tensor<16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>
}
