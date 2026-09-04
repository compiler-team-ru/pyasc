// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s --ascendc-remove-debug-ops | FileCheck %s

// CHECK-LABEL: func.func @test_remove_debug_ops(%arg0: i1, %arg1: tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @test_remove_debug_ops(%arg0: i1, %arg1: tensor<32xf32, #asctile.local<UB>>) {
  ascendc.printf %arg0 {desc = "test print %d"} : i1
  asctile.assert %arg0, "test assert" : i1
  asctile.dump_tensor %arg1 : tensor<32xf32, #asctile.local<UB>>
  return
}
