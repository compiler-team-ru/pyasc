// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asclower-tensor -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.duplicate_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
// CHECK-NEXT:  return %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = tensor.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}
