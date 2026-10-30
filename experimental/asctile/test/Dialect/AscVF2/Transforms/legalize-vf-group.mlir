// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-legalize-vf-group %s | FileCheck %s

module {

// CHECK-LABEL: func.func @hoist_static_local_tensor_auto
// CHECK-NEXT: %[[TENSOR:.*]] = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT: %[[GROUP:.*]] = ascvf2.vf_group {
// CHECK-NEXT: ascvf2.yield %[[TENSOR]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GROUP]]
func.func @hoist_static_local_tensor_auto() -> !ascendc.local_tensor<32xf32> {
  %0 = ascvf2.vf_group {
    %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
    ascvf2.yield %1 : !ascendc.local_tensor<32xf32>
  } : !ascendc.local_tensor<32xf32>
  return %0 : !ascendc.local_tensor<32xf32>
}

// CHECK-LABEL: func.func @keep_dynamic_local_tensor_auto
// CHECK-NEXT: %[[SHAPE:.*]] = arith.constant 32 : i64
// CHECK-NEXT: %[[GROUP:.*]] = ascvf2.vf_group {
// CHECK-NEXT: %[[TENSOR:.*]] = ascendc.local_tensor_auto veccalc(%[[SHAPE]]) : <32xf32>
// CHECK-NEXT: ascvf2.yield %[[TENSOR]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[GROUP]]
func.func @keep_dynamic_local_tensor_auto() -> !ascendc.local_tensor<32xf32> {
  %shape = arith.constant 32 : i64
  %0 = ascvf2.vf_group {
    %1 = ascendc.local_tensor_auto veccalc(%shape) : <32xf32>
    ascvf2.yield %1 : !ascendc.local_tensor<32xf32>
  } : !ascendc.local_tensor<32xf32>
  return %0 : !ascendc.local_tensor<32xf32>
}

}
