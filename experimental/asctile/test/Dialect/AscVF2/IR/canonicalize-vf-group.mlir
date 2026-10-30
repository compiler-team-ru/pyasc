// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @erase_empty_group(
func.func @erase_empty_group(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  // CHECK-NEXT: return %arg0 : tensor<32xf32, #asctile.local<UB>>
  %0 = ascvf2.vf_group {
    ascvf2.yield %arg0 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @erase_unused_result(
func.func @erase_unused_result(%arg0: tensor<32xf32, #asctile.local<UB>>) {
  // CHECK-NEXT: return
  %0 = ascvf2.vf_group {
    %1 = arith.addf %arg0, %arg0 : tensor<32xf32, #asctile.local<UB>>
    ascvf2.yield %1 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return
}

// CHECK-LABEL: func.func @erase_forwarded_result(
func.func @erase_forwarded_result(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> (tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>) {
  // CHECK: %0 = ascvf2.vf_group {
  // CHECK: %1 = arith.addf %arg0, %arg0
  // CHECK: ascvf2.yield %1
  // CHECK: return %0, %arg1
  %0:2 = ascvf2.vf_group {
    %1 = arith.addf %arg0, %arg0 : tensor<32xf32, #asctile.local<UB>>
    ascvf2.yield %1, %arg1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0#0, %0#1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @erase_unused_reglower_result(
func.func @erase_unused_reglower_result(%arg0: !ascendc.local_tensor<32xf32>) {
  // CHECK-NEXT: return
  // CHECK-NEXT: }
  %0 = ascvf2.vf_group {
    ascvf2.yield %arg0 : !ascendc.local_tensor<32xf32>
  } : !ascendc.local_tensor<32xf32>
  return
}
