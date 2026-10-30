// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-shrink-vf-signature %s | FileCheck %s

// CHECK-LABEL: func.func @kernel(
// CHECK: %0:2 = ascvf2.vf_group {
// CHECK: %1:2 = func.call @kernel_vf1(%arg0, %arg1)
// CHECK: ascvf2.yield %1#0, %1#1
// CHECK: return %0#0, %0#1
func.func @kernel(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) {
  %group:2 = ascvf2.vf_group {
    %call:5 = func.call @kernel_vf1(%arg0, %arg1) : (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>)
    ascvf2.yield %call#1, %call#4 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
  } : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
  return %group#0, %group#1 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func private @kernel_vf1(
// CHECK-SAME: %arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>
// CHECK-SAME: attributes {ascvf2.vector_func}
// CHECK: %0 = arith.addf %arg0, %arg0
// CHECK: %1 = arith.subf %0, %arg1
// CHECK: %2 = arith.mulf %1, %arg1
// CHECK: %3 = arith.divf %2, %arg1
// CHECK: %4 = arith.maximumf %3, %arg1
// CHECK: return %1, %4
func.func private @kernel_vf1(%arg0: tensor<32x32xf32, #asctile.local<UB>>, %arg1: tensor<32x32xf32, #asctile.local<UB>>) -> (tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>) attributes {ascvf2.vector_func} {
  %0 = arith.addf %arg0, %arg0 : tensor<32x32xf32, #asctile.local<UB>>
  %1 = arith.subf %0, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %2 = arith.mulf %1, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %3 = arith.divf %2, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  %4 = arith.maximumf %3, %arg1 : tensor<32x32xf32, #asctile.local<UB>>
  return %0, %1, %2, %3, %4 : tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.local<UB>>
}
