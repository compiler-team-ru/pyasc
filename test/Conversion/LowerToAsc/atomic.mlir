// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asclower-atomic %s | FileCheck %s

// CHECK-LABEL: func.func @lower_atomic_add(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.global>, %arg2: i32) {
// CHECK-NEXT:  ascendc.set_atomic_add  {dtype = f32} :
// CHECK-NEXT:  asctile.store %arg0, %arg1[%arg2] : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.global>
// CHECK-NEXT:  ascendc.set_atomic_none
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_atomic_add(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.global>, %arg2: i32) {
  asctile.atomic_rmw <Add> %arg0, %arg1 [%arg2] : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_atomic_max(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.global>, %arg2: i32) {
// CHECK-NEXT:  ascendc.set_atomic_max  {dtype = f32} :
// CHECK-NEXT:  asctile.store %arg0, %arg1[%arg2] : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.global>
// CHECK-NEXT:  ascendc.set_atomic_none
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_atomic_max(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.global>, %arg2: i32) {
  asctile.atomic_rmw <Max> %arg0, %arg1 [%arg2] : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_atomic_min(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.global>, %arg2: i32) {
// CHECK-NEXT:  ascendc.set_atomic_min  {dtype = f32} :
// CHECK-NEXT:  asctile.store %arg0, %arg1[%arg2] : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.global>
// CHECK-NEXT:  ascendc.set_atomic_none
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_atomic_min(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.global>, %arg2: i32) {
  asctile.atomic_rmw <Min> %arg0, %arg1 [%arg2] : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.global>
  return
}
