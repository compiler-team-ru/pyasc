// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @fold_to_buffer_of_to_tensor(
// CHECK-NOT: ascvf2.to_
// CHECK-NEXT: return %arg0
func.func @fold_to_buffer_of_to_tensor(%arg0: memref<8xf32, 26>) -> memref<8xf32, 26> {
  %tensor = ascvf2.to_tensor %arg0 : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  %buffer = ascvf2.to_buffer %tensor : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  return %buffer : memref<8xf32, 26>
}

// CHECK-LABEL: func.func @fold_to_tensor_of_to_buffer(
// CHECK-NOT: ascvf2.to_
// CHECK-NEXT: return %arg0
func.func @fold_to_tensor_of_to_buffer(%arg0: tensor<8xf32, #asctile.local<UB>>) -> tensor<8xf32, #asctile.local<UB>> {
  %buffer = ascvf2.to_buffer %arg0 : tensor<8xf32, #asctile.local<UB>> to memref<8xf32, 26>
  %tensor = ascvf2.to_tensor %buffer : memref<8xf32, 26> to tensor<8xf32, #asctile.local<UB>>
  return %tensor : tensor<8xf32, #asctile.local<UB>>
}
