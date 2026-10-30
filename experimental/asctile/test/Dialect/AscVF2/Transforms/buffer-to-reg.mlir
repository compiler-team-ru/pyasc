// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-lower-to-reg %s | FileCheck %s

func.func @buffer_boundaries(%arg0: tensor<4xf32, #asctile.local<UB>>) -> tensor<4xf32, #asctile.local<UB>> attributes {ascvf2.vector_func} {
  %buffer = ascvf2.to_buffer %arg0 : tensor<4xf32, #asctile.local<UB>> to memref<4xf32, 26>
  %result = ascvf2.to_tensor %buffer : memref<4xf32, 26> to tensor<4xf32, #asctile.local<UB>>
  return %result : tensor<4xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @buffer_boundaries(
// CHECK: ascendc.local_tensor.get_phy_addr_v2
// CHECK-NOT: ascvf2.to_buffer
// CHECK-NOT: ascvf2.to_tensor
// CHECK-NOT: builtin.unrealized_conversion_cast {{.*}} to !ascendc.local_tensor<4xf32>
