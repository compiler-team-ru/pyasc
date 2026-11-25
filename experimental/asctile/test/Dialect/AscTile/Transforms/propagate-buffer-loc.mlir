// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt --mlir-print-debuginfo -asctile-propagate-buffer-loc %s | FileCheck %s

// CHECK-LABEL: func.func @propagate_buffer_location(%arg0: tensor<32xf32, #asctile.local<UB>> loc(fused<#asctile.buffer<7>>[#loc2])) -> tensor<32xf32, #asctile.local<L1>> {
// CHECK-NEXT: %cast = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<L1>> loc(#loc4)
// CHECK-NEXT: return %cast : tensor<32xf32, #asctile.local<L1>> loc(#loc3)
// CHECK-NEXT:}
func.func @propagate_buffer_location(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<L1>> {
  %cast = tensor.cast %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<L1>> loc(fused<#asctile.buffer<7>>["propagate-buffer-loc.mlir":18:18])
  return %cast : tensor<32xf32, #asctile.local<L1>>
}
