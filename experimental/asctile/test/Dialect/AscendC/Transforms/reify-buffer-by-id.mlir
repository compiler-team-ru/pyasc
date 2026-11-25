// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascendc-reify-buffer-by-id %s | FileCheck %s

// CHECK-LABEL: func.func @reify_shared_buffer_id() {
// CHECK-NEXT: %0 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT: %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<16xi32> to !ascendc.local_tensor<8xi32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @reify_shared_buffer_id() {
  %0 = ascendc.local_tensor_auto veccalc() : <8xi32> loc(fused<#asctile.buffer<7>>["reify-buffer-by-id.mlir":18:18])
  %1 = ascendc.local_tensor_auto veccalc() : <16xi32> loc(fused<#asctile.buffer<7>>["reify-buffer-by-id.mlir":19:19])
  return
}

// CHECK-LABEL: func.func @single_tensor_not_reified() {
// CHECK-NEXT: %0 = ascendc.local_tensor_auto veccalc() : <8xi32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @single_tensor_not_reified() {
  %0 = ascendc.local_tensor_auto veccalc() : <8xi32> loc(fused<#asctile.buffer<7>>["reify-buffer-by-id.mlir":28:28])
  return
}
