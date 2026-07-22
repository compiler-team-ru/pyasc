// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void fuse(uint32_t v1)
// CHECK-NEXT: __VEC_SCOPE__
// CHECK-NEXT: {
// CHECK-NEXT:   for (uint16_t v2 = 0; v2 < static_cast<uint16_t>(v1); v2 += 1) {
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @fuse(%calCount : index) {
    ascvf.vec_scope {
        ascvf.vf_for %calCount : index {
        ^bb0(%arg0: index):
        }
    }
    return
}

// CHECK-LABEL: void emit_vf_group(uint32_t v1) {
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
func.func @emit_vf_group(%calCount : index) {
    ascvf.vf_group %calCount : index {
        ascvf.yield
    } {operandSegmentSizes = array<i32: 0, 0, 1>}
    return
}
