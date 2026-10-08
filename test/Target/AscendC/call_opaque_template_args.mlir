// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// Builtin template_args format as callee<...>(...). The legacy form stays a
// bare callee(...).

// CHECK-LABEL: void kernel(
// CHECK: Foo<0, true>(
// CHECK: Bar(
// CHECK: Bounds<4294967295, -1, true, false, half, TOKEN>(
module {
  func.func @kernel(%a: i32) {
    %0 = emitasc.call_opaque "Foo" template_args [0 : i32, true](%a) : (i32) -> i32
    %1 = emitasc.call_opaque "Bar"(%a) : (i32) -> i32
    %2 = emitasc.call_opaque "Bounds" template_args [4294967295 : ui32, -1 : i32, true, false, f16, #emitc.opaque<"TOKEN">](%a) : (i32) -> i32
    return
  }
}
