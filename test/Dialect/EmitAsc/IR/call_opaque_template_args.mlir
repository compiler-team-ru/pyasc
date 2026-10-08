// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s | ascir-opt | FileCheck %s

// The optional `template_args` attribute on emitasc.call_opaque round-trips
// for builtin attributes. The legacy string-callee form is unchanged.

// CHECK-LABEL: func.func @template_args
func.func @template_args(%a: i32) {
  // CHECK: emitasc.call_opaque "Foo" template_args [0 : i32, true](%{{.*}}) : (i32) -> i32
  %0 = emitasc.call_opaque "Foo" template_args [0 : i32, true](%a) : (i32) -> i32
  // Legacy string-callee form (no template_args) is unaffected.
  // CHECK: emitasc.call_opaque "Bar"(%{{.*}}) : (i32) -> i32
  %1 = emitasc.call_opaque "Bar"(%a) : (i32) -> i32
  return
}
