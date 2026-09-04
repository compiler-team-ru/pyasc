// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_printf(int32_t v1) {
// CHECK-NEXT:  AscendC::printf("test print %d", v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_printf(%c1_i32 : i32) {
  ascendc.printf %c1_i32 {desc = "test print %d"} : i32
  return
}

// CHECK-LABEL: void emit_printf_escaped(int32_t v1) {
// CHECK-NEXT:   AscendC::printf("q:\" n:\n t:\t b:\\ %d\n", v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_printf_escaped(%c1_i32 : i32) {
  ascendc.printf %c1_i32 {desc = "q:\22 n:\0A t:\09 b:\\ %d\0A"} : i32
  return
}

// CHECK-LABEL:void emit_dump_tensor_global(AscendC::GlobalTensor<float> v1, uint32_t v2, uint32_t v3) {
// CHECK-NEXT:  AscendC::DumpTensor(v1, v2, v3);
// CHECK-NEXT:  AscendC::ShapeInfo v4;
// CHECK-NEXT:  AscendC::DumpTensor(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_dump_tensor_global(%arg0: !ascendc.global_tensor<*xf32>, %c1_ui32 : ui32, %c2_ui32 : ui32) {
  ascendc.dump_tensor %arg0, %c1_ui32, %c2_ui32: !ascendc.global_tensor<*xf32>, ui32, ui32
  %shape_info = ascendc.construct !ascendc.shape_info() []
  ascendc.dump_tensor %arg0, %c1_ui32, %c2_ui32, %shape_info: !ascendc.global_tensor<*xf32>, ui32, ui32, !ascendc.shape_info
  return
}

// CHECK-LABEL:void emit_dump_tensor_local(AscendC::LocalTensor<float> v1, uint32_t v2, uint32_t v3) {
// CHECK-NEXT:  AscendC::DumpTensor(v1, v2, v3);
// CHECK-NEXT:  AscendC::ShapeInfo v4;
// CHECK-NEXT:  AscendC::DumpTensor(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_dump_tensor_local(%arg0: !ascendc.local_tensor<*xf32>, %c1_ui32 : ui32, %c2_ui32 : ui32) {
  ascendc.dump_tensor %arg0, %c1_ui32, %c2_ui32: !ascendc.local_tensor<*xf32>, ui32, ui32
  %shape_info = ascendc.construct !ascendc.shape_info() []
  ascendc.dump_tensor %arg0, %c1_ui32, %c2_ui32, %shape_info: !ascendc.local_tensor<*xf32>, ui32, ui32, !ascendc.shape_info
  return
}


// CHECK-LABEL:void emit_print_time_stamp(uint32_t v1) {
// CHECK-NEXT:  AscendC::PrintTimeStamp(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_print_time_stamp(%arg0: ui32) {
  ascendc.print_time_stamp %arg0: ui32
  return
}

// CHECK-LABEL:void emit_dump_acc_chk_point(AscendC::GlobalTensor<float> v1, uint32_t v2, uint32_t v3, uint32_t v4) {
// CHECK-NEXT:  AscendC::DumpAccChkPoint(v1, v2, v3, v4);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_dump_acc_chk_point(%arg0: !ascendc.global_tensor<*xf32>, %c1_ui32 : ui32, %c2_ui32 : ui32, %c3_ui32 : ui32) {
  ascendc.dump_acc_chk_point %arg0, %c1_ui32, %c2_ui32, %c3_ui32 : !ascendc.global_tensor<*xf32>, ui32, ui32, ui32
  return
}

// CHECK-LABEL:void emit_metrics_prof_start() {
// CHECK-NEXT:  AscendC::MetricsProfStart(); 
// CHECK-NEXT:  return; 
// CHECK-NEXT: }
func.func @emit_metrics_prof_start() {
  ascendc.metrics_prof_start
  return
}

// CHECK-LABEL:void emit_metrics_prof_stop() {
// CHECK-NEXT:  AscendC::MetricsProfStop(); 
// CHECK-NEXT:  return; 
// CHECK-NEXT: }
func.func @emit_metrics_prof_stop() {
  ascendc.metrics_prof_stop
  return
}
