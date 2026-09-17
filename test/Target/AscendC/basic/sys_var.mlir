// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_get_block_info() {
// CHECK-NEXT:  uint32_t v1 = AscendC::GetBlockIdx();
// CHECK-NEXT:  int64_t v2 = AscendC::GetBlockIdx();
// CHECK-NEXT:  uint32_t v3 = AscendC::GetBlockNum();
// CHECK-NEXT:  int32_t v4 = AscendC::GetBlockNum();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_get_block_info() {
  %0 = ascendc.get_block_idx : index
  %1 = ascendc.get_block_idx : i64
  %2 = ascendc.get_block_num : index
  %3 = ascendc.get_block_num : i32
  return
}

// CHECK-LABEL:void emit_get_sub_block_info(__gm__ uint64_t* v1) {
// CHECK-NEXT:  set_ffts_base_addr(*v1);
// CHECK-NEXT:  int32_t v2 = AscendC::GetSubBlockIdx();
// CHECK-NEXT:  int32_t v3 = AscendC::GetSubBlockNum();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_sub_block_info(%arg0: memref<?xui64, 22>) {
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = ascendc.get_sub_block_idx : i32
  %1 = ascendc.get_sub_block_num : i32
  return
}

// CHECK-LABEL:void emit_get_arch_version(int32_t v1) {
// CHECK-NEXT:  uint32_t v2 = (uint32_t) v1;
// CHECK-NEXT:  AscendC::GetArchVersion(v2);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_get_arch_version(%arg0 : i32) {
  %v1 = emitc.cast %arg0 : i32 to ui32
  ascendc.get_arch_version %v1 : ui32
  return
}


// CHECK-LABEL: void emit_get_data_block_size_in_bytes() {
// CHECK-NEXT:   int16_t v1 = AscendC::GetDataBlockSizeInBytes();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_get_data_block_size_in_bytes() {
  %0 = ascendc.get_data_block_size_in_bytes : i16
  return
}

// CHECK-LABEL:void emit_get_program_counter() {
// CHECK-NEXT:  int32_t v1 = AscendC::GetProgramCounter();
// CHECK-NEXT:  int64_t v2 = AscendC::GetProgramCounter();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_program_counter() {
  %0 = ascendc.get_program_counter : i32
  %1 = ascendc.get_program_counter : i64
  return
}

// CHECK-LABEL:void emit_get_system_cycle() {
// CHECK-NEXT:  int32_t v1 = AscendC::GetSystemCycle();
// CHECK-NEXT:  int64_t v2 = AscendC::GetSystemCycle();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_system_cycle() {
  %0 = ascendc.get_system_cycle : i32
  %1 = ascendc.get_system_cycle : i64
  return
}

// CHECK-LABEL:void emit_get_task_ratio(__gm__ uint64_t* v1) {
// CHECK-NEXT:  set_ffts_base_addr(*v1);
// CHECK-NEXT:  int64_t v2 = AscendC::GetTaskRatio();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_task_ratio(%arg0: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = ascendc.get_task_ratio : i64
  return
}

// CHECK-LABEL:void emit_trap() {
// CHECK-NEXT:  AscendC::Trap();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_trap() {
  ascendc.trap
  return
}
