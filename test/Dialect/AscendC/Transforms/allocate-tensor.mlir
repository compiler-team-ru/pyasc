// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-allocate-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @test_single_veccalc() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_single_veccalc() {
  %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
  return
}

// CHECK-LABEL: func.func @test_multiple_veccalc() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 veccalc, 256, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_multiple_veccalc() {
  %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
  return
}

// CHECK-LABEL: func.func @test_a1_tensor() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a1_tensor() {
  %0 = ascendc.local_tensor_auto a1() : <64xf32>
  return
}

// CHECK-LABEL: func.func @test_a2_tensor() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a2, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a2_tensor() {
  %0 = ascendc.local_tensor_auto a2() : <64xf32>
  return
}

// CHECK-LABEL: func.func @test_vecin_normalized() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_vecin_normalized() {
  %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
  return
}

// CHECK-LABEL: func.func private @test_declaration(i32) -> i32
func.func private @test_declaration(%arg0: i32) -> i32

// CHECK-LABEL: func.func @test_different_types() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 16 : !ascendc.local_tensor<8xf16>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 veccalc, 32, 8 : !ascendc.local_tensor<8xi32>
// CHECK-NEXT: %2 = ascendc.local_tensor_v3 veccalc, 64, 8 : !ascendc.local_tensor<8xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_different_types() {
  %0 = ascendc.local_tensor_auto veccalc() : <8xf16>
  %1 = ascendc.local_tensor_auto veccalc() : <8xi32>
  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
  return
}

// CHECK-LABEL: func.func @test_cube_alignment_a2() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a2, 0, 256 : !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_cube_alignment_a2() {
  %0 = ascendc.local_tensor_auto a2() : <3x16xf32>
  return
}

// CHECK-LABEL: func.func @test_cube_alignment_b2() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 b2, 0, 128 : !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_cube_alignment_b2() {
  %0 = ascendc.local_tensor_auto b2() : <3x16xf32>
  return
}

// CHECK-LABEL: func.func @test_address_accumulation() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 32 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 veccalc, 128, 32 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: %2 = ascendc.local_tensor_v3 veccalc, 256, 32 : !ascendc.local_tensor<32xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_address_accumulation() {
  %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
  %2 = ascendc.local_tensor_auto veccalc() : <32xf32>
  return
}

// CHECK-LABEL: func.func @test_input_output_attrs() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 veccalc, 256, 64 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_input_output_attrs() {
  %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
  %1 = ascendc.local_tensor_auto vecout() output : <64xf32>
  return
}

// CHECK-LABEL: func.func @test_ub_alignment() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 8 : !ascendc.local_tensor<7xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 veccalc, 32, 8 : !ascendc.local_tensor<5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_ub_alignment() {
  %0 = ascendc.local_tensor_auto veccalc() : <7xf32>
  %1 = ascendc.local_tensor_auto veccalc() : <5xf32>
  return
}

// CHECK-LABEL: func.func @test_multiple_a1() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<1x16xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 a1, 1024, 256 : !ascendc.local_tensor<1x16xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_multiple_a1() {
  %0 = ascendc.local_tensor_auto a1() : <1x16xf32>
  %1 = ascendc.local_tensor_auto a1() : <1x16xf32>
  return
}

// CHECK-LABEL: func.func @test_veccalc_2d() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 32 : !ascendc.local_tensor<4x8xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_veccalc_2d() {
  %0 = ascendc.local_tensor_auto veccalc() : <4x8xf32>
  return
}

// CHECK-LABEL: func.func @test_a1_2d() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a1_2d() {
  %0 = ascendc.local_tensor_auto a1() : <3x16xf32>
  return
}

// CHECK-LABEL: func.func @test_b1_normalized() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_b1_normalized() {
  %0 = ascendc.local_tensor_auto b1() : <3x16xf32>
  return
}

// CHECK-LABEL: func.func @test_a1_unaligned() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<3x5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a1_unaligned() {
  %0 = ascendc.local_tensor_auto a1() : <3x5xf32>
  return
}

// CHECK-LABEL: func.func @test_a2_unaligned() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a2, 0, 128 : !ascendc.local_tensor<3x5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a2_unaligned() {
  %0 = ascendc.local_tensor_auto a2() : <3x5xf32>
  return
}

// CHECK-LABEL: func.func @test_b2_unaligned() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 b2, 0, 128 : !ascendc.local_tensor<3x5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_b2_unaligned() {
  %0 = ascendc.local_tensor_auto b2() : <3x5xf32>
  return
}

// CHECK-LABEL: func.func @test_co1_unaligned() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 co1, 0, 16 : !ascendc.local_tensor<3x5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_co1_unaligned() {
  %0 = ascendc.local_tensor_auto co1() : <3x5xf32>
  return
}

// CHECK-LABEL: func.func @test_c2_unaligned() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 c2, 0, 16 : !ascendc.local_tensor<3x5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_c2_unaligned() {
  %0 = ascendc.local_tensor_auto c2() : <3x5xf32>
  return
}

// CHECK-LABEL: func.func @test_a1_unaligned_f16() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 256 : !ascendc.local_tensor<3x5xf16>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a1_unaligned_f16() {
  %0 = ascendc.local_tensor_auto a1() : <3x5xf16>
  return
}

// CHECK-LABEL: func.func @test_a2_unaligned_f16() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a2, 0, 256 : !ascendc.local_tensor<3x5xf16>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_a2_unaligned_f16() {
  %0 = ascendc.local_tensor_auto a2() : <3x5xf16>
  return
}

// CHECK-LABEL: func.func @test_b2_unaligned_f16() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 b2, 0, 256 : !ascendc.local_tensor<3x5xf16>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_b2_unaligned_f16() {
  %0 = ascendc.local_tensor_auto b2() : <3x5xf16>
  return
}

// CHECK-LABEL: func.func @test_multiple_unaligned_a1() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 a1, 0, 128 : !ascendc.local_tensor<2x3xf32>
// CHECK-NEXT: %1 = ascendc.local_tensor_v3 a1, 512, 128 : !ascendc.local_tensor<5x7xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_multiple_unaligned_a1() {
  %0 = ascendc.local_tensor_auto a1() : <2x3xf32>
  %1 = ascendc.local_tensor_auto a1() : <5x7xf32>
  return
}

// CHECK-LABEL: func.func @test_veccalc_unaligned_2d() {
// CHECK-NEXT: %0 = ascendc.local_tensor_v3 veccalc, 0, 16 : !ascendc.local_tensor<3x5xf32>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_veccalc_unaligned_2d() {
  %0 = ascendc.local_tensor_auto veccalc() : <3x5xf32>
  return
}
