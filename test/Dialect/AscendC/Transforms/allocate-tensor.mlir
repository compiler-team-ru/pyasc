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
