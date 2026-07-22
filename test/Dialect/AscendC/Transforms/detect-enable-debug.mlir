// RUN: ascir-opt %s --ascendc-detect-enable-debug --split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {asc.enable_debug} {
// CHECK-NEXT: func.func @test_printf(%arg0: i32) {
// CHECK-NEXT: ascendc.printf %arg0 {desc = "test print %d"} : i32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_printf(%arg0: i32) {
    ascendc.printf %arg0 {desc = "test print %d"} : i32
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.enable_debug} {
// CHECK-NEXT: func.func @test_dump_tensor(%arg0: !ascendc.global_tensor<*xf32>, %arg1: ui32, %arg2: ui32) {
// CHECK-NEXT: ascendc.dump_tensor %arg0, %arg1, %arg2 : !ascendc.global_tensor<*xf32>, ui32, ui32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_dump_tensor(%arg0: !ascendc.global_tensor<*xf32>, %arg1: ui32, %arg2: ui32) {
    ascendc.dump_tensor %arg0, %arg1, %arg2: !ascendc.global_tensor<*xf32>, ui32, ui32
    return
  }
}

// -----

// CHECK-LABEL: module {
// CHECK-NEXT: func.func @test_no_debug(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_no_debug(%arg0: !ascendc.local_tensor<*xf32>) {
    return
  }
}
