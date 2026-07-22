// RUN: ascir-opt %s --ascendc-detect-kernel-type --split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {asc.kernel_type = "vector"} {
// CHECK-NEXT: func.func @test_vector(%arg0: !ascendc.local_tensor<*xf32>, %arg1: i32) {
// CHECK-NEXT: ascendc.add_l2 %arg0, %arg0, %arg0, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_vector(%arg0: !ascendc.local_tensor<*xf32>, %arg1: i32) {
    ascendc.add_l2 %arg0, %arg0, %arg0, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "cube"} {
// CHECK-NEXT: func.func @test_cube(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params) {
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_cube(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params) {
    ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
    return
  }
}

// -----

// CHECK-LABEL: module attributes {asc.kernel_type = "mixed"} {
// CHECK-NEXT: func.func @test_mixed(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params, %arg4: i32) {
// CHECK-NEXT: ascendc.add_l2 %arg0, %arg0, %arg0, %arg4 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT: ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
module {
  func.func @test_mixed(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>, %arg2: !ascendc.local_tensor<*xf16>, %arg3: !ascendc.mmad_params, %arg4: i32) {
    ascendc.add_l2 %arg0, %arg0, %arg0, %arg4 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.mmad %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.mmad_params
    return
  }
}
