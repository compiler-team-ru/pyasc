// RUN: ascir-opt -ascendc-fill-asc-operands -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @test_binary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i32) {
// CHECK:       ascendc.add_l2 %arg0, %arg1, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_binary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i32) {
  ascendc.add_l2 %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
  return
}

// CHECK-LABEL: func.func @test_unary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i32) {
// CHECK:       ascendc.abs_l2 %arg0, %arg1, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_unary_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i32) {
  ascendc.abs_l2 %arg0, %arg1, %arg2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
  return
}

// CHECK-LABEL: func.func @test_duplicate_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
// CHECK:       ascendc.duplicate_l2 %arg0, %arg1, %c64_i64 : !ascendc.local_tensor<64xf32>, f32, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_duplicate_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
  ascendc.duplicate_l2 %arg0, %arg1, %arg2 : !ascendc.local_tensor<64xf32>, f32, i32
  return
}

// CHECK-LABEL: func.func @test_vecscalar_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i32) {
// CHECK:       ascendc.adds_l2 %arg0, %arg1, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_vecscalar_l2(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i32) {
  ascendc.adds_l2 %arg0, %arg1, %arg2, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i32
  return
}

// CHECK-LABEL: func.func @test_l2_cal_count_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
// CHECK-NEXT:  ascendc.duplicate_l2 %arg0, %arg1, %arg2 {asc.cal_count_set} : !ascendc.local_tensor<64xf32>, f32, i32
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_l2_cal_count_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i32) {
  ascendc.duplicate_l2 %arg0, %arg1, %arg2 {asc.cal_count_set} : !ascendc.local_tensor<64xf32>, f32, i32
  return
}

// CHECK-LABEL: func.func @test_unary_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.abs_l0 %arg0, %arg1, %0, %c1_i64, %1 {isSetMask} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !emitasc.mask, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_unary_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
  ascendc.abs_l0 %arg0, %arg1, %arg2, %arg3, %arg4 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_binary_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.binary_repeat_params(%c1_i64, %c1_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64, i64, i64
// CHECK-NEXT:  ascendc.add_l0 %arg0, %arg1, %arg2, %0, %c1_i64, %1 {isSetMask} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !emitasc.mask, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_binary_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
  ascendc.add_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_cast_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xi32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.cast_l0 %arg0, %arg1, %0, %c1_i64, %1 {isSetMask, roundMode = 0 : i32} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xi32>, !emitasc.mask, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_cast_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xi32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
  ascendc.cast_l0 %arg0, %arg1, %arg2, %arg3, %arg4 {roundMode = 0 : i32} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xi32>, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_compare_scalar_l0(%arg0: !ascendc.local_tensor<2xui8>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.compare_scalar_l0 %arg0, %arg1, %arg2, %0, %c1_i64, %1 {cmpMode = 0 : i64, isSetMask} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<64xf32>, f32, !emitasc.mask, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_compare_scalar_l0(%arg0: !ascendc.local_tensor<2xui8>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
  ascendc.compare_scalar_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {cmpMode = 0 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<64xf32>, f32, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_compare_l0(%arg0: !ascendc.local_tensor<2xui8>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.binary_repeat_params(%c1_i64, %c1_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64, i64, i64
// CHECK-NEXT:  ascendc.compare_l0 %arg0, %arg1, %arg2, %0, %c1_i64, %1 {cmpMode = 1 : i64, isSetMask} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !emitasc.mask, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_compare_l0(%arg0: !ascendc.local_tensor<2xui8>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: i64, %arg4: i64, %arg5: !ascendc.binary_repeat_params) {
  ascendc.compare_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {cmpMode = 1 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_select_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<4xui8>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: !ascendc.local_tensor<64xf32>, %arg4: i64, %arg5: i64, %arg6: !ascendc.binary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.binary_repeat_params(%c1_i64, %c1_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64, i64, i64
// CHECK-NEXT:  ascendc.select_l0 %arg0, %arg1, %arg2, %arg3, %0, %c1_i64, %1 {isSetMask, selMode = 2 : i32} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4xui8>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !emitasc.mask, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_select_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<4xui8>, %arg2: !ascendc.local_tensor<64xf32>, %arg3: !ascendc.local_tensor<64xf32>, %arg4: i64, %arg5: i64, %arg6: !ascendc.binary_repeat_params) {
  ascendc.select_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {selMode = 2 : i32} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<4xui8>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_duplicate_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  ascendc.duplicate_l0 %arg0, %arg1, %0, %c1_i64, %arg4, %arg5 {isSetMask} : !ascendc.local_tensor<64xf32>, f32, !emitasc.mask, i64, i64, i64
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_duplicate_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: f32, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
  ascendc.duplicate_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !ascendc.local_tensor<64xf32>, f32, i64, i64, i64, i64
  return
}

// CHECK-LABEL: func.func @test_vecscalar_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
// CHECK:       %0 = emitasc.mask %c0_i64, %c-1_i64
// CHECK-NEXT:  %1 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.adds_l0 %arg0, %arg1, %arg2, %0, %c1_i64, %1 {isSetMask} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, !emitasc.mask, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_vecscalar_l0(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: f32, %arg3: i64, %arg4: i64, %arg5: !ascendc.unary_repeat_params) {
  ascendc.adds_l0 %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64, i64, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL: func.func @test_l0_mask_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
// CHECK:       %0 = ascendc.construct !ascendc.unary_repeat_params(%c1_i64, %c1_i64, %c8_i64, %c8_i64) : i64, i64, i64, i64
// CHECK-NEXT:  ascendc.abs_l0 %arg0, %arg1, %arg2, %c1_i64, %0 {asc.mask_set, isSetMask} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @test_l0_mask_set(%arg0: !ascendc.local_tensor<64xf32>, %arg1: !ascendc.local_tensor<64xf32>, %arg2: i64, %arg3: i64, %arg4: !ascendc.unary_repeat_params) {
  ascendc.abs_l0 %arg0, %arg1, %arg2, %arg3, %arg4 {asc.mask_set} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64, i64, !ascendc.unary_repeat_params
  return
}
