// RUN: ascir-opt --asclower-asctile --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_splat(%arg0: f32) -> !asctile.tile<32xf32, UB> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  ascendc.duplicate_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
// CHECK-NEXT:  return %1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_splat(%arg0: f32) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.splat %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_relu(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32xf32, UB> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  ascendc.relu_l2 %1, %0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_relu(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.relu %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_cast(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32xf32, UB> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to !asctile.tile<32xi32, UB>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 5 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func @lower_cast(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xf32, UB> to !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func @lower_matmul(%arg0: !asctile.tile<8x16xf32, UB>, %arg1: !asctile.tile<16x8xf32, UB>) -> !asctile.tile<8x8xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x8xf32, UB> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<8x16xf32, UB> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8x8xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x8xf32> to !asctile.tile<8x8xf32, UB>
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %4 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  return %3 : !asctile.tile<8x8xf32, UB>
// CHECK-NEXT:}
func.func @lower_matmul(%arg0: !asctile.tile<8x16xf32, UB>, %arg1: !asctile.tile<16x8xf32, UB>) -> !asctile.tile<8x8xf32, UB> {
  %0 = asctile.matmul %arg0, %arg1 : !asctile.tile<8x16xf32, UB>, !asctile.tile<16x8xf32, UB> -> !asctile.tile<8x8xf32, UB>
  return %0 : !asctile.tile<8x8xf32, UB>
}

// CHECK-LABEL: func.func @lower_reshape(%arg0: !asctile.tile<16x16xf32, UB>) -> !asctile.tile<8x32xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<8x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<8x32xf32> to !asctile.tile<8x32xf32, UB>
// CHECK-NEXT:  return %2 : !asctile.tile<8x32xf32, UB>
// CHECK-NEXT:}
func.func @lower_reshape(%arg0: !asctile.tile<16x16xf32, UB>) -> !asctile.tile<8x32xf32, UB> {
  %0 = asctile.reshape %arg0 : !asctile.tile<16x16xf32, UB> to !asctile.tile<8x32xf32, UB>
  return %0 : !asctile.tile<8x32xf32, UB>
}

// CHECK-LABEL: func.func @lower_broadcast(%arg0: !asctile.tile<1xf32, UB>, %arg1: !asctile.tile<16x1xf32, UB>) -> (!asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x1xf32, UB> to !ascendc.local_tensor<16x1xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<1xf32, UB> to !ascendc.local_tensor<1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %4 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  ascendc.duplicate_l2 %2, %4, %c0_i64 : !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x32xf32> to !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:  ascendc.broadcast %5, %0, %c1_i32, %c16_i32, %c32_i32, %c1_i32, %c16_i32, %c1_i32 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x1xf32>, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  return %3, %6 : !asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:}
func.func @lower_broadcast(%arg0: !asctile.tile<1xf32, UB>, %arg1: !asctile.tile<16x1xf32, UB>) -> (!asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>) {
  %0 = asctile.broadcast %arg0 : !asctile.tile<1xf32, UB> to !asctile.tile<16xf32, UB>
  %1 = asctile.broadcast %arg1 : !asctile.tile<16x1xf32, UB> to !asctile.tile<16x32xf32, UB>
  return %0, %1 : !asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>
}

// CHECK-LABEL: func.func @lower_softmax(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <128xui8>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:  %7 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c1_i32 : i32, "srcK" = %c16_i32 : i32, "oriSrcM" = %c1_i32 : i32, "oriSrcK" = %c16_i32 : i32)
// CHECK-NEXT:  ascendc.softmax %1, %3, %4, %0, %5, %6, %7 {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<128xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_softmax(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.softmax %arg0 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_softmax_2D(%arg0: !asctile.tile<16x32xf32, UB>) -> !asctile.tile<16x32xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16x32xf32, UB> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <128xf32>
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <128xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <4096xui8>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:  %7 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c16_i32 : i32, "srcK" = %c32_i32 : i32, "oriSrcM" = %c16_i32 : i32, "oriSrcK" = %c32_i32 : i32)
// CHECK-NEXT:  ascendc.softmax %1, %3, %4, %0, %5, %6, %7 {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<4096xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:  return %2 : !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:}
func.func @lower_softmax_2D(%arg0: !asctile.tile<16x32xf32, UB>) -> !asctile.tile<16x32xf32, UB> {
  %0 = asctile.softmax %arg0 : !asctile.tile<16x32xf32, UB>
  return %0 : !asctile.tile<16x32xf32, UB>
}

// CHECK-LABEL: func.func @lower_adds(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  ascendc.adds_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_adds(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.adds %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_muls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  ascendc.muls_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_muls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.muls %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_shls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  ascendc.shift_left_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_shls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.shls %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_shrs(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  ascendc.shift_right_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_shrs(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.shrs %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_sum_as_1d(%arg0: !asctile.tile<16x32x8xf32, UB>) -> f32 {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16x32x8xf32, UB> to !ascendc.local_tensor<16x32x8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  ascendc.reduce_sum_l2 %1, %0, %2, %c4096_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16x32x8xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  return %3 : f32
// CHECK-NEXT:}
func.func @lower_reduce_sum_as_1d(%arg0: !asctile.tile<16x32x8xf32, UB>) -> f32 {
  %0 = asctile.reduce_as_1d <sum> %arg0 : !asctile.tile<16x32x8xf32, UB>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_min_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <0xf32>
// CHECK-NEXT:  ascendc.reduce_min_l2 %1, %0, %2, %c16_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<0xf32>, i64, i64
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  return %3 : f32
// CHECK-NEXT:}
func.func @lower_reduce_min_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
  %0 = asctile.reduce_as_1d <min> %arg0 : !asctile.tile<16xf32, UB>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_max_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <0xf32>
// CHECK-NEXT:  ascendc.reduce_max_l2 %1, %0, %2, %c16_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<0xf32>, i64, i64
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  return %3 : f32
// CHECK-NEXT:}
func.func @lower_reduce_max_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
  %0 = asctile.reduce_as_1d <max> %arg0 : !asctile.tile<16xf32, UB>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_sum(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_sum %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_reduce_sum(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce <sum> %arg0 {dims = [0 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_min(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_min %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_reduce_min(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce <min> %arg0 {dims = [0 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_max(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_max %1, %0, %3, %c64_i32, %c32_i32 {pattern = 1 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_reduce_max(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce <max> %arg0 {dims = [1 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_prod(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_prod %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_reduce_prod(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce <prod> %arg0 {dims = [0 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}
