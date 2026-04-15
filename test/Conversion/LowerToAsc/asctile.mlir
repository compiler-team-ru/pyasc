// RUN: ascir-opt --asclower-asctile --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32) {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    %1 = ascendc.global_tensor : !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:    ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<32x32xf32>, memref<*xf32, 22>
// CHECK-NEXT:    %2 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:    %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:    %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:    %5 = arith.subi %c1024_i32, %3 : i32
// CHECK-NEXT:    %6 = arith.minsi %5, %c16_i32 : i32
// CHECK-NEXT:    %7 = arith.muli %6, %c4_i32 : i32
// CHECK-NEXT:    %8 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %7, %c0_i32, %c64_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_pad_l2_ext %4, %0, %8 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  asctile.store %arg1, %0 [%arg2, %arg3] : !asctile.tile<16x16xf32, UB>, !asctile.tensor<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    %1 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:    ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
// CHECK-NEXT:    %2 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:    %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:    %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:    %5 = arith.muli %arg4, %arg5 : i32
// CHECK-NEXT:    %6 = arith.subi %5, %3 : i32
// CHECK-NEXT:    %7 = arith.subi %arg5, %c16_i32 : i32
// CHECK-NEXT:    %8 = arith.muli %7, %c4_i32 : i32
// CHECK-NEXT:    %9 = arith.minsi %6, %c16_i32 : i32
// CHECK-NEXT:    %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:    %11 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %10, %c0_i32, %8, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_pad_l2_ext %4, %0, %11 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  asctile.store %arg1, %0 [%arg2, %arg3] : !asctile.tile<16x16xf32, UB>, !asctile.tensor<?x?xf32>
  return
}

// CHECK-LABEL: func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> !asctile.tile<16x16xf32, UB> {
// CHECK:         %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %0 = ascendc.global_tensor : !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:    ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<32x32xf32>, memref<*xf32, 22>
// CHECK-NEXT:    %1 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:    %2 = arith.addi %arg2, %1 : i32
// CHECK-NEXT:    %3 = ascendc.global_tensor.subindex %0[%2] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:    %4 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:    %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<16x16xf32> to !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:    %6 = arith.subi %c1024_i32, %2 : i32
// CHECK-NEXT:    %7 = arith.minsi %6, %c16_i32 : i32
// CHECK-NEXT:    %8 = arith.muli %7, %c4_i32 : i32
// CHECK-NEXT:    %9 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %8, %c64_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:    %10 = arith.addi %2, %c-896_i32 : i32
// CHECK-NEXT:    %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:    %12 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %11, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:    ascendc.data_copy_pad_l0_ext %4, %3, %9, %12 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:    return %5 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2], %cst : !asctile.tensor<32x32xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
}

// CHECK-LABEL: func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<16x16xf32, UB> {
// CHECK:         %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:    ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
// CHECK-NEXT:    %1 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:    %2 = arith.addi %arg4, %1 : i32
// CHECK-NEXT:    %3 = ascendc.global_tensor.subindex %0[%2] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:    %4 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:    %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<16x16xf32> to !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:    %6 = arith.muli %arg1, %arg2 : i32
// CHECK-NEXT:    %7 = arith.subi %6, %2 : i32
// CHECK-NEXT:    %8 = arith.subi %arg2, %c16_i32 : i32
// CHECK-NEXT:    %9 = arith.muli %8, %c4_i32 : i32
// CHECK-NEXT:    %10 = arith.minsi %7, %c16_i32 : i32
// CHECK-NEXT:    %11 = arith.muli %10, %c4_i32 : i32
// CHECK-NEXT:    %12 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %11, %9, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:    %13 = arith.subi %c128_i32, %7 : i32
// CHECK-NEXT:    %14 = arith.maxsi %13, %c0_i32 : i32
// CHECK-NEXT:    %15 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %14, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:    ascendc.data_copy_pad_l0_ext %4, %3, %12, %15 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:    return %5 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst : !asctile.tensor<?x?xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
}

// CHECK-LABEL: func.func @lower_splat(%arg0: f32) -> !asctile.tile<32xf32, UB> {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:    ascendc.duplicate_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
// CHECK-NEXT:    return %1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_splat(%arg0: f32) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.splat %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_relu(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32xf32, UB> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:    ascendc.relu_l2 %1, %0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:    return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_relu(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.relu %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_cast(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32xf32, UB> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to !asctile.tile<32xi32, UB>
// CHECK-NEXT:    ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 5 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:    return %2 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  }
func.func @lower_cast(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xf32, UB> to !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func @lower_matmul(%arg0: !asctile.tile<8x16xf32, UB>, %arg1: !asctile.tile<16x8xf32, UB>) -> !asctile.tile<8x8xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x8xf32, UB> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<8x16xf32, UB> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <8x8xf32>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x8xf32> to !asctile.tile<8x8xf32, UB>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT:    ascendc.mmad %2, %1, %0, %4 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:    return %3 : !asctile.tile<8x8xf32, UB>
// CHECK-NEXT:  }
func.func @lower_matmul(%arg0: !asctile.tile<8x16xf32, UB>, %arg1: !asctile.tile<16x8xf32, UB>) -> !asctile.tile<8x8xf32, UB> {
  %0 = asctile.matmul %arg0, %arg1 : !asctile.tile<8x16xf32, UB>, !asctile.tile<16x8xf32, UB> -> !asctile.tile<8x8xf32, UB>
  return %0 : !asctile.tile<8x8xf32, UB>
}

// CHECK-LABEL: func.func @lower_reshape(%arg0: !asctile.tile<16x16xf32, UB>) -> !asctile.tile<8x32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <8x32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<8x32xf32> to !asctile.tile<8x32xf32, UB>
// CHECK-NEXT:    ascendc.data_copy_l2 %1, %0, %c256_i64 : !ascendc.local_tensor<8x32xf32>, !ascendc.local_tensor<16x16xf32>, i64
// CHECK-NEXT:    return %2 : !asctile.tile<8x32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_reshape(%arg0: !asctile.tile<16x16xf32, UB>) -> !asctile.tile<8x32xf32, UB> {
  %0 = asctile.reshape %arg0 : !asctile.tile<16x16xf32, UB> to !asctile.tile<8x32xf32, UB>
  return %0 : !asctile.tile<8x32xf32, UB>
}

// CHECK-LABEL: func.func @lower_broadcast(%arg0: !asctile.tile<1xf32, UB>, %arg1: !asctile.tile<16x1xf32, UB>) -> (!asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>) {
// CHECK:        %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x1xf32, UB> to !ascendc.local_tensor<16x1xf32>
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<1xf32, UB> to !ascendc.local_tensor<1xf32>
// CHECK-NEXT:   %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:   %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:   %4 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:   ascendc.duplicate_l2 %2, %4, %c0_i64 : !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:   %5 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:   %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x32xf32> to !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:   ascendc.broadcast %5, %0, %c1_i32, %c16_i32, %c32_i32, %c1_i32, %c16_i32, %c1_i32 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x1xf32>, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:   return %3, %6 : !asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>
// CHECK-NEXT: }
func.func @lower_broadcast(%arg0: !asctile.tile<1xf32, UB>, %arg1: !asctile.tile<16x1xf32, UB>) -> (!asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>) {
  %0 = asctile.broadcast %arg0 : !asctile.tile<1xf32, UB> to !asctile.tile<16xf32, UB>
  %1 = asctile.broadcast %arg1 : !asctile.tile<16x1xf32, UB> to !asctile.tile<16x32xf32, UB>
  return %0, %1 : !asctile.tile<16xf32, UB>, !asctile.tile<16x32xf32, UB>
}

// CHECK-LABEL: func.func @lower_softmax(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %4 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:    %5 = ascendc.local_tensor_auto veccalc() : <128xui8>
// CHECK-NEXT:    %6 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:    %7 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c1_i32 : i32, "srcK" = %c16_i32 : i32, "oriSrcM" = %c1_i32 : i32, "oriSrcK" = %c16_i32 : i32)
// CHECK-NEXT:    ascendc.softmax %1, %3, %4, %0, %5, %6, %7 {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<128xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:    return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_softmax(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.softmax %arg0 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_softmax_2D(%arg0: !asctile.tile<16x32xf32, UB>) -> !asctile.tile<16x32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16x32xf32, UB> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <128xf32>
// CHECK-NEXT:    %4 = ascendc.local_tensor_auto veccalc() : <128xf32>
// CHECK-NEXT:    %5 = ascendc.local_tensor_auto veccalc() : <4096xui8>
// CHECK-NEXT:    %6 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:    %7 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c16_i32 : i32, "srcK" = %c32_i32 : i32, "oriSrcM" = %c16_i32 : i32, "oriSrcK" = %c32_i32 : i32)
// CHECK-NEXT:    ascendc.softmax %1, %3, %4, %0, %5, %6, %7 {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<128xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<4096xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:    return %2 : !asctile.tile<16x32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_softmax_2D(%arg0: !asctile.tile<16x32xf32, UB>) -> !asctile.tile<16x32xf32, UB> {
  %0 = asctile.softmax %arg0 : !asctile.tile<16x32xf32, UB>
  return %0 : !asctile.tile<16x32xf32, UB>
}

// CHECK-LABEL: func.func @lower_adds(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    ascendc.adds_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:    return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_adds(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.adds %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_muls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    ascendc.muls_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:    return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_muls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.muls %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_shls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    ascendc.shift_left_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:    return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_shls(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.shls %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_shrs(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    ascendc.shift_right_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:    return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  }
func.func @lower_shrs(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.shrs %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_sum_as_1d(%arg0: !asctile.tile<16x32x8xf32, UB>) -> f32 {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16x32x8xf32, UB> to !ascendc.local_tensor<16x32x8xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    ascendc.reduce_sum_l2 %1, %0, %2, %c4096_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16x32x8xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:    return %3 : f32
// CHECK-NEXT:  }
func.func @lower_reduce_sum_as_1d(%arg0: !asctile.tile<16x32x8xf32, UB>) -> f32 {
  %0 = asctile.reduce_sum_as_1d %arg0 : !asctile.tile<16x32x8xf32, UB>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_min_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <0xf32>
// CHECK-NEXT:    ascendc.reduce_min_l2 %1, %0, %2, %c16_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<0xf32>, i64, i64
// CHECK-NEXT:    %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:    return %3 : f32
// CHECK-NEXT:  }
func.func @lower_reduce_min_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
  %0 = asctile.reduce_min_as_1d %arg0 : !asctile.tile<16xf32, UB>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_max_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto veccalc() : <0xf32>
// CHECK-NEXT:    ascendc.reduce_max_l2 %1, %0, %2, %c16_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<0xf32>, i64, i64
// CHECK-NEXT:    %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:    return %3 : f32
// CHECK-NEXT:  }
func.func @lower_reduce_max_as_1d(%arg0: !asctile.tile<16xf32, UB>) -> f32 {
  %0 = asctile.reduce_max_as_1d %arg0 : !asctile.tile<16xf32, UB>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_sum(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:    ascendc.reduce_sum %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:    return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_reduce_sum(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce_sum %arg0 {dims = [0 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_min(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:    ascendc.reduce_min %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:    return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_reduce_min(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce_min %arg0 {dims = [0 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_max(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:    ascendc.reduce_max %1, %0, %3, %c64_i32, %c32_i32 {pattern = 1 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:    return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_reduce_max(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce_max %arg0 {dims = [1 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_reduce_prod(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<64x32xf32, UB> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:    %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:    ascendc.reduce_prod %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:    return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  }
func.func @lower_reduce_prod(%arg0: !asctile.tile<64x32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.reduce_prod %arg0 {dims = [0 : i32]} : !asctile.tile<64x32xf32, UB>, !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32)
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.global_tensor : !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<32x32xf32>, memref<*xf32, 22>
// CHECK-NEXT:  %2 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %5 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32)
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.fixpipe_config(%6) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %4, %0, %5, %7 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3], ,  : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32)
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.global_tensor : !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<32x32xf32>, memref<*xf32, 22>
// CHECK-NEXT:  %2 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %5 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32)
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.fixpipe_config(%6) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %4, %0, %5, %7 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3], unit, : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32)
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.global_tensor : !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<32x32xf16>, memref<*xf32, 22>
// CHECK-NEXT:  %2 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf16>, i32, !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %6 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %5 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.fixpipe_config(%7) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %4, %0, %6, %8 : !ascendc.global_tensor<32x32xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf16>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3], unit, unit : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<32x32xf16>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32)
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<?x?xf16>, memref<*xf32, 22>
// CHECK-NEXT:  %2 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %3 = arith.addi %arg3, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %6 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %5 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.fixpipe_config(%7) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %4, %0, %6, %8 : !ascendc.global_tensor<?x?xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf16>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3], unit, unit : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<?x?xf16>
  return
}
