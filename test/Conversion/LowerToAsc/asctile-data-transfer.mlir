// RUN: ascir-opt -asclower-asctile-data-transfer -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> !asctile.tile<16x16xf32, UB> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !asctile.tensor<32x32xf32> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %2 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:  %7 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %8 = arith.cmpi slt, %7, %c0_i32 : i32
// CHECK-NEXT:  %9 = arith.select %8, %c0_i32, %7 : i32
// CHECK-NEXT:  %10 = arith.minsi %9, %c16_i32 : i32
// CHECK-NEXT:  %11 = arith.muli %10, %c4_i32 : i32
// CHECK-NEXT:  %12 = arith.subi %c32_i32, %10 : i32
// CHECK-NEXT:  %13 = arith.subi %c16_i32, %10 : i32
// CHECK-NEXT:  %14 = arith.cmpi sgt, %13, %c8_i32 : i32
// CHECK-NEXT:  %15 = arith.select %14, %c8_i32, %13 : i32
// CHECK-NEXT:  scf.if %14 {
// CHECK-NEXT:    ascendc.duplicate_l2 %5, %cst, %c1024_i32 : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %16 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %17 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %11, %16, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %18 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %15, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %17, %18 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %6 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:}
func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2], %cst : !asctile.tensor<32x32xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
}

// CHECK-LABEL: func.func @lower_load_4d_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<2x2x3x4xf32, UB> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<3x4x5x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !asctile.tensor<3x4x5x8xf32> to !ascendc.global_tensor<3x4x5x8xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %c8_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = arith.muli %arg2, %c40_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %3, %4 : i32
// CHECK-NEXT:  %6 = arith.muli %arg1, %c160_i32 : i32
// CHECK-NEXT:  %7 = arith.addi %5, %6 : i32
// CHECK-NEXT:  %8 = ascendc.global_tensor.subindex %1[%7] : !ascendc.global_tensor<3x4x5x8xf32>, i32, !ascendc.global_tensor<3x4x5x8xf32>
// CHECK-NEXT:  %9 = ascendc.local_tensor_auto veccalc() : <2x2x3x4xf32>
// CHECK-NEXT:  %10 = builtin.unrealized_conversion_cast %9 : !ascendc.local_tensor<2x2x3x4xf32> to !asctile.tile<2x2x3x4xf32, UB>
// CHECK-NEXT:  %11 = arith.subi %c8_i32, %arg4 : i32
// CHECK-NEXT:  %12 = arith.cmpi slt, %11, %c0_i32 : i32
// CHECK-NEXT:  %13 = arith.select %12, %c0_i32, %11 : i32
// CHECK-NEXT:  %14 = arith.minsi %13, %c4_i32 : i32
// CHECK-NEXT:  %15 = arith.muli %14, %c4_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c8_i32, %14 : i32
// CHECK-NEXT:  %17 = arith.subi %c4_i32, %14 : i32
// CHECK-NEXT:  %18 = arith.cmpi sgt, %17, %c8_i32 : i32
// CHECK-NEXT:  %19 = arith.select %18, %c8_i32, %17 : i32
// CHECK-NEXT:  scf.if %18 {
// CHECK-NEXT:    ascendc.duplicate_l2 %9, %cst, %c480_i32 : !ascendc.local_tensor<2x2x3x4xf32>, f32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %20 = arith.muli %16, %c4_i32 : i32
// CHECK-NEXT:  %21 = ascendc.construct !ascendc.data_copy_ext_params(%c12_i32, %15, %20, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %22 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %19, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %9, %8, %21, %22 : !ascendc.local_tensor<2x2x3x4xf32>, !ascendc.global_tensor<3x4x5x8xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %10 : !asctile.tile<2x2x3x4xf32, UB>
// CHECK-NEXT:}
func.func @lower_load_4d_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<2x2x3x4xf32, UB> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<3x4x5x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2, %arg3, %arg4], %cst : !asctile.tensor<3x4x5x8xf32>, !asctile.tile<2x2x3x4xf32, UB>
  return %1: !asctile.tile<2x2x3x4xf32, UB>
}

// CHECK-LABEL: func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<16x16xf32, UB> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !asctile.tensor<?x?xf32> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:  %7 = arith.muli %arg1, %arg2 : i32
// CHECK-NEXT:  %8 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %9 = arith.cmpi slt, %8, %c0_i32 : i32
// CHECK-NEXT:  %10 = arith.select %9, %c0_i32, %8 : i32
// CHECK-NEXT:  %11 = arith.minsi %10, %c16_i32 : i32
// CHECK-NEXT:  %12 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %13 = arith.subi %arg2, %11 : i32
// CHECK-NEXT:  %14 = arith.subi %c16_i32, %11 : i32
// CHECK-NEXT:  %15 = arith.cmpi sgt, %14, %c8_i32 : i32
// CHECK-NEXT:  %16 = arith.select %15, %c8_i32, %14 : i32
// CHECK-NEXT:  scf.if %15 {
// CHECK-NEXT:    ascendc.duplicate_l2 %5, %cst, %7 : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:  }
// CHECK-NEXT:  %17 = arith.muli %13, %c4_i32 : i32
// CHECK-NEXT:  %18 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %12, %17, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %19 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %16, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %18, %19 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %6 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:}
func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst : !asctile.tensor<?x?xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
}

// CHECK-LABEL: func.func @lower_load_real_shape(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> !asctile.tile<16x16xf32, UB> {
// CHECK:        %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %0 : !asctile.tensor<?x?xf32> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:   %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:   %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:   %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:   %7 = arith.muli %arg1, %arg2 : i32
// CHECK-NEXT:   %8 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:   %9 = arith.cmpi slt, %8, %c0_i32 : i32
// CHECK-NEXT:   %10 = arith.select %9, %c0_i32, %8 : i32
// CHECK-NEXT:   %11 = arith.minsi %arg6, %10 : i32
// CHECK-NEXT:   %12 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:   %13 = arith.subi %arg2, %11 : i32
// CHECK-NEXT:   %14 = arith.subi %c16_i32, %arg6 : i32
// CHECK-NEXT:   %15 = arith.cmpi sgt, %14, %c8_i32 : i32
// CHECK-NEXT:   %16 = arith.select %15, %c8_i32, %14 : i32
// CHECK-NEXT:   scf.if %15 {
// CHECK-NEXT:     ascendc.duplicate_l2 %5, %cst, %7 : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %17 = arith.muli %13, %c4_i32 : i32
// CHECK-NEXT:   %18 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %12, %17, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:   %19 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %16, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:   ascendc.data_copy_pad_l0_ext %5, %4, %18, %19 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:   return %6 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT: }
func.func @lower_load_real_shape(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst, (%arg5, %arg6) : !asctile.tensor<?x?xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
}

// CHECK-LABEL: func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<32x32xf32> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %c16_i32, %9 : i32
// CHECK-NEXT:  %12 = arith.subi %c32_i32, %9 : i32
// CHECK-NEXT:  %13 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %14 = arith.divsi %13, %c32_i32 : i32
// CHECK-NEXT:  %15 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %10, %14, %15, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %5, %0, %16 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  asctile.store %arg1, %0 [%arg2, %arg3] : !asctile.tile<16x16xf32, UB>, !asctile.tensor<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<?x?xf32> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %c16_i32, %9 : i32
// CHECK-NEXT:  %12 = arith.subi %arg5, %9 : i32
// CHECK-NEXT:  %13 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %14 = arith.divsi %13, %c32_i32 : i32
// CHECK-NEXT:  %15 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %10, %14, %15, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %5, %0, %16 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  asctile.store %arg1, %0 [%arg2, %arg3] : !asctile.tile<16x16xf32, UB>, !asctile.tensor<?x?xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<32x32xf32> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32)
// CHECK-NEXT:  %11 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %12 = ascendc.construct !ascendc.fixpipe_config(%11) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %10, %12 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<32x32xf32> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32)
// CHECK-NEXT:  %11 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %12 = ascendc.construct !ascendc.fixpipe_config(%11) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %10, %12 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {relu} : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<32x32xf32>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf16>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<32x32xf16> to !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf16>, i32, !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %11 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %10 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %12 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.fixpipe_config(%12) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %11, %13 : !ascendc.global_tensor<32x32xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf16>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<32x32xf16>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, L0C> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf16>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<?x?xf16> to !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %11 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %10 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %12 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.fixpipe_config(%12) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %11, %13 : !ascendc.global_tensor<?x?xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf16>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<?x?xf16>
  return
}

// CHECK-LABEL: func.func @lower_store_real_shape(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<2x8xf32, UB>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:   %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<2x8xf32, UB> to !ascendc.local_tensor<2x8xf32>
// CHECK-NEXT:   %1 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<?x?xf32> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %3 = ascendc.global_tensor.subindex %2[%c0_i32] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %4 = arith.cmpi slt, %arg3, %c0_i32 : i32
// CHECK-NEXT:   %5 = arith.select %4, %c0_i32, %arg3 : i32
// CHECK-NEXT:   %6 = arith.minsi %arg5, %5 : i32
// CHECK-NEXT:   %7 = arith.muli %6, %c4_i32 : i32
// CHECK-NEXT:   %8 = arith.subi %c8_i32, %6 : i32
// CHECK-NEXT:   %9 = arith.subi %arg3, %6 : i32
// CHECK-NEXT:   %10 = arith.muli %8, %c4_i32 : i32
// CHECK-NEXT:   %11 = arith.divsi %10, %c32_i32 : i32
// CHECK-NEXT:   %12 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:   %13 = ascendc.construct !ascendc.data_copy_ext_params(%arg4, %7, %11, %12, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:   ascendc.data_copy_pad_l2_ext %3, %0, %13 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<2x8xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_store_real_shape(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<2x8xf32, UB>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  asctile.store %arg1, %0 [%c0_i32, %c0_i32], (%arg4, %arg5) : !asctile.tile<2x8xf32, UB>, !asctile.tensor<?x?xf32>
  return
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_rows(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<32x16xf16, L0A>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32x32xf16, L1> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto a2() : <32x16xf16>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32x16xf16> to !asctile.tile<32x16xf16, L0A>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c2_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %1[%c0_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %6 = ascendc.local_tensor.subindex %2[%c0_i32] : !ascendc.local_tensor<32x16xf16>, i32, !ascendc.local_tensor<32x16xf16>
// CHECK-NEXT:    ascendc.load_data_g2l %6, %5, %4 : !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:    return %3 : !asctile.tile<32x16xf16, L0A>
func.func @lower_copy_l1_l0a_multiple_rows(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<32x16xf16, L0A> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : !asctile.tile<32x32xf16, L1>, !asctile.tile<32x16xf16, L0A>
  return %0: !asctile.tile<32x16xf16, L0A>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_cols(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<16x32xf16, L0A>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32x32xf16, L1> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor.subindex %0[%c256_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto a2() : <16x32xf16>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x32xf16> to !asctile.tile<16x32xf16, L0A>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c1_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      %5 = arith.muli %arg1, %c512_i32 : i32
// CHECK-NEXT:      %6 = ascendc.local_tensor.subindex %1[%5] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:      %7 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:      %8 = ascendc.local_tensor.subindex %2[%7] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:      ascendc.load_data_g2l %8, %6, %4 : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:    } {asctile.parallel}
// CHECK-NEXT:    return %3 : !asctile.tile<16x32xf16, L0A>
func.func @lower_copy_l1_l0a_multiple_cols(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<16x32xf16, L0A> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : !asctile.tile<32x32xf16, L1>, !asctile.tile<16x32xf16, L0A>
  return %0: !asctile.tile<16x32xf16, L0A>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_rows_f32(%arg0: !asctile.tile<32x32xf32, L1>) -> !asctile.tile<32x16xf32, L0A>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32x32xf32, L1> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto a2() : <32x16xf32>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32x16xf32> to !asctile.tile<32x16xf32, L0A>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c2_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      %5 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:      %6 = ascendc.local_tensor.subindex %1[%5] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:      %7 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:      %8 = ascendc.local_tensor.subindex %2[%7] : !ascendc.local_tensor<32x16xf32>, i32, !ascendc.local_tensor<32x16xf32>
// CHECK-NEXT:      ascendc.load_data_g2l %8, %6, %4 : !ascendc.local_tensor<32x16xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params
// CHECK-NEXT:    } {asctile.parallel}
// CHECK-NEXT:    return %3 : !asctile.tile<32x16xf32, L0A>
func.func @lower_copy_l1_l0a_multiple_rows_f32(%arg0: !asctile.tile<32x32xf32, L1>) -> !asctile.tile<32x16xf32, L0A> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : !asctile.tile<32x32xf32, L1>, !asctile.tile<32x16xf32, L0A>
  return %0: !asctile.tile<32x16xf32, L0A>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_cols_f32(%arg0: !asctile.tile<32x32xf32, L1>) -> !asctile.tile<16x32xf32, L0A>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32x32xf32, L1> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor.subindex %0[%c128_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto a2() : <16x32xf32>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x32xf32> to !asctile.tile<16x32xf32, L0A>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c1_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      %5 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:      %6 = ascendc.local_tensor.subindex %1[%5] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:      %7 = arith.muli %arg1, %c128_i32 : i32
// CHECK-NEXT:      %8 = ascendc.local_tensor.subindex %2[%7] : !ascendc.local_tensor<16x32xf32>, i32, !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:      ascendc.load_data_g2l %8, %6, %4 : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params
// CHECK-NEXT:    } {asctile.parallel}
// CHECK-NEXT:    return %3 : !asctile.tile<16x32xf32, L0A>
func.func @lower_copy_l1_l0a_multiple_cols_f32(%arg0: !asctile.tile<32x32xf32, L1>) -> !asctile.tile<16x32xf32, L0A> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : !asctile.tile<32x32xf32, L1>, !asctile.tile<16x32xf32, L0A>
  return %0: !asctile.tile<16x32xf32, L0A>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_multiple_rows(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<32x16xf16, L0B>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32x32xf16, L1> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto b2() : <32x16xf16>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32x16xf16> to !asctile.tile<32x16xf16, L0B>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c2_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:    %5 = ascendc.local_tensor.subindex %1[%c0_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %6 = ascendc.local_tensor.subindex %2[%c0_i32] : !ascendc.local_tensor<32x16xf16>, i32, !ascendc.local_tensor<32x16xf16>
// CHECK-NEXT:    ascendc.load_data_g2l %6, %5, %4 : !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:    return %3 : !asctile.tile<32x16xf16, L0B>
func.func @lower_copy_l1_l0b_multiple_rows(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<32x16xf16, L0B> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : !asctile.tile<32x32xf16, L1>, !asctile.tile<32x16xf16, L0B>
  return %0: !asctile.tile<32x16xf16, L0B>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_multiple_cols(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<16x32xf16, L0B>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32x32xf16, L1> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor.subindex %0[%c256_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto b2() : <16x32xf16>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x32xf16> to !asctile.tile<16x32xf16, L0B>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c1_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c1_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
// CHECK-NEXT:      %5 = arith.muli %arg1, %c512_i32 : i32
// CHECK-NEXT:      %6 = ascendc.local_tensor.subindex %1[%5] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:      %7 = arith.muli %arg1, %c256_i32 : i32
// CHECK-NEXT:      %8 = ascendc.local_tensor.subindex %2[%7] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x32xf16>
// CHECK-NEXT:      ascendc.load_data_g2l %8, %6, %4 : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params
// CHECK-NEXT:    } {asctile.parallel}
// CHECK-NEXT:    return %3 : !asctile.tile<16x32xf16, L0B>
func.func @lower_copy_l1_l0b_multiple_cols(%arg0: !asctile.tile<32x32xf16, L1>) -> !asctile.tile<16x32xf16, L0B> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : !asctile.tile<32x32xf16, L1>, !asctile.tile<16x32xf16, L0B>
  return %0: !asctile.tile<16x32xf16, L0B>
}
