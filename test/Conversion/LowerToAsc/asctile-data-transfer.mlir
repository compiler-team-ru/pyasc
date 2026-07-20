// RUN: ascir-opt -asclower-asctile-data-transfer -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %2 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %7 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %8 = arith.cmpi slt, %7, %c0_i32 : i32
// CHECK-NEXT:  %9 = arith.select %8, %c0_i32, %7 : i32
// CHECK-NEXT:  %10 = arith.minsi %9, %c16_i32 : i32
// CHECK-NEXT:  %11 = arith.muli %10, %c4_i32 : i32
// CHECK-NEXT:  %12 = arith.subi %c32_i32, %10 : i32
// CHECK-NEXT:  %13 = arith.remsi %11, %c32_i32 : i32
// CHECK-NEXT:  %14 = arith.cmpi eq, %13, %c0_i32 : i32
// CHECK-NEXT:  %15 = arith.subi %c32_i32, %13 : i32
// CHECK-NEXT:  %16 = arith.select %14, %c0_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.addi %11, %16 : i32
// CHECK-NEXT:  %18 = arith.addi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %19 = arith.minsi %18, %c32_i32 : i32
// CHECK-NEXT:  %20 = arith.subi %19, %arg1 : i32
// CHECK-NEXT:  %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:  %22 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %23 = arith.divsi %16, %c4_i32 : i32
// CHECK-NEXT:  %24 = arith.subi %c64_i32, %17 : i32
// CHECK-NEXT:  %25 = arith.divsi %24, %c32_i32 : i32
// CHECK-NEXT:  %26 = ascendc.construct !ascendc.data_copy_ext_params(%21, %11, %22, %25, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %27 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %23, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %26, %27 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %6 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2], %cst : tensor<32x32xf32, #asctile.global>, tensor<16x16xf32, #asctile.local<UB>>
  return %1: tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_load_4d_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<2x2x3x4xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<3x4x5x8xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<3x4x5x8xf32, #asctile.global> to !ascendc.global_tensor<3x4x5x8xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %c8_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = arith.muli %arg2, %c40_i32 : i32
// CHECK-NEXT:  %5 = arith.addi %3, %4 : i32
// CHECK-NEXT:  %6 = arith.muli %arg1, %c160_i32 : i32
// CHECK-NEXT:  %7 = arith.addi %5, %6 : i32
// CHECK-NEXT:  %8 = ascendc.global_tensor.subindex %1[%7] : !ascendc.global_tensor<3x4x5x8xf32>, i32, !ascendc.global_tensor<3x4x5x8xf32>
// CHECK-NEXT:  %9 = ascendc.local_tensor_auto veccalc() : <2x2x3x4xf32>
// CHECK-NEXT:  %10 = builtin.unrealized_conversion_cast %9 : !ascendc.local_tensor<2x2x3x4xf32> to tensor<2x2x3x4xf32, #asctile.local<UB>>
// CHECK-NEXT:  %11 = arith.subi %c8_i32, %arg4 : i32
// CHECK-NEXT:  %12 = arith.cmpi slt, %11, %c0_i32 : i32
// CHECK-NEXT:  %13 = arith.select %12, %c0_i32, %11 : i32
// CHECK-NEXT:  %14 = arith.minsi %13, %c4_i32 : i32
// CHECK-NEXT:  %15 = arith.muli %14, %c4_i32 : i32
// CHECK-NEXT:  %16 = arith.subi %c8_i32, %14 : i32
// CHECK-NEXT:  %17 = arith.remsi %15, %c32_i32 : i32
// CHECK-NEXT:  %18 = arith.cmpi eq, %17, %c0_i32 : i32
// CHECK-NEXT:  %19 = arith.subi %c32_i32, %17 : i32
// CHECK-NEXT:  %20 = arith.select %18, %c0_i32, %19 : i32
// CHECK-NEXT:  %21 = arith.addi %15, %20 : i32
// CHECK-NEXT:  %22 = arith.addi %arg1, %c2_i32 : i32
// CHECK-NEXT:  %23 = arith.minsi %22, %c3_i32 : i32
// CHECK-NEXT:  %24 = arith.subi %23, %arg1 : i32
// CHECK-NEXT:  %25 = arith.maxsi %24, %c0_i32 : i32
// CHECK-NEXT:  %26 = arith.muli %16, %c4_i32 : i32
// CHECK-NEXT:  %27 = arith.divsi %20, %c4_i32 : i32
// CHECK-NEXT:  %28 = arith.subi %c16_i32, %21 : i32
// CHECK-NEXT:  %29 = arith.divsi %28, %c32_i32 : i32
// CHECK-NEXT:  %30 = ascendc.construct !ascendc.data_copy_ext_params(%25, %15, %26, %29, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %31 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %27, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %9, %8, %30, %31 : !ascendc.local_tensor<2x2x3x4xf32>, !ascendc.global_tensor<3x4x5x8xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %10 : tensor<2x2x3x4xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_4d_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<2x2x3x4xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<3x4x5x8xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2, %arg3, %arg4], %cst : tensor<3x4x5x8xf32, #asctile.global>, tensor<2x2x3x4xf32, #asctile.local<UB>>
  return %1: tensor<2x2x3x4xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %7 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %8 = arith.cmpi slt, %7, %c0_i32 : i32
// CHECK-NEXT:  %9 = arith.select %8, %c0_i32, %7 : i32
// CHECK-NEXT:  %10 = arith.minsi %9, %c16_i32 : i32
// CHECK-NEXT:  %11 = arith.muli %10, %c4_i32 : i32
// CHECK-NEXT:  %12 = arith.subi %arg2, %10 : i32
// CHECK-NEXT:  %13 = arith.remsi %11, %c32_i32 : i32
// CHECK-NEXT:  %14 = arith.cmpi eq, %13, %c0_i32 : i32
// CHECK-NEXT:  %15 = arith.subi %c32_i32, %13 : i32
// CHECK-NEXT:  %16 = arith.select %14, %c0_i32, %15 : i32
// CHECK-NEXT:  %17 = arith.addi %11, %16 : i32
// CHECK-NEXT:  %18 = arith.addi %arg3, %c16_i32 : i32
// CHECK-NEXT:  %19 = arith.minsi %arg1, %18 : i32
// CHECK-NEXT:  %20 = arith.subi %19, %arg3 : i32
// CHECK-NEXT:  %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:  %22 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:  %23 = arith.divsi %16, %c4_i32 : i32
// CHECK-NEXT:  %24 = arith.subi %c64_i32, %17 : i32
// CHECK-NEXT:  %25 = arith.divsi %24, %c32_i32 : i32
// CHECK-NEXT:  %26 = ascendc.construct !ascendc.data_copy_ext_params(%21, %11, %22, %25, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %27 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %23, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %26, %27 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %6 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst : tensor<?x?xf32, #asctile.global>, tensor<16x16xf32, #asctile.local<UB>>
  return %1: tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_load_real_shape(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:        %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:   %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:   %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:   %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:   %7 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:   %8 = arith.cmpi slt, %7, %c0_i32 : i32
// CHECK-NEXT:   %9 = arith.select %8, %c0_i32, %7 : i32
// CHECK-NEXT:   %10 = arith.minsi %arg6, %9 : i32
// CHECK-NEXT:   %11 = arith.muli %10, %c4_i32 : i32
// CHECK-NEXT:   %12 = arith.subi %arg2, %10 : i32
// CHECK-NEXT:   %13 = arith.remsi %11, %c32_i32 : i32
// CHECK-NEXT:   %14 = arith.cmpi eq, %13, %c0_i32 : i32
// CHECK-NEXT:   %15 = arith.subi %c32_i32, %13 : i32
// CHECK-NEXT:   %16 = arith.select %14, %c0_i32, %15 : i32
// CHECK-NEXT:   %17 = arith.addi %11, %16 : i32
// CHECK-NEXT:   %18 = arith.cmpi slt, %17, %c64_i32 : i32
// CHECK-NEXT:   scf.if %18 {
// CHECK-NEXT:     ascendc.duplicate_l2 %5, %cst, %c0_i32 : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %19 = arith.addi %arg3, %c16_i32 : i32
// CHECK-NEXT:   %20 = arith.minsi %arg1, %19 : i32
// CHECK-NEXT:   %21 = arith.subi %20, %arg3 : i32
// CHECK-NEXT:   %22 = arith.maxsi %21, %c0_i32 : i32
// CHECK-NEXT:   %23 = arith.minsi %22, %arg5 : i32
// CHECK-NEXT:   %24 = arith.subi %c16_i32, %23 : i32
// CHECK-NEXT:   %25 = arith.cmpi sgt, %24, %c0_i32 : i32
// CHECK-NEXT:   scf.if %25 {
// CHECK-NEXT:     %32 = arith.muli %23, %c16_i32 : i32
// CHECK-NEXT:     %33 = ascendc.local_tensor.subindex %5[%32] : !ascendc.local_tensor<16x16xf32>, i32, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:     %34 = arith.muli %24, %c16_i32 : i32
// CHECK-NEXT:     ascendc.duplicate_l2 %33, %cst, %34 {asc.cal_count_set} : !ascendc.local_tensor<16x16xf32>, f32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %26 = arith.muli %12, %c4_i32 : i32
// CHECK-NEXT:   %27 = arith.divsi %16, %c4_i32 : i32
// CHECK-NEXT:   %28 = arith.subi %c64_i32, %17 : i32
// CHECK-NEXT:   %29 = arith.divsi %28, %c32_i32 : i32
// CHECK-NEXT:   %30 = ascendc.construct !ascendc.data_copy_ext_params(%23, %11, %26, %29, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:   %31 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %27, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:   ascendc.data_copy_pad_l0_ext %5, %4, %30, %31 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:   return %6 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: }
func.func @lower_load_real_shape(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst, (%arg5, %arg6) : tensor<?x?xf32, #asctile.global>, tensor<16x16xf32, #asctile.local<UB>>
  return %1: tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
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
func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  asctile.store %arg1, %0 [%arg2, %arg3] : tensor<16x16xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
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
func.func @lower_store_dynamic(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.store %arg1, %0 [%arg2, %arg3] : tensor<16x16xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32)
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.fixpipe_config(%14) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %13, %15 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf32, #asctile.global> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32)
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.fixpipe_config(%14) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %13, %15 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static_relu(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf32, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<32x32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf16, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<32x32xf16, #asctile.global> to !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf16>, i32, !ascendc.global_tensor<32x32xf16>
// CHECK-NEXT:  %6 = arith.subi %c32_i32, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %c32_i32, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %14 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %13 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.fixpipe_config(%15) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %14, %16 : !ascendc.global_tensor<32x32xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_static_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32) {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32xf16, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<32x32xf16, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf16, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf16, #asctile.global> to !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.subi %arg4, %arg2 : i32
// CHECK-NEXT:  %11 = arith.maxsi %10, %c0_i32 : i32
// CHECK-NEXT:  %12 = arith.minsi %11, %c16_i32 : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %14 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %9 : i32, "mSize" = %12 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %13 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %15 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.fixpipe_config(%15) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %14, %16 : !ascendc.global_tensor<?x?xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf16, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<?x?xf16, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_fixpipe_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x16xf32, #asctile.local<L0C>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %arg5 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %6 = arith.subi %arg5, %arg3 : i32
// CHECK-NEXT:  %7 = arith.cmpi slt, %6, %c0_i32 : i32
// CHECK-NEXT:  %8 = arith.select %7, %c0_i32, %6 : i32
// CHECK-NEXT:  %9 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.minsi %9, %arg7 : i32
// CHECK-NEXT:  %11 = arith.minsi %arg6, %c16_i32 : i32
// CHECK-NEXT:  %12 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %10 : i32, "mSize" = %11 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32)
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %14 = ascendc.construct !ascendc.fixpipe_config(%13) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %12, %14 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
func.func @lower_store_fixpipe_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<16x16xf32, #asctile.local<L0C>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3], (%arg6, %arg7) : tensor<16x16xf32, #asctile.local<L0C>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_store_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<2x8xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK:   %0 = builtin.unrealized_conversion_cast %arg1 : tensor<2x8xf32, #asctile.local<UB>> to !ascendc.local_tensor<2x8xf32>
// CHECK-NEXT:   %1 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %1 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:   %3 = arith.cmpi slt, %arg3, %c0_i32 : i32
// CHECK-NEXT:   %4 = arith.select %3, %c0_i32, %arg3 : i32
// CHECK-NEXT:   %5 = arith.minsi %arg5, %4 : i32
// CHECK-NEXT:   %6 = arith.muli %5, %c4_i32 : i32
// CHECK-NEXT:   %7 = arith.subi %c8_i32, %5 : i32
// CHECK-NEXT:   %8 = arith.subi %arg3, %5 : i32
// CHECK-NEXT:   %9 = arith.muli %7, %c4_i32 : i32
// CHECK-NEXT:   %10 = arith.divsi %9, %c32_i32 : i32
// CHECK-NEXT:   %11 = arith.muli %8, %c4_i32 : i32
// CHECK-NEXT:   %12 = ascendc.construct !ascendc.data_copy_ext_params(%arg4, %6, %10, %11, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:   ascendc.data_copy_pad_l2_ext %2, %0, %12 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<2x8xf32>, !ascendc.data_copy_ext_params
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @lower_store_real_shape(%arg0: memref<*xf32, 22>, %arg1: tensor<2x8xf32, #asctile.local<UB>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  asctile.store %arg1, %0 [%c0_i32, %c0_i32], (%arg4, %arg5) : tensor<2x8xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<16x64xf16, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<16x128xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<16x128xf16, #asctile.global> to !ascendc.global_tensor<16x128xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <16x64xf16>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x64xf16> to tensor<16x64xf16, #asctile.local<L1>>
// CHECK-NEXT:  %4 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c16_i32, %c64_i32, %c0_i32, %c128_i32, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %4 : !ascendc.local_tensor<16x64xf16>, !ascendc.global_tensor<16x128xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  return %3 : tensor<16x64xf16, #asctile.local<L1>>
func.func @lower_load_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<16x64xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<16x128xf16, #asctile.global>
  %1 = asctile.load %0[%c0_i32, %c0_i32], %cst : tensor<16x128xf16, #asctile.global>, tensor<16x64xf16, #asctile.local<L1>>
  return %1 : tensor<16x64xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<16x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<16x128xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<16x128xf32, #asctile.global> to !ascendc.global_tensor<16x128xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <16x64xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  %4 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c16_i32, %c64_i32, %c0_i32, %c128_i32, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %4 : !ascendc.local_tensor<16x64xf32>, !ascendc.global_tensor<16x128xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  return %3 : tensor<16x64xf32, #asctile.local<L1>>
func.func @lower_load_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<16x64xf32, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<16x128xf32, #asctile.global>
  %1 = asctile.load %0[%c0_i32, %c0_i32], %cst : tensor<16x128xf32, #asctile.global>, tensor<16x64xf32, #asctile.local<L1>>
  return %1 : tensor<16x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp16_dynamic(%arg0: memref<*xf16, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf16, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf16, 22>, tensor<?x?xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf16, #asctile.global> to !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <16x64xf16>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x64xf16> to tensor<16x64xf16, #asctile.local<L1>>
// CHECK-NEXT:  %7 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %8 = arith.minsi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %9 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %10 = arith.maxsi %9, %c0_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %12 = arith.maxsi %11, %c0_i32 : i32
// CHECK-NEXT:  %13 = arith.minsi %8, %10 : i32
// CHECK-NEXT:  %14 = arith.minsi %7, %12 : i32
// CHECK-NEXT:  %15 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %15 {
// CHECK-NEXT:    %20 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:    %22 = arith.minsi %13, %21 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %22 step %c1_i32  : i32 {
// CHECK-NEXT:      %23 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %24 = ascendc.global_tensor.subindex %4[%23] : !ascendc.global_tensor<?x?xf16>, i32, !ascendc.global_tensor<?x?xf16>
// CHECK-NEXT:      %25 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %26 = ascendc.local_tensor.subindex %5[%25] : !ascendc.local_tensor<16x64xf16>, i32, !ascendc.local_tensor<16x64xf16>
// CHECK-NEXT:      %27 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %14, %c0_i32, %14, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %26, %24, %27 : !ascendc.local_tensor<16x64xf16>, !ascendc.global_tensor<?x?xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %20 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %13, %14, %c0_i32, %arg2, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %20 : !ascendc.local_tensor<16x64xf16>, !ascendc.global_tensor<?x?xf16>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %16 = arith.muli %13, %c128_i32 : i32
// CHECK-NEXT:  %17 = arith.subi %c2048_i32, %16 : i32
// CHECK-NEXT:  %18 = arith.divsi %17, %c32_i32 : i32
// CHECK-NEXT:  %19 = arith.cmpi sgt, %18, %c0_i32 : i32
// CHECK-NEXT:  scf.if %19 {
// CHECK-NEXT:    %20 = arith.muli %13, %c16_i32 : i32
// CHECK-NEXT:    %21 = arith.subi %c16_i32, %13 : i32
// CHECK-NEXT:    %22 = ascendc.local_tensor.subindex %5[%20] : !ascendc.local_tensor<16x64xf16>, i32, !ascendc.local_tensor<16x64xf16>
// CHECK-NEXT:    %23 = ascendc.construct !ascendc.init_const_value_params(%c4_i32, %21, %13, %c0_i32) [ui16, ui16, ui16, f16] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %22, %23 : !ascendc.local_tensor<16x64xf16>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  return %6 : tensor<16x64xf16, #asctile.local<L1>>
func.func @lower_load_gm_l1_fp16_dynamic(%arg0: memref<*xf16, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf16, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf16, 22>, tensor<?x?xf16, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst : tensor<?x?xf16, #asctile.global>, tensor<16x64xf16, #asctile.local<L1>>
  return %1 : tensor<16x64xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_load_gm_l1_fp32_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf32, #asctile.local<L1>> {
// CHECK:       %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<?x?xf32, #asctile.global> to !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %2 = arith.muli %arg3, %arg2 : i32
// CHECK-NEXT:  %3 = arith.addi %arg4, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto a1() : <16x64xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L1>>
// CHECK-NEXT:  %7 = arith.minsi %arg2, %c64_i32 : i32
// CHECK-NEXT:  %8 = arith.minsi %arg1, %c16_i32 : i32
// CHECK-NEXT:  %9 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:  %10 = arith.maxsi %9, %c0_i32 : i32
// CHECK-NEXT:  %11 = arith.subi %arg2, %arg4 : i32
// CHECK-NEXT:  %12 = arith.maxsi %11, %c0_i32 : i32
// CHECK-NEXT:  %13 = arith.minsi %8, %10 : i32
// CHECK-NEXT:  %14 = arith.minsi %7, %12 : i32
// CHECK-NEXT:  %15 = arith.cmpi sgt, %arg2, %c65535_i32 : i32
// CHECK-NEXT:  scf.if %15 {
// CHECK-NEXT:    %20 = arith.subi %arg1, %arg3 : i32
// CHECK-NEXT:    %21 = arith.maxsi %20, %c0_i32 : i32
// CHECK-NEXT:    %22 = arith.minsi %13, %21 : i32
// CHECK-NEXT:    scf.for %arg7 = %c0_i32 to %22 step %c1_i32  : i32 {
// CHECK-NEXT:      %23 = arith.muli %arg7, %arg2 : i32
// CHECK-NEXT:      %24 = ascendc.global_tensor.subindex %4[%23] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
// CHECK-NEXT:      %25 = arith.muli %arg7, %c16_i32 : i32
// CHECK-NEXT:      %26 = ascendc.local_tensor.subindex %5[%25] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
// CHECK-NEXT:      %27 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c1_i32, %14, %c0_i32, %14, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:      ascendc.data_copy_l2 %26, %24, %27 : !ascendc.local_tensor<16x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:    }
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %20 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %13, %14, %c0_i32, %arg2, %c16_i32, %c1_i32, %c0_i32) [ui16, ui16, ui32, ui64, ui32, ui16, ui16, ui64] : i32, i32, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:    ascendc.data_copy_l2 %5, %4, %20 : !ascendc.local_tensor<16x64xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.nd2nz_params
// CHECK-NEXT:  }
// CHECK-NEXT:  %16 = arith.muli %13, %c256_i32 : i32
// CHECK-NEXT:  %17 = arith.subi %c4096_i32, %16 : i32
// CHECK-NEXT:  %18 = arith.divsi %17, %c32_i32 : i32
// CHECK-NEXT:  %19 = arith.cmpi sgt, %18, %c0_i32 : i32
// CHECK-NEXT:  scf.if %19 {
// CHECK-NEXT:    %20 = arith.muli %13, %c8_i32 : i32
// CHECK-NEXT:    %21 = arith.subi %c16_i32, %13 : i32
// CHECK-NEXT:    %22 = ascendc.local_tensor.subindex %5[%20] : !ascendc.local_tensor<16x64xf32>, i32, !ascendc.local_tensor<16x64xf32>
// CHECK-NEXT:    %23 = ascendc.construct !ascendc.init_const_value_params(%c8_i32, %21, %13, %c0_i32) [ui16, ui16, ui16, f32] : i32, i32, i32, i32
// CHECK-NEXT:    ascendc.fill %22, %23 : !ascendc.local_tensor<16x64xf32>, !ascendc.init_const_value_params
// CHECK-NEXT:  }
// CHECK-NEXT:  return %6 : tensor<16x64xf32, #asctile.local<L1>>
func.func @lower_load_gm_l1_fp32_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> tensor<16x64xf32, #asctile.local<L1>> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %1 = asctile.load %0[%arg3, %arg4], %cst : tensor<?x?xf32, #asctile.global>, tensor<16x64xf32, #asctile.local<L1>>
  return %1 : tensor<16x64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0A>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto a2() : <32x16xf16>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x16xf16> to tensor<32x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<32x16xf16, #asctile.local<L0A>>
func.func @lower_copy_l1_l0a_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<32x16xf16, #asctile.local<L0A>>
  return %0: tensor<32x16xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0A>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto a2() : <16x32xf16>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L0A>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c256_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x32xf16, #asctile.local<L0A>>
func.func @lower_copy_l1_l0a_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<16x32xf16, #asctile.local<L0A>>
  return %0: tensor<16x32xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_rows_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<32x16xf32, #asctile.local<L0A>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf32, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto a2() : <32x16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x16xf32> to tensor<32x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<32x16xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<32x16xf32, #asctile.local<L0A>>
func.func @lower_copy_l1_l0a_multiple_rows_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<32x16xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<32x32xf32, #asctile.local<L1>>, tensor<32x16xf32, #asctile.local<L0A>>
  return %0: tensor<32x16xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_multiple_cols_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0A>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf32, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto a2() : <16x32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L0A>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c128_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c4_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x32xf32, #asctile.local<L0A>>
func.func @lower_copy_l1_l0a_multiple_cols_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : tensor<32x32xf32, #asctile.local<L1>>, tensor<16x32xf32, #asctile.local<L0A>>
  return %0: tensor<16x32xf32, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0B>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto b2() : <32x16xf16>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x16xf16> to tensor<32x16xf16, #asctile.local<L0B>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<32x16xf16, #asctile.local<L0B>>
func.func @lower_copy_l1_l0b_multiple_rows(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<32x16xf16, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<32x16xf16, #asctile.local<L0B>>
  return %0: tensor<32x16xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0B>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf16, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto b2() : <16x32xf16>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf16> to tensor<16x32xf16, #asctile.local<L0B>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c256_i32] : !ascendc.local_tensor<32x32xf16>, i32, !ascendc.local_tensor<32x32xf16>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x32xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x32xf16, #asctile.local<L0B>>
func.func @lower_copy_l1_l0b_multiple_cols(%arg0: tensor<32x32xf16, #asctile.local<L1>>) -> tensor<16x32xf16, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c0_i32] : tensor<32x32xf16, #asctile.local<L1>>, tensor<16x32xf16, #asctile.local<L0B>>
  return %0: tensor<16x32xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_b_trans_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0B>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32x32xf32, #asctile.local<L1>> to !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto b2() : <16x32xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<L0B>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c512_i32] : !ascendc.local_tensor<32x32xf32>, i32, !ascendc.local_tensor<32x32xf32>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c2_i32 : i32, "kStep" = %c2_i32 : i32, "srcStride" = %c2_i32 : i32, "dstStride" = %c2_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x32xf32, #asctile.local<L0B>>
func.func @lower_copy_l1_l0b_b_trans_f32(%arg0: tensor<32x32xf32, #asctile.local<L1>>) -> tensor<16x32xf32, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] {asctile.transpose_b} : tensor<32x32xf32, #asctile.local<L1>>, tensor<16x32xf32, #asctile.local<L0B>>
  return %0: tensor<16x32xf32, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0b_diff_src_dst_f32(%arg0: tensor<96x128xf32, #asctile.local<L1>>) -> tensor<16x64xf32, #asctile.local<L0B>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<96x128xf32, #asctile.local<L1>> to !ascendc.local_tensor<96x128xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto b2() : <16x64xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x64xf32> to tensor<16x64xf32, #asctile.local<L0B>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c6272_i32] : !ascendc.local_tensor<96x128xf32>, i32, !ascendc.local_tensor<96x128xf32>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c8_i32 : i32, "srcStride" = %c6_i32 : i32, "dstStride" = %c4_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x64xf32>, !ascendc.local_tensor<96x128xf32>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x64xf32, #asctile.local<L0B>>
func.func @lower_copy_l1_l0b_diff_src_dst_f32(%arg0: tensor<96x128xf32, #asctile.local<L1>>) -> tensor<16x64xf32, #asctile.local<L0B>> {
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c64_i32] : tensor<96x128xf32, #asctile.local<L1>>, tensor<16x64xf32, #asctile.local<L0B>>
  return %0: tensor<16x64xf32, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_diff_src_dst(%arg0: tensor<160x96xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<160x96xf16, #asctile.local<L1>> to !ascendc.local_tensor<160x96xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto a2() : <16x16xf16>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x16xf16> to tensor<16x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c2560_i32] : !ascendc.local_tensor<160x96xf16>, i32, !ascendc.local_tensor<160x96xf16>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c10_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %false : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<160x96xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x16xf16, #asctile.local<L0A>>
func.func @lower_copy_l1_l0a_diff_src_dst(%arg0: tensor<160x96xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = asctile.copy %arg0[%c0_i32, %c16_i32] : tensor<160x96xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0A>>
  return %0: tensor<16x16xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_copy_l1_l0a_trans_diff_src_dst(%arg0: tensor<96x160xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>>
// CHECK:         %0 = builtin.unrealized_conversion_cast %arg0 : tensor<96x160xf16, #asctile.local<L1>> to !ascendc.local_tensor<96x160xf16>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto a2() : <16x16xf16>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x16xf16> to tensor<16x16xf16, #asctile.local<L0A>>
// CHECK-NEXT:    %3 = ascendc.local_tensor.subindex %0[%c3328_i32] : !ascendc.local_tensor<96x160xf16>, i32, !ascendc.local_tensor<96x160xf16>
// CHECK-NEXT:    %4 = emitasc.init_struct !ascendc.load_data_2d_params_v2("mStep" = %c1_i32 : i32, "kStep" = %c1_i32 : i32, "srcStride" = %c6_i32 : i32, "dstStride" = %c1_i32 : i32, "ifTranspose" = %true : i1)
// CHECK-NEXT:    ascendc.load_data_l0_v2 %1, %3, %4 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<96x160xf16>, !ascendc.load_data_2d_params_v2
// CHECK-NEXT:    return %2 : tensor<16x16xf16, #asctile.local<L0A>>
func.func @lower_copy_l1_l0a_trans_diff_src_dst(%arg0: tensor<96x160xf16, #asctile.local<L1>>) -> tensor<16x16xf16, #asctile.local<L0A>> {
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = asctile.copy %arg0[%c16_i32, %c32_i32] {asctile.transpose_a} : tensor<96x160xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0A>>
  return %0: tensor<16x16xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @lower_load_bias_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<64xf32, #asctile.local<L1>>
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<64xf32, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<64xf32, #asctile.global> to !ascendc.global_tensor<64xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <64xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<64xf32> to tensor<64xf32, #asctile.local<L1>>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<64xf32>, i32
// CHECK-NEXT:  return %3 : tensor<64xf32, #asctile.local<L1>>
func.func @lower_load_bias_gm_l1_fp32(%arg0: memref<*xf32, 22>) -> tensor<64xf32, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<64xf32, #asctile.global>
  %1 = asctile.load %0[%c0_i32], %cst {asctile.is_bias} : tensor<64xf32, #asctile.global>, tensor<64xf32, #asctile.local<L1>>
  return %1 : tensor<64xf32, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_bias_l1_bt_fp32(%arg0: tensor<64xf32, #asctile.local<L1>>) -> tensor<64xf32, #asctile.local<BT>>
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64xf32, #asctile.local<L1>> to !ascendc.local_tensor<64xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto c2() : <64xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<64xf32> to tensor<64xf32, #asctile.local<BT>>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.data_copy_params(%c1_i32, %c8_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l0 %1, %0, %3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.data_copy_params
// CHECK-NEXT:  return %2 : tensor<64xf32, #asctile.local<BT>>
func.func @lower_copy_bias_l1_bt_fp32(%arg0: tensor<64xf32, #asctile.local<L1>>) -> tensor<64xf32, #asctile.local<BT>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32] : tensor<64xf32, #asctile.local<L1>>, tensor<64xf32, #asctile.local<BT>>
  return %0 : tensor<64xf32, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @lower_load_bias_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<64xf16, #asctile.local<L1>>
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<64xf16, #asctile.global>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : tensor<64xf16, #asctile.global> to !ascendc.global_tensor<64xf16>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto a1() : <64xf16>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<64xf16> to tensor<64xf16, #asctile.local<L1>>
// CHECK-NEXT:  ascendc.data_copy_l2 %2, %1, %c64_i32 : !ascendc.local_tensor<64xf16>, !ascendc.global_tensor<64xf16>, i32
// CHECK-NEXT:  return %3 : tensor<64xf16, #asctile.local<L1>>
func.func @lower_load_bias_gm_l1_fp16(%arg0: memref<*xf16, 22>) -> tensor<64xf16, #asctile.local<L1>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %0 = asctile.tensor %arg0() : memref<*xf16, 22>, tensor<64xf16, #asctile.global>
  %1 = asctile.load %0[%c0_i32], %cst {asctile.is_bias} : tensor<64xf16, #asctile.global>, tensor<64xf16, #asctile.local<L1>>
  return %1 : tensor<64xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @lower_copy_bias_l1_bt_fp16(%arg0: tensor<64xf16, #asctile.local<L1>>) -> tensor<64xf16, #asctile.local<BT>>
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64xf16, #asctile.local<L1>> to !ascendc.local_tensor<64xf16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto c2() : <64xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<64xf32> to tensor<64xf16, #asctile.local<BT>>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.data_copy_params(%c1_i32, %c4_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_l0 %1, %0, %3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf16>, !ascendc.data_copy_params
// CHECK-NEXT:  return %2 : tensor<64xf16, #asctile.local<BT>>
func.func @lower_copy_bias_l1_bt_fp16(%arg0: tensor<64xf16, #asctile.local<L1>>) -> tensor<64xf16, #asctile.local<BT>> {
  %c0_i32 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0_i32] : tensor<64xf16, #asctile.local<L1>>, tensor<64xf16, #asctile.local<BT>>
  return %0 : tensor<64xf16, #asctile.local<BT>>
}

// CHECK-LABEL: func.func @lower_store_3d_tensor
// CHECK-DAG: %[[STRIDE1:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[COUNT1:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[PARAMS:.*]] = ascendc.construct !ascendc.loop_mode_params(%[[COUNT1]], %[[ONE]], %[[STRIDE1]], %[[STRIDE1]]
// CHECK: ascendc.set_loop_mode_para %[[PARAMS]] {mvType = 0 : i32}
// CHECK: ascendc.data_copy_pad
// CHECK: ascendc.reset_loop_mode_para {mvType = 0 : i32}

func.func @lower_store_3d_tensor(%arg0: memref<*xf32, 22>, %arg1: tensor<32x32x32xf32, #asctile.local<UB>>) {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32x32xf32, #asctile.global>
    %c0_i32 = arith.constant 0 : i32
    asctile.store %arg1, %0[%c0_i32, %c0_i32, %c0_i32] : tensor<32x32x32xf32, #asctile.local<UB>>, tensor<32x32x32xf32, #asctile.global>
    return
}

// CHECK-LABEL: func.func @lower_store_4d_tensor
// CHECK-DAG: %[[STRIDE1:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[STRIDE2:.*]] = arith.constant 131072 : i32
// CHECK-DAG: %[[COUNT1:.*]] = arith.constant 32 : i32
// CHECK: %[[PARAMS:.*]] = ascendc.construct !ascendc.loop_mode_params(%[[COUNT1]], %[[COUNT1:.*]], %[[STRIDE1]], %[[STRIDE1]], %[[STRIDE2]], %[[STRIDE2]]
// CHECK: ascendc.set_loop_mode_para %[[PARAMS]] {mvType = 0 : i32}
// CHECK: ascendc.data_copy_pad
// CHECK: ascendc.reset_loop_mode_para {mvType = 0 : i32}


func.func @lower_store_4d_tensor(%arg0: memref<*xf32, 22>, %arg1: tensor<32x32x32x32xf32, #asctile.local<UB>>) {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x32x32x32xf32, #asctile.global>
    %c0_i32 = arith.constant 0 : i32
    asctile.store %arg1, %0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] : tensor<32x32x32x32xf32, #asctile.local<UB>>, tensor<32x32x32x32xf32, #asctile.global>
    return
}
