// RUN: ascir-opt -asclower-asctile-data-transfer -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> !asctile.tile<16x16xf32, UB> {
// CHECK:       %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !asctile.tensor<32x32xf32> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %2 = arith.muli %arg1, %c32_i32 : i32
// CHECK-NEXT:  %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:  %4 = ascendc.global_tensor.subindex %1[%3] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x16xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x16xf32> to !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:  %7 = arith.subi %c1024_i32, %3 : i32
// CHECK-NEXT:  %8 = arith.minsi %7, %c16_i32 : i32
// CHECK-NEXT:  %9 = arith.muli %8, %c4_i32 : i32
// CHECK-NEXT:  %10 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %9, %c64_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %11 = arith.addi %3, %c-896_i32 : i32
// CHECK-NEXT:  %12 = arith.maxsi %11, %c0_i32 : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %12, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %10, %13 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<32x32xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %6 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:}
func.func @lower_load_static(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg1, %arg2], %cst : !asctile.tensor<32x32xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
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
// CHECK-NEXT:  %8 = arith.subi %7, %3 : i32
// CHECK-NEXT:  %9 = arith.subi %arg2, %c16_i32 : i32
// CHECK-NEXT:  %10 = arith.muli %9, %c4_i32 : i32
// CHECK-NEXT:  %11 = arith.minsi %8, %c16_i32 : i32
// CHECK-NEXT:  %12 = arith.muli %11, %c4_i32 : i32
// CHECK-NEXT:  %13 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %12, %10, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  %14 = arith.subi %c128_i32, %8 : i32
// CHECK-NEXT:  %15 = arith.maxsi %14, %c0_i32 : i32
// CHECK-NEXT:  %16 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %15, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
// CHECK-NEXT:  ascendc.data_copy_pad_l0_ext %5, %4, %13, %16 : !ascendc.local_tensor<16x16xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
// CHECK-NEXT:  return %6 : !asctile.tile<16x16xf32, UB>
// CHECK-NEXT:}
func.func @lower_load_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !asctile.tile<16x16xf32, UB> {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = asctile.load %0 [%arg3, %arg4], %cst : !asctile.tensor<?x?xf32>, !asctile.tile<16x16xf32, UB>
  return %1: !asctile.tile<16x16xf32, UB>
}

// CHECK-LABEL: func.func @lower_store_static(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, UB>, %arg2: i32, %arg3: i32) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16x16xf32, UB> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<32x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !asctile.tensor<32x32xf32> to !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %3 = arith.muli %arg2, %c32_i32 : i32
// CHECK-NEXT:  %4 = arith.addi %arg3, %3 : i32
// CHECK-NEXT:  %5 = ascendc.global_tensor.subindex %2[%4] : !ascendc.global_tensor<32x32xf32>, i32, !ascendc.global_tensor<32x32xf32>
// CHECK-NEXT:  %6 = arith.subi %c1024_i32, %4 : i32
// CHECK-NEXT:  %7 = arith.minsi %6, %c16_i32 : i32
// CHECK-NEXT:  %8 = arith.muli %7, %c4_i32 : i32
// CHECK-NEXT:  %9 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %8, %c0_i32, %c64_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %5, %0, %9 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
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
// CHECK-NEXT:  %6 = arith.muli %arg4, %arg5 : i32
// CHECK-NEXT:  %7 = arith.subi %6, %4 : i32
// CHECK-NEXT:  %8 = arith.subi %arg5, %c16_i32 : i32
// CHECK-NEXT:  %9 = arith.muli %8, %c4_i32 : i32
// CHECK-NEXT:  %10 = arith.minsi %7, %c16_i32 : i32
// CHECK-NEXT:  %11 = arith.muli %10, %c4_i32 : i32
// CHECK-NEXT:  %12 = ascendc.construct !ascendc.data_copy_ext_params(%c16_i32, %11, %c0_i32, %9, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
// CHECK-NEXT:  ascendc.data_copy_pad_l2_ext %5, %0, %12 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.data_copy_ext_params
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
// CHECK-NEXT:  %6 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32)
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.fixpipe_config(%7) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %6, %8 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
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
// CHECK-NEXT:  %6 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32)
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.fixpipe_config(%7) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %6, %8 : !ascendc.global_tensor<32x32xf32>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
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
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %7 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %c32_i32 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %6 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %9 = ascendc.construct !ascendc.fixpipe_config(%8) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %7, %9 : !ascendc.global_tensor<32x32xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
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
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.quant_mode_t(%c1_i32) [!ascendc.quant_mode_t] constexpr static : i32
// CHECK-NEXT:  %7 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c16_i32 : i32, "mSize" = %c16_i32 : i32, "srcStride" = %c16_i32 : i32, "dstStride" = %arg5 : i32, "reluEn" = %c1_i32 : i32, "quantPre" = %6 : !ascendc.quant_mode_t)
// CHECK-NEXT:  %8 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
// CHECK-NEXT:  %9 = ascendc.construct !ascendc.fixpipe_config(%8) constexpr static : !ascendc.co2_layout
// CHECK-NEXT:  ascendc.fixpipe %5, %0, %7, %9 : !ascendc.global_tensor<?x?xf16>, !ascendc.local_tensor<16x16xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_store_fixpipe_dynamic_relu_quantize(%arg0: memref<*xf32, 22>, %arg1: !asctile.tile<16x16xf32, L0C>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = asctile.tensor %arg0(%arg4, %arg5) : memref<*xf32, 22>, !asctile.tensor<?x?xf16>
  asctile.store_fixpipe %arg1, %0 [%arg2, %arg3] {quantize, relu} : !asctile.tile<16x16xf32, L0C>, !asctile.tensor<?x?xf16>
  return
}
