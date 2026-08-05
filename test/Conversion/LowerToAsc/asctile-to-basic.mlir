// RUN: ascir-opt -asclower-asctile-to-basic -canonicalize -split-input-file %s | FileCheck %s


// CHECK-LABEL: func.func @lower_relu(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.relu_l2 %1, %0, %c32_i64 : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_relu(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_matmul(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto co1() : <8x8xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x8xf32> to tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %4 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  return %3 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %0 = asctile.matmul %arg0, %arg1 : tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>> -> tensor<8x8xf32, #asctile.local<L0C>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_matmul_bias(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8xf32, #asctile.local<BT>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg2 : tensor<8xf32, #asctile.local<BT>> to !ascendc.local_tensor<8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto co1() : <8x8xf32>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<8x8xf32> to tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:  %5 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %false : i1, "cmatrixSource" = %true : i1)
// CHECK-NEXT:  ascendc.mmad_with_bias %3, %2, %1, %0, %5 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.local_tensor<8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  return %4 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_bias(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8xf32, #asctile.local<BT>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %0 = asctile.matmul %arg0, %arg1, %arg2 : tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>, tensor<8xf32, #asctile.local<BT>> -> tensor<8x8xf32, #asctile.local<L0C>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_matmul_hf32(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto co1() : <8x8xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x8xf32> to tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:  ascendc.set_hf32_mode %true : i1
// CHECK-NEXT:  ascendc.set_hf32_trans_mode %true : i1
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %4 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  ascendc.set_hf32_mode %false : i1
// CHECK-NEXT:  return %3 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_hf32(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %0 = asctile.matmul %arg0, %arg1 {hf32} : tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>> -> tensor<8x8xf32, #asctile.local<L0C>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_matmul_acc(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg2 : tensor<8x8xf32, #asctile.local<L0C>> to !ascendc.local_tensor<8x8xf32>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %true : i1, "cmatrixSource" = %false : i1)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %3 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_acc(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  asctile.matmul_acc %arg2, %arg0, %arg1 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>
  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_matmul_acc_hf32(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg2 : tensor<8x8xf32, #asctile.local<L0C>> to !ascendc.local_tensor<8x8xf32>
// CHECK-NEXT:  ascendc.set_hf32_mode %true : i1
// CHECK-NEXT:  ascendc.set_hf32_trans_mode %true : i1
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %true : i1, "cmatrixSource" = %false : i1)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %3 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  ascendc.set_hf32_mode %false : i1
// CHECK-NEXT:  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_acc_hf32(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  asctile.matmul_acc %arg2, %arg0, %arg1 {hf32} : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>
  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_matmul_acc_with_bias(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg2 : tensor<8x8xf32, #asctile.local<L0C>> to !ascendc.local_tensor<8x8xf32>
// CHECK-NEXT:  %3 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %false : i1, "cmatrixSource" = %true : i1)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %3 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_acc_with_bias(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  asctile.matmul_acc %arg2, %arg0, %arg1 {asctile.has_bias} : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>
  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_adds(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.adds_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_adds(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.adds %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_muls(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.muls_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_muls(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.muls %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_shls(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.shift_left_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_shls(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.shls %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_shrs(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.shift_right_l2 %1, %0, %arg1, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_shrs(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.shrs %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_sum_as_1d(%arg0: tensor<16x32x8xf32, #asctile.local<UB>>) -> f32 {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32x8xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32x8xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  ascendc.reduce_sum_l2 %1, %0, %2, %c4096_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16x32x8xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  return %3 : f32
// CHECK-NEXT:}
func.func @lower_reduce_sum_as_1d(%arg0: tensor<16x32x8xf32, #asctile.local<UB>>) -> f32 {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16x32x8xf32, #asctile.local<UB>>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_min_as_1d(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.reduce_min_l2 %1, %0, %2, %c16_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>, i64, i64
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  return %3 : f32
// CHECK-NEXT:}
func.func @lower_reduce_min_as_1d(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
  %0 = asctile.reduce_as_1d <min> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_reduce_max_as_1d(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <8xf32>
// CHECK-NEXT:  ascendc.reduce_max_l2 %1, %0, %2, %c16_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<8xf32>, i64, i64
// CHECK-NEXT:  %3 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  return %3 : f32
// CHECK-NEXT:}
func.func @lower_reduce_max_as_1d(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
  %0 = asctile.reduce_as_1d <max> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 5 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_floor(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 2 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_floor(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <floor> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_ceil(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 3 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_ceil(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <ceil> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_trunc(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 5 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_trunc(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <trunc> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_round(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 4 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_round(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <round> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_noround(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf16>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf16> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_noround(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
  %0 = asctile.cast <noround> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  return %0 : tensor<32xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_rint(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 1 : i32} : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_rint(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.cast <rint> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cast_round_odd(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf16>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf16> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c32_i64 {roundMode = 6 : i32} : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf32>, i64
// CHECK-NEXT:  return %2 : tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cast_round_odd(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
  %0 = asctile.cast <odd> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf16, #asctile.local<UB>>
  return %0 : tensor<32xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_transpose_ub(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<32x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32x16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32x16xf32> to tensor<32x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.trans_data_to_5hd_params(%false, %false, %c4_i32, %c16_i16, %c1_i16) [i1, i1, ui8, ui16, ui16] : i1, i1, i32, i16, i16
// CHECK-NEXT:  ascendc.trans_data_to_5hd_tensor %1, %0, %3 {dstOffsets = array<i32: 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480>, srcOffsets = array<i32: 0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920>} : !ascendc.local_tensor<32x16xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.trans_data_to_5hd_params
// CHECK-NEXT:  return %2 : tensor<32x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_transpose_ub(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<32x16xf32, #asctile.local<UB>> {
  %0 = asctile.transpose %arg0, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<UB>> to tensor<32x16xf32, #asctile.local<UB>>
  return %0 : tensor<32x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_subs(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<4x256xf32, #asctile.local<UB>> to !ascendc.local_tensor<4x256xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<4x256xf32> to tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  ascendc.duplicate_l2 %3, %arg1, %c1024_i64 : !ascendc.local_tensor<4x256xf32>, f32, i64
// CHECK-NEXT:  ascendc.sub_l2 %1, %0, %3, %c1024_i64 : !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4x256xf32>, i64
// CHECK-NEXT:  return %2 : tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_subs(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
  %0 = asctile.subs %arg0, %arg1 : tensor<4x256xf32, #asctile.local<UB>>
  return %0 : tensor<4x256xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_divs(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<4x256xf32, #asctile.local<UB>> to !ascendc.local_tensor<4x256xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<4x256xf32> to tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  ascendc.duplicate_l2 %3, %arg1, %c1024_i64 : !ascendc.local_tensor<4x256xf32>, f32, i64
// CHECK-NEXT:  ascendc.div_l2 %1, %0, %3, %c1024_i64 : !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4x256xf32>, i64
// CHECK-NEXT:  return %2 : tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_divs(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
  %0 = asctile.divs %arg0, %arg1 : tensor<4x256xf32, #asctile.local<UB>>
  return %0 : tensor<4x256xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_bitwise_not(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<32xi32, #asctile.local<UB>> to !ascendc.local_tensor<32xi32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xi32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xi32> to tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.not_l2 %1, %0, %c32_i64 : !ascendc.local_tensor<32xi32>, !ascendc.local_tensor<32xi32>, i64
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_bitwise_not(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = asctile.bitwise_not %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}


// CHECK-LABEL: func.func @lower_cmps_f32(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %4 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %2, %0, %arg1, %c0_i64, %c0_i64, %4 {cmpMode = 0 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf32>, f32, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %3 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_f32(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "LT" %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmps_i8(%arg0: tensor<16xi8, #asctile.local<UB>>, %arg1: i8) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi8, #asctile.local<UB>> to !ascendc.local_tensor<16xi8>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf16>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c16_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf16>, !ascendc.local_tensor<16xi8>, i64
// CHECK-NEXT:  %2 = arith.sitofp %arg1 : i8 to f16
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %4, %1, %2, %c0_i64, %c0_i64, %6 {cmpMode = 5 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf16>, f16, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %5 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_i8(%arg0: tensor<16xi8, #asctile.local<UB>>, %arg1: i8) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "NE" %arg0, %arg1 : tensor<16xi8, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmps_i16(%arg0: tensor<16xi16, #asctile.local<UB>>, %arg1: i16) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi16, #asctile.local<UB>> to !ascendc.local_tensor<16xi16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf16>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c16_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf16>, !ascendc.local_tensor<16xi16>, i64
// CHECK-NEXT:  %2 = arith.sitofp %arg1 : i16 to f16
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %4, %1, %2, %c0_i64, %c0_i64, %6 {cmpMode = 5 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf16>, f16, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %5 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_i16(%arg0: tensor<16xi16, #asctile.local<UB>>, %arg1: i16) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "NE" %arg0, %arg1 : tensor<16xi16, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmps_i32(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c16_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  %2 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %4, %1, %2, %c0_i64, %c0_i64, %6 {cmpMode = 5 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf32>, f32, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %5 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_i32(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "NE" %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// -----

module attributes {asc.compilation_arch = "c310"} {
// CHECK-LABEL: func.func @lower_subs_c310(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<4x256xf32, #asctile.local<UB>> to !ascendc.local_tensor<4x256xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<4x256xf32> to tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.subs_l2 %1, %0, %arg1, %c1024_i64 : !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4x256xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:}
  func.func @lower_subs_c310(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
    %0 = asctile.subs %arg0, %arg1 : tensor<4x256xf32, #asctile.local<UB>>
    return %0 : tensor<4x256xf32, #asctile.local<UB>>
  }

// CHECK-LABEL: func.func @lower_divs_c310(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<4x256xf32, #asctile.local<UB>> to !ascendc.local_tensor<4x256xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<4x256xf32> to tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.divs_l2 %1, %0, %arg1, %c1024_i64 : !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4x256xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:}
  func.func @lower_divs_c310(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<4x256xf32, #asctile.local<UB>> {
    %0 = asctile.divs %arg0, %arg1 : tensor<4x256xf32, #asctile.local<UB>>
    return %0 : tensor<4x256xf32, #asctile.local<UB>>
  }
}
