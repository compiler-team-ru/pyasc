// RUN: ascir-opt --asclower-asctile --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.duplicate_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
// CHECK-NEXT:  return %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

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

// CHECK-LABEL: func.func @lower_matmul_acc_from_accumulator(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto co1() : <8x8xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x8xf32> to tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %true : i1, "cmatrixSource" = %false : i1)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %4 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  return %3 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_acc_from_accumulator(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %0 = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
  asctile.matmul_acc %0, %arg0, %arg1 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_matmul_acc_from_accumulator_hf32(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x8xf32, #asctile.local<L0B>> to !ascendc.local_tensor<16x8xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<8x16xf32, #asctile.local<L0A>> to !ascendc.local_tensor<8x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto co1() : <8x8xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<8x8xf32> to tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:  ascendc.set_hf32_mode %true : i1
// CHECK-NEXT:  ascendc.set_hf32_trans_mode %true : i1
// CHECK-NEXT:  %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c8_i32 : i32, "n" = %c8_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %true : i1, "cmatrixSource" = %false : i1)
// CHECK-NEXT:  ascendc.mmad %2, %1, %0, %4 : !ascendc.local_tensor<8x8xf32>, !ascendc.local_tensor<8x16xf32>, !ascendc.local_tensor<16x8xf32>, !ascendc.mmad_params
// CHECK-NEXT:  ascendc.set_hf32_mode %false : i1
// CHECK-NEXT:  return %3 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @lower_matmul_acc_from_accumulator_hf32(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %0 = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
  asctile.matmul_acc %0, %arg0, %arg1 {hf32} : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @lower_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<8x32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  %1 = ascendc.reinterpret_cast %0 : !ascendc.local_tensor<16x16xf32> to !ascendc.local_tensor<8x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<8x32xf32> to tensor<8x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %2 : tensor<8x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<8x32xf32, #asctile.local<UB>> {
  %0 = asctile.reshape %arg0 : tensor<16x16xf32, #asctile.local<UB>> to tensor<8x32xf32, #asctile.local<UB>>
  return %0 : tensor<8x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<16x1xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16x1xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x1xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<1xf32, #asctile.local<UB>> to !ascendc.local_tensor<1xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = ascendc.local_tensor.get_value %1, %c0_i64 : !ascendc.local_tensor<1xf32>, i64, f32
// CHECK-NEXT:  ascendc.duplicate_l2 %2, %4, %c0_i64 : !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.broadcast %5, %0, %c1_i32, %c16_i32, %c32_i32, %c1_i32, %c16_i32, %c1_i32 {operandSegmentSizes = array<i32: 1, 1, 3, 3>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x1xf32>, i32, i32, i32, i32, i32, i32
// CHECK-NEXT:  return %3, %6 : tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_broadcast(%arg0: tensor<1xf32, #asctile.local<UB>>, %arg1: tensor<16x1xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>) {
  %0 = asctile.broadcast %arg0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
  %1 = asctile.broadcast %arg1 : tensor<16x1xf32, #asctile.local<UB>> to tensor<16x32xf32, #asctile.local<UB>>
  return %0, %1 : tensor<16xf32, #asctile.local<UB>>, tensor<16x32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_softmax(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <1152xui8>
// CHECK-NEXT:  %4 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:  %5 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c1_i32 : i32, "srcK" = %c16_i32 : i32, "oriSrcM" = %c1_i32 : i32, "oriSrcK" = %c16_i32 : i32)
// CHECK-NEXT:  ascendc.softmax %1, %0, %3, %4, %5 {operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 1, 1>} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<1152xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:  return %2 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_softmax(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.softmax %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_softmax_2D(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<16x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16x32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16x32xf32> to tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <3648xui8>
// CHECK-NEXT:  %4 = ascendc.construct !ascendc.softmax_tiling()
// CHECK-NEXT:  %5 = emitasc.init_struct !ascendc.softmax_shape_info("srcM" = %c16_i32 : i32, "srcK" = %c32_i32 : i32, "oriSrcM" = %c16_i32 : i32, "oriSrcK" = %c32_i32 : i32)
// CHECK-NEXT:  ascendc.softmax %1, %0, %3, %4, %5 {operandSegmentSizes = array<i32: 1, 0, 0, 1, 1, 1, 1>} : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<3648xui8>, !ascendc.softmax_tiling, !ascendc.softmax_shape_info
// CHECK-NEXT:  return %2 : tensor<16x32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_softmax_2D(%arg0: tensor<16x32xf32, #asctile.local<UB>>) -> tensor<16x32xf32, #asctile.local<UB>> {
  %0 = asctile.softmax %arg0 : tensor<16x32xf32, #asctile.local<UB>>
  return %0 : tensor<16x32xf32, #asctile.local<UB>>
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

// CHECK-LABEL: func.func @lower_reduce_sum(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_sum %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_sum(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <sum> %arg0 {dims = [0 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_min(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_min %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_min(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <min> %arg0 {dims = [0 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_max(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_max %1, %0, %3, %c64_i32, %c32_i32 {pattern = 1 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_max(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <max> %arg0 {dims = [1 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_with_reuse(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_max %1, %0, %3, %c64_i32, %c32_i32 {isReuseSource, pattern = 1 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_with_reuse(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <max> %arg0 {asctile.reuse_source, dims = [1 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_reduce_prod(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<64x32xf32, #asctile.local<UB>> to !ascendc.local_tensor<64x32xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <8192xui8>
// CHECK-NEXT:  ascendc.reduce_prod %1, %0, %3, %c64_i32, %c32_i32 {pattern = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<64x32xf32>, !ascendc.local_tensor<8192xui8>, i32, i32
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_reduce_prod(%arg0: tensor<64x32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.reduce <prod> %arg0 {dims = [0 : i32]} : tensor<64x32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
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

// CHECK-LABEL: func.func @lower_inline_vf(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> (tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<32xi32, #asctile.local<UB>> to !ascendc.local_tensor<32xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <32xf16>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<32xf16> to tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  ascvf.vf_group %2, %c0_i32 : !ascendc.local_tensor<32xf16>, i32 {
// CHECK-NEXT:    ascvf.vec_scope {
// CHECK-NEXT:      emitasc.verbatim ";;; // $0" %2 : !ascendc.local_tensor<32xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {operandSegmentSizes = array<i32: 1, 0, 1>}
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <32xi16>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<32xi16> to tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:  ascvf.vf_group %4, %1, %0, %c0_i32 : !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xi32>, i32 {
// CHECK-NEXT:    ascvf.vec_scope {
// CHECK-NEXT:      emitasc.verbatim ";;; // $0 $1 $2" %4, %1, %0 : !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } {operandSegmentSizes = array<i32: 1, 2, 1>}
// CHECK-NEXT:  return %3, %5 : tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_inline_vf(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> (tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>) {
  %0 = asctile.inline_vf() ";;; // $0" : () -> tensor<32xf16, #asctile.local<UB>>
  %1 = asctile.inline_vf(%arg0, %arg1) ";;; // $0 $1 $2" : (tensor<32xf32, #asctile.local<UB>>, tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi16, #asctile.local<UB>>
  return %0, %1 : tensor<32xf16, #asctile.local<UB>>, tensor<32xi16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_layer_norm(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: tensor<256xf32, #asctile.local<UB>>, %arg2: tensor<256xf32, #asctile.local<UB>>) -> (tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>) {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg2 : tensor<256xf32, #asctile.local<UB>> to !ascendc.local_tensor<256xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg1 : tensor<256xf32, #asctile.local<UB>> to !ascendc.local_tensor<256xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg0 : tensor<4x256xf32, #asctile.local<UB>> to !ascendc.local_tensor<4x256xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <4x256xf32>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<4x256xf32> to tensor<4x256xf32, #asctile.local<UB>>
// CHECK-NEXT:  %5 = ascendc.local_tensor_auto veccalc() : <4xf32>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<4xf32> to tensor<4x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %7 = ascendc.local_tensor_auto veccalc() : <4xf32>
// CHECK-NEXT:  %8 = builtin.unrealized_conversion_cast %7 : !ascendc.local_tensor<4xf32> to tensor<4x1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %9 = emitasc.init_struct !ascendc.layernorm_tiling("bLength" = %c4_i32 : i32, "sLength" = %c1_i32 : i32, "hLength" = %c256_i32 : i32, "originalHLength" = %c256_i32 : i32, "inputXSize" = %c1024_i32 : i32, "meanVarSize" = %c4_i32 : i32, "numberOfTmpBuf" = %c3_i32 : i32, "meanTmpTensorPos" = %c3072_i32 : i32, "meanTmpTensorSize" = %c4_i32 : i32, "varianceTmpTensorPos" = %c3076_i32 : i32, "varianceTmpTensorSize" = %c4_i32 : i32, "tmpBufSize" = %c65536_i32 : i32, "oneTmpSize" = %c1024_i32 : i32, "firstTmpStartPos" = %c0_i32 : i32, "secondTmpStartPos" = %c1024_i32 : i32, "thirdTmpStartPos" = %c2048_i32 : i32, "loopRound" = %c1_i32 : i32, "inputRoundSize" = %c1024_i32 : i32, "inputTailSize" = %c0_i32 : i32, "inputTailPos" = %c1024_i32 : i32, "meanVarRoundSize" = %c4_i32 : i32, "meanVarTailSize" = %c0_i32 : i32, "meanVarTailPos" = %c4_i32 : i32, "bshCurLength" = %c1024_i32 : i32, "bsCurLength" = %c4_i32 : i32, "lastDimValueBack" = %cst : f32)
// CHECK-NEXT:  ascendc.layer_norm %3, %5, %7, %2, %1, %0, %cst_0, %9 : !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<4xf32>, !ascendc.local_tensor<4xf32>, !ascendc.local_tensor<4x256xf32>, !ascendc.local_tensor<256xf32>, !ascendc.local_tensor<256xf32>, f32, !ascendc.layernorm_tiling
// CHECK-NEXT:  return %4, %6, %8 : tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_layer_norm(%arg0: tensor<4x256xf32, #asctile.local<UB>>, %arg1: tensor<256xf32, #asctile.local<UB>>, %arg2: tensor<256xf32, #asctile.local<UB>>) -> (tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>) {
  %cst = arith.constant 1.000000e-05 : f32
  %result, %mean, %var = asctile.layer_norm %arg0, %arg1, %arg2, %cst : tensor<4x256xf32, #asctile.local<UB>>, tensor<256xf32, #asctile.local<UB>>, tensor<256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>
  return %result, %mean, %var : tensor<4x256xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>, tensor<4x1xf32, #asctile.local<UB>>
}
