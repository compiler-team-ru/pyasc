// RUN: ascir-opt -split-input-file -verify-diagnostics %s

func.func @concat_non_ub_location(%arg0: tensor<16xf32, #asctile.local<L1>>, %arg1: tensor<16xf32, #asctile.local<L1>>) -> tensor<32xf32, #asctile.local<L1>> {
  // expected-error@below {{tensor operands must have UB tensor location}}
  %0 = asctile.concat %arg0, %arg1 : tensor<16xf32, #asctile.local<L1>>, tensor<16xf32, #asctile.local<L1>> -> tensor<32xf32, #asctile.local<L1>>
  return %0 : tensor<32xf32, #asctile.local<L1>>
}

// -----

func.func @concat_mismatched_shapes(%arg0: tensor<16x32xf32, #asctile.local<UB>>, %arg1: tensor<16x64xf32, #asctile.local<UB>>) -> tensor<32x32xf32, #asctile.local<UB>> {
  // expected-error@below {{tensor operands must have the same shape except their first dimension}}
  %0 = asctile.concat %arg0, %arg1 : tensor<16x32xf32, #asctile.local<UB>>, tensor<16x64xf32, #asctile.local<UB>> -> tensor<32x32xf32, #asctile.local<UB>>
  return %0 : tensor<32x32xf32, #asctile.local<UB>>
}

// -----

func.func @concat_wrong_result_shape(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<64xf32, #asctile.local<UB>> {
  // expected-error@below {{result tensor shape must be [32]}}
  %0 = asctile.concat %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>> -> tensor<64xf32, #asctile.local<UB>>
  return %0 : tensor<64xf32, #asctile.local<UB>>
}

// -----

func.func @tensor_wrong_sizes(%arg0: memref<*xf32, 22>, %arg1: i32) -> tensor<?x?xf32, #asctile.global> {
  // expected-error@below {{must have value in 'sizes' for each dynamic dimension}}
  %0 = asctile.tensor %arg0(%arg1) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  return %0 : tensor<?x?xf32, #asctile.global>
}

// -----

func.func @dim_index_exceeds_rank(%arg0: memref<*xf32, 22>) -> i32 {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x64xf32, #asctile.global>
  // expected-error@below {{'index' must not exceed the tensor rank}}
  %d = asctile.dim %0, 2 : tensor<32x64xf32, #asctile.global>
  return %d : i32
}

// -----

func.func @load_real_shape_size_mismatch(%arg0: tensor<64x128xf16, #asctile.global>, %arg1: f16, %arg2: i32) -> tensor<64x64xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{real_shape must have same size as tensor shape}}
  %0 = asctile.load %arg0[%c0, %c0], %arg1, (%arg2) : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L1>>
  return %0 : tensor<64x64xf16, #asctile.local<L1>>
}

// -----

func.func @load_real_shape_exceeds_tile(%arg0: tensor<64x128xf16, #asctile.global>, %arg1: f16) -> tensor<64x64xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // expected-error@below {{real_shape exceeds tensor shape}}
  %0 = asctile.load %arg0[%c0, %c0], %arg1, (%c128, %c128) : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L1>>
  return %0 : tensor<64x64xf16, #asctile.local<L1>>
}

// -----

func.func @store_real_shape_size_mismatch(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<?x?xf32, #asctile.global>, %arg2: i32) {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{real_shape must have same size as tensor shape}}
  asctile.store %arg0, %arg1[%c0, %c0], (%arg2) : tensor<16x16xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// -----

func.func @store_real_shape_exceeds_tile(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<?x?xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  // expected-error@below {{real_shape exceeds tensor shape}}
  asctile.store %arg0, %arg1[%c0, %c0], (%c32, %c32) : tensor<16x16xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
  return
}

// -----

func.func @store_fixpipe_type_mismatch(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xi8, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{failed to verify that all of {base, value} have same element type}}
  asctile.store_fixpipe %arg0, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.global>
  return
}

// -----

func.func @store_fixpipe_real_shape_exceeds(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<?x?xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  // expected-error@below {{real_shape exceeds tensor shape}}
  asctile.store_fixpipe %arg0, %arg1[%c0, %c0], (%c32, %c32) : tensor<16x16xf32, #asctile.local<L0C>>, tensor<?x?xf32, #asctile.global>
  return
}

// -----

func.func @copy_fixpipe_type_mismatch(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xi8, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  // expected-error@below {{failed to verify that all of {base, result} have same element type}}
  %0 = asctile.copy_fixpipe %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.local<UB>>
  return %0 : tensor<16x16xi8, #asctile.local<UB>>
}

// -----

func.func @accumulator_bias_not_bt(%arg0: tensor<8xf32, #asctile.local<UB>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  // expected-error@below {{bias must have BT tensor location}}
  %0 = asctile.accumulator %arg0 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8xf32, #asctile.local<UB>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// -----

func.func @accumulator_bias_shape_mismatch(%arg0: tensor<16xf32, #asctile.local<BT>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  // expected-error@below {{bias shape must match result's second dimension}}
  %0 = asctile.accumulator %arg0 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<16xf32, #asctile.local<BT>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// -----

func.func @matmul_wrong_a_location(%arg0: tensor<8x16xf32, #asctile.local<UB>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  // expected-error@below {{matrixA must have L0A tensor location}}
  %0 = asctile.matmul %arg0, %arg1 : tensor<8x16xf32, #asctile.local<UB>>, tensor<16x8xf32, #asctile.local<L0B>> -> tensor<8x8xf32, #asctile.local<L0C>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// -----

func.func @matmul_wrong_b_location(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<UB>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  // expected-error@below {{matrixB must have L0B tensor location}}
  %0 = asctile.matmul %arg0, %arg1 : tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<UB>> -> tensor<8x8xf32, #asctile.local<L0C>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// -----

func.func @matmul_wrong_result_location(%arg0: tensor<8x16xf32, #asctile.local<L0A>>, %arg1: tensor<16x8xf32, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<UB>> {
  // expected-error@below {{result must have L0C tensor location}}
  %0 = asctile.matmul %arg0, %arg1 : tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>> -> tensor<8x8xf32, #asctile.local<UB>>
  return %0 : tensor<8x8xf32, #asctile.local<UB>>
}

// -----

func.func @matmul_acc_wrong_acc_location(%arg0: tensor<8x8xf32, #asctile.local<UB>>, %arg1: tensor<8x16xf32, #asctile.local<L0A>>, %arg2: tensor<16x8xf32, #asctile.local<L0B>>) {
  // expected-error@below {{acc must have L0C tensor location}}
  asctile.matmul_acc %arg0, %arg1, %arg2 : tensor<8x8xf32, #asctile.local<UB>>, tensor<8x16xf32, #asctile.local<L0A>>, tensor<16x8xf32, #asctile.local<L0B>>
  return
}

// -----

func.func @cube_group_operand_count_mismatch(%arg0: tensor<64xf32, #asctile.global>) -> tensor<64xf32, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant 0.0 : f32
  // expected-error@below {{number of operands (1) must match number of block arguments (2)}}
  %0 = asctile.cube_group(%arg0 : tensor<64xf32, #asctile.global>) {
  ^bb0(%a: tensor<64xf32, #asctile.global>, %b: i32):
    %1 = asctile.load %a[%b], %cst : tensor<64xf32, #asctile.global>, tensor<64xf32, #asctile.local<L1>>
    asctile.yield %1 : tensor<64xf32, #asctile.local<L1>>
  } : tensor<64xf32, #asctile.local<L1>>
  return %0 : tensor<64xf32, #asctile.local<L1>>
}

// -----

func.func @vector_group_yield_count_mismatch(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  // expected-error@below {{number of yield operands (2) must match number of results (1)}}
  %0 = asctile.vector_group(%arg0 : tensor<32xf32, #asctile.local<UB>>) {
  ^bb0(%a: tensor<32xf32, #asctile.local<UB>>):
    %1 = asctile.relu %a : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %1, %1 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}
