// RUN: ascir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @erase_unused_accumulator() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @erase_unused_accumulator() {
  %0 = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
  return
}

// CHECK-LABEL: func.func @erase_unused_copy(%arg0: tensor<64x128xf16, #asctile.local<L1>>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @erase_unused_copy(%arg0: tensor<64x128xf16, #asctile.local<L1>>) {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] : tensor<64x128xf16, #asctile.local<L1>>, tensor<64x64xf16, #asctile.local<L0A>>
  return
}

// CHECK-LABEL: func.func @erase_unused_load(%arg0: tensor<64x128xf16, #asctile.global>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @erase_unused_load(%arg0: tensor<64x128xf16, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %cst = arith.constant 0.0 : f16
  %0 = asctile.load %arg0[%c0, %c0], %cst : tensor<64x128xf16, #asctile.global>, tensor<64x64xf16, #asctile.local<L1>>
  return
}

// CHECK-LABEL: func.func @erase_unused_copy_fixpipe(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @erase_unused_copy_fixpipe(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy_fixpipe %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
  return
}

// CHECK-LABEL: func.func @keep_used_accumulator() -> tensor<8x8xf32, #asctile.local<L0C>> {
// CHECK-NEXT: %0 = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT: return %0 : tensor<8x8xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @keep_used_accumulator() -> tensor<8x8xf32, #asctile.local<L0C>> {
  %0 = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
  return %0 : tensor<8x8xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @fold_identity_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_identity_cast(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_double_cast_back(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_double_cast_back(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = asctile.cast <default> %arg0 : tensor<32xf32, #asctile.local<UB>> to tensor<32xi32, #asctile.local<UB>>
  %1 = asctile.cast <default> %0 : tensor<32xi32, #asctile.local<UB>> to tensor<32xf32, #asctile.local<UB>>
  return %1 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_identity_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_identity_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.reshape %arg0 : tensor<16x16xf32, #asctile.local<UB>> to tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_chain_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<4x64xf32, #asctile.local<UB>> {
// CHECK-NEXT: %0 = asctile.reshape %arg0 : tensor<16x16xf32, #asctile.local<UB>> to tensor<4x64xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<4x64xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_chain_reshape(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<4x64xf32, #asctile.local<UB>> {
  %0 = asctile.reshape %arg0 : tensor<16x16xf32, #asctile.local<UB>> to tensor<8x32xf32, #asctile.local<UB>>
  %1 = asctile.reshape %0 : tensor<8x32xf32, #asctile.local<UB>> to tensor<4x64xf32, #asctile.local<UB>>
  return %1 : tensor<4x64xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_reshape_back(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_reshape_back(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %0 = asctile.reshape %arg0 : tensor<16x16xf32, #asctile.local<UB>> to tensor<8x32xf32, #asctile.local<UB>>
  %1 = asctile.reshape %0 : tensor<8x32xf32, #asctile.local<UB>> to tensor<16x16xf32, #asctile.local<UB>>
  return %1 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_single_concat(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT: return %arg0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_single_concat(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.concat %arg0 : tensor<16xf32, #asctile.local<UB>> -> tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_splat_constant() -> tensor<4xf32, #asctile.local<UB>> {
// CHECK-NEXT: %cst = arith.constant dense<1.000000e+00> : tensor<4xf32, #asctile.local<UB>>
// CHECK-NEXT: return %cst : tensor<4xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_splat_constant() -> tensor<4xf32, #asctile.local<UB>> {
  %cst = arith.constant 1.0 : f32
  %0 = asctile.splat %cst : tensor<4xf32, #asctile.local<UB>>
  return %0 : tensor<4xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fold_dim_static(%arg0: memref<*xf32, 22>) -> i32 {
// CHECK-NEXT: %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT: return %c32_i32 : i32
// CHECK-NEXT:}
func.func @fold_dim_static(%arg0: memref<*xf32, 22>) -> i32 {
  %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<32x64xf32, #asctile.global>
  %d = asctile.dim %0, 0 : tensor<32x64xf32, #asctile.global>
  return %d : i32
}

// CHECK-LABEL: func.func @fold_dim_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> i32 {
// CHECK-NEXT: return %arg2 : i32
// CHECK-NEXT:}
func.func @fold_dim_dynamic(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32) -> i32 {
  %0 = asctile.tensor %arg0(%arg1, %arg2) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
  %d = asctile.dim %0, 1 : tensor<?x?xf32, #asctile.global>
  return %d : i32
}

// CHECK-LABEL: func.func @fold_broadcast_constant() -> tensor<4xf32, #asctile.local<UB>> {
// CHECK-NEXT: %cst = arith.constant dense<2.000000e+00> : tensor<4xf32, #asctile.local<UB>>
// CHECK-NEXT: return %cst : tensor<4xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_broadcast_constant() -> tensor<4xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<2.0> : tensor<1xf32, #asctile.local<UB>>
  %0 = asctile.broadcast %cst : tensor<1xf32, #asctile.local<UB>> to tensor<4xf32, #asctile.local<UB>>
  return %0 : tensor<4xf32, #asctile.local<UB>>
}
