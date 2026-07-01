// RUN: ascir-opt -asctile-cube-transpose-to-load %s | FileCheck %s

// CHECK-LABEL: func.func @transpose_l0a_basic(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a} : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a} : tensor<32x128xf16, #asctile.local<L1>>, tensor<128x32xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY]] : tensor<128x32xf16, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @transpose_l0a_basic(%arg0: tensor<32x128xf16, #asctile.global>) -> tensor<128x32xf16, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<L1>>, tensor<32x128xf16, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<L0A>> to tensor<128x32xf16, #asctile.local<L0A>>
  return %transpose : tensor<128x32xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @transpose_l0b_basic(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b} : tensor<64x32xf16, #asctile.global>, tensor<64x32xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b} : tensor<64x32xf16, #asctile.local<L1>>, tensor<32x64xf16, #asctile.local<L0B>>
// CHECK-NEXT: return %[[COPY]] : tensor<32x64xf16, #asctile.local<L0B>>
// CHECK-NEXT:}
func.func @transpose_l0b_basic(%arg0: tensor<64x32xf16, #asctile.global>) -> tensor<32x64xf16, #asctile.local<L0B>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<64x32xf16, #asctile.global>, tensor<64x32xf16, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<64x32xf16, #asctile.local<L1>>, tensor<64x32xf16, #asctile.local<L0B>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<64x32xf16, #asctile.local<L0B>> to tensor<32x64xf16, #asctile.local<L0B>>
  return %transpose : tensor<32x64xf16, #asctile.local<L0B>>
}

// CHECK-LABEL: func.func @transpose_ub_not_transformed(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32x128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<UB>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : tensor<32x128xf16, #asctile.local<UB>>, tensor<32x128xf16, #asctile.local<UB>>
// CHECK-NEXT: %[[TRANSPOSE:.*]] = asctile.transpose %[[COPY]], [1 : i32, 0 : i32] : tensor<32x128xf16, #asctile.local<UB>> to tensor<128x32xf16, #asctile.local<UB>>
// CHECK-NEXT: return %[[TRANSPOSE]] : tensor<128x32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @transpose_ub_not_transformed(%arg0: tensor<32x128xf16, #asctile.global>) -> tensor<128x32xf16, #asctile.local<UB>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<32x128xf16, #asctile.global>, tensor<32x128xf16, #asctile.local<UB>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<32x128xf16, #asctile.local<UB>>, tensor<32x128xf16, #asctile.local<UB>>
  %transpose = asctile.transpose %copy, [1: i32, 0 : i32] : tensor<32x128xf16, #asctile.local<UB>> to tensor<128x32xf16, #asctile.local<UB>>
  return %transpose : tensor<128x32xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @transpose_l0a_f32(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<16x32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a} : tensor<16x32xf32, #asctile.global>, tensor<16x32xf32, #asctile.local<L1>>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a} : tensor<16x32xf32, #asctile.local<L1>>, tensor<32x16xf32, #asctile.local<L0A>>
// CHECK-NEXT: return %[[COPY]] : tensor<32x16xf32, #asctile.local<L0A>>
// CHECK-NEXT:}
func.func @transpose_l0a_f32(%arg0: tensor<16x32xf32, #asctile.global>) -> tensor<32x16xf32, #asctile.local<L0A>> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : tensor<16x32xf32, #asctile.global>, tensor<16x32xf32, #asctile.local<L1>>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : tensor<16x32xf32, #asctile.local<L1>>, tensor<16x32xf32, #asctile.local<L0A>>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<L0A>> to tensor<32x16xf32, #asctile.local<L0A>>
  return %transpose : tensor<32x16xf32, #asctile.local<L0A>>
}
