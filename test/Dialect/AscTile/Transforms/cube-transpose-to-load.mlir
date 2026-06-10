// RUN: ascir-opt -asctile-cube-transpose-to-load %s | FileCheck %s

// CHECK-LABEL: func.func @transpose_l0a_basic(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<32x128xf16>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a} : !asctile.tensor<32x128xf16>, !asctile.tile<32x128xf16, L1>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a} : !asctile.tile<32x128xf16, L1>, !asctile.tile<128x32xf16, L0A>
// CHECK-NEXT: return %[[COPY]] : !asctile.tile<128x32xf16, L0A>
// CHECK-NEXT:}
func.func @transpose_l0a_basic(%arg0: !asctile.tensor<32x128xf16>) -> !asctile.tile<128x32xf16, L0A> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : !asctile.tensor<32x128xf16>, !asctile.tile<32x128xf16, L1>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : !asctile.tile<32x128xf16, L1>, !asctile.tile<32x128xf16, L0A>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : !asctile.tile<32x128xf16, L0A> to !asctile.tile<128x32xf16, L0A>
  return %transpose : !asctile.tile<128x32xf16, L0A>
}

// CHECK-LABEL: func.func @transpose_l0b_basic(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<64x32xf16>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_b} : !asctile.tensor<64x32xf16>, !asctile.tile<64x32xf16, L1>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_b} : !asctile.tile<64x32xf16, L1>, !asctile.tile<32x64xf16, L0B>
// CHECK-NEXT: return %[[COPY]] : !asctile.tile<32x64xf16, L0B>
// CHECK-NEXT:}
func.func @transpose_l0b_basic(%arg0: !asctile.tensor<64x32xf16>) -> !asctile.tile<32x64xf16, L0B> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : !asctile.tensor<64x32xf16>, !asctile.tile<64x32xf16, L1>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : !asctile.tile<64x32xf16, L1>, !asctile.tile<64x32xf16, L0B>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : !asctile.tile<64x32xf16, L0B> to !asctile.tile<32x64xf16, L0B>
  return %transpose : !asctile.tile<32x64xf16, L0B>
}

// CHECK-LABEL: func.func @transpose_ub_not_transformed(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<32x128xf16>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] : !asctile.tensor<32x128xf16>, !asctile.tile<32x128xf16, UB>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] : !asctile.tile<32x128xf16, UB>, !asctile.tile<32x128xf16, UB>
// CHECK-NEXT: %[[TRANSPOSE:.*]] = asctile.transpose %[[COPY]], [1 : i32, 0 : i32] : !asctile.tile<32x128xf16, UB> to !asctile.tile<128x32xf16, UB>
// CHECK-NEXT: return %[[TRANSPOSE]] : !asctile.tile<128x32xf16, UB>
// CHECK-NEXT:}
func.func @transpose_ub_not_transformed(%arg0: !asctile.tensor<32x128xf16>) -> !asctile.tile<128x32xf16, UB> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f16
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : !asctile.tensor<32x128xf16>, !asctile.tile<32x128xf16, UB>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : !asctile.tile<32x128xf16, UB>, !asctile.tile<32x128xf16, UB>
  %transpose = asctile.transpose %copy, [1: i32, 0 : i32] : !asctile.tile<32x128xf16, UB> to !asctile.tile<128x32xf16, UB>
  return %transpose : !asctile.tile<128x32xf16, UB>
}

// CHECK-LABEL: func.func @transpose_l0a_f32(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<16x32xf32>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]], %[[C0]]], %[[PAD]] {asctile.transpose_a} : !asctile.tensor<16x32xf32>, !asctile.tile<16x32xf32, L1>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]], %[[C0]]] {asctile.transpose_a} : !asctile.tile<16x32xf32, L1>, !asctile.tile<32x16xf32, L0A>
// CHECK-NEXT: return %[[COPY]] : !asctile.tile<32x16xf32, L0A>
// CHECK-NEXT:}
func.func @transpose_l0a_f32(%arg0: !asctile.tensor<16x32xf32>) -> !asctile.tile<32x16xf32, L0A> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %load = asctile.load %arg0[%c0_i32, %c0_i32], %cst : !asctile.tensor<16x32xf32>, !asctile.tile<16x32xf32, L1>
  %copy = asctile.copy %load[%c0_i32, %c0_i32] : !asctile.tile<16x32xf32, L1>, !asctile.tile<16x32xf32, L0A>
  %transpose = asctile.transpose %copy, [1 : i32, 0 : i32] : !asctile.tile<16x32xf32, L0A> to !asctile.tile<32x16xf32, L0A>
  return %transpose : !asctile.tile<32x16xf32, L0A>
}
