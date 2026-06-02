// RUN: ascir-opt -asctile-detect-bias-load %s | FileCheck %s

// CHECK-LABEL: func.func @mark_bias_basic(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<128xf16>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]]] {asctile.is_bias} : !asctile.tensor<128xf16>, !asctile.tile<128xf16, L1>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]]] : !asctile.tile<128xf16, L1>, !asctile.tile<128xf16, BT>
// CHECK-NEXT: return %[[COPY]] : !asctile.tile<128xf16, BT>
// CHECK-NEXT:}
func.func @mark_bias_basic(%arg0: !asctile.tensor<128xf16>) -> !asctile.tile<128xf16, BT> {
  %c0_i32 = arith.constant 0 : i32
  %load = asctile.load %arg0[%c0_i32] : !asctile.tensor<128xf16>, !asctile.tile<128xf16, L1>
  %copy = asctile.copy %load[%c0_i32] : !asctile.tile<128xf16, L1>, !asctile.tile<128xf16, BT>
  return %copy : !asctile.tile<128xf16, BT>
}

// CHECK-LABEL: func.func @no_mark_non_bt_copy(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<128xf16>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]]] : !asctile.tensor<128xf16>, !asctile.tile<128xf16, L1>
// CHECK-NEXT: %[[COPY:.*]] = asctile.copy %[[LOAD]][%[[C0]]] : !asctile.tile<128xf16, L1>, !asctile.tile<128xf16, UB>
// CHECK-NEXT: return %[[COPY]] : !asctile.tile<128xf16, UB>
// CHECK-NEXT:}
func.func @no_mark_non_bt_copy(%arg0: !asctile.tensor<128xf16>) -> !asctile.tile<128xf16, UB> {
  %c0_i32 = arith.constant 0 : i32
  %load = asctile.load %arg0[%c0_i32] : !asctile.tensor<128xf16>, !asctile.tile<128xf16, L1>
  %copy = asctile.copy %load[%c0_i32] : !asctile.tile<128xf16, L1>, !asctile.tile<128xf16, UB>
  return %copy : !asctile.tile<128xf16, UB>
}

// CHECK-LABEL: func.func @mark_bias_with_other_use(
// CHECK-SAME: %[[TENSOR:.*]]: !asctile.tensor<128xf16>, %[[TENSOR2:.*]]: !asctile.tensor<128xf16>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LOAD:.*]] = asctile.load %[[TENSOR]][%[[C0]]] {asctile.is_bias} : !asctile.tensor<128xf16>, !asctile.tile<128xf16, L1>
// CHECK-NEXT: %[[COPY_BT:.*]] = asctile.copy %[[LOAD]][%[[C0]]] : !asctile.tile<128xf16, L1>, !asctile.tile<128xf16, BT>
// CHECK-NEXT: asctile.store %[[LOAD]], %[[TENSOR2]][%[[C0]]] : !asctile.tile<128xf16, L1>, !asctile.tensor<128xf16>
// CHECK-NEXT: return %[[COPY_BT]] : !asctile.tile<128xf16, BT>
// CHECK-NEXT:}
func.func @mark_bias_with_other_use(%arg0: !asctile.tensor<128xf16>, %arg1: !asctile.tensor<128xf16>) -> !asctile.tile<128xf16, BT> {
  %c0_i32 = arith.constant 0 : i32
  %load = asctile.load %arg0[%c0_i32] : !asctile.tensor<128xf16>, !asctile.tile<128xf16, L1>
  %copy_bt = asctile.copy %load[%c0_i32] : !asctile.tile<128xf16, L1>, !asctile.tile<128xf16, BT>
  asctile.store %load, %arg1[%c0_i32] : !asctile.tile<128xf16, L1>, !asctile.tensor<128xf16>
  return %copy_bt : !asctile.tile<128xf16, BT>
}
