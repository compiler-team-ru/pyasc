// RUN: ascir-opt -asctile-wrap-cv-groups %s | FileCheck %s

// CHECK-LABEL: func.func @cube_load_and_copy(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<128xf16, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[BLOCK0:.*]] = asctile.cube_group(%[[TENSOR]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<128xf16, #asctile.global>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[LOAD:.*]] = asctile.load %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[LOAD]]
// CHECK: } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[BLOCK1:.*]] = asctile.cube_group(%[[BLOCK0]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<128xf16, #asctile.local<L1>>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[COPY:.*]] = asctile.copy %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[COPY]]
// CHECK: } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: return %[[BLOCK1]]
func.func @cube_load_and_copy(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L0A>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %1 = asctile.copy %0[%c0] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
  return %1 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @vector_ops(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[BLOCK0:.*]] = asctile.vector_group(%[[TENSOR]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.global>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[LOAD:.*]] = asctile.load %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[LOAD]]
// CHECK: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: %[[BLOCK1:.*]] = asctile.vector_group(%[[BLOCK0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.local<UB>>):
// CHECK-NEXT:   %[[RELU:.*]] = asctile.relu %[[ARG1]]
// CHECK-NEXT:   asctile.yield %[[RELU]]
// CHECK: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: asctile.vector_group(%[[BLOCK1]], %[[TENSOR]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.local<UB>>, %[[ARG2:.*]]: tensor<32xf32, #asctile.global>, %[[ARG3:.*]]: i32):
// CHECK-NEXT:   asctile.store %[[ARG1]], %[[ARG2]][%[[ARG3]]]
// CHECK: return
func.func @vector_ops(%arg0: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  %1 = asctile.relu %0 : tensor<32xf32, #asctile.local<UB>>
  asctile.store %1, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @mixed_cube_and_vector(
// CHECK-SAME: %[[T1:.*]]: tensor<128xf16, #asctile.global>, %[[T2:.*]]: tensor<32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[BLOCK0:.*]] = asctile.cube_group(%[[T1]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<128xf16, #asctile.global>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[LOAD:.*]] = asctile.load %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[LOAD]]
// CHECK: } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: %[[BLOCK1:.*]] = asctile.cube_group(%[[BLOCK0]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<128xf16, #asctile.local<L1>>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[COPY:.*]] = asctile.copy %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[COPY]]
// CHECK: } : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: %[[BLOCK2:.*]] = asctile.vector_group(%[[T2]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.global>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[LOAD2:.*]] = asctile.load %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[LOAD2]]
// CHECK: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: %[[BLOCK3:.*]] = asctile.vector_group(%[[BLOCK2]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.local<UB>>):
// CHECK-NEXT:   %[[RELU:.*]] = asctile.relu %[[ARG1]]
// CHECK-NEXT:   asctile.yield %[[RELU]]
// CHECK: return %[[BLOCK1]]
func.func @mixed_cube_and_vector(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) -> tensor<128xf16, #asctile.local<L0A>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
  %1 = asctile.copy %0[%c0] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
  %2 = asctile.load %arg1[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  %3 = asctile.relu %2 : tensor<32xf32, #asctile.local<UB>>
  return %1 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @no_wrap_scalar_ops(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[C0]], %[[C1]] : i32
// CHECK-NEXT: %[[BLOCK:.*]] = asctile.vector_group(%[[TENSOR]], %[[SUM]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.global>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[LOAD:.*]] = asctile.load %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[LOAD]]
// CHECK: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %[[BLOCK]]
func.func @no_wrap_scalar_ops(%arg0: tensor<32xf32, #asctile.global>) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %c0, %c1 : i32
  %0 = asctile.load %arg0[%sum] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @matmul_cube_group(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<128xf16, #asctile.global>)
// CHECK: asctile.cube_group
// CHECK:   asctile.accumulator
// CHECK: asctile.cube_group
// CHECK:   asctile.load
// CHECK: asctile.cube_group
// CHECK:   asctile.copy
// CHECK: asctile.cube_group
// CHECK:   asctile.load
// CHECK: asctile.cube_group
// CHECK:   asctile.copy
// CHECK: asctile.cube_group
// CHECK:   asctile.matmul
func.func @matmul_cube_group(%arg0: tensor<128xf16, #asctile.global>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  %c0 = arith.constant 0 : i32
  %acc = asctile.accumulator : tensor<16x16xf32, #asctile.local<L0C>>
  %la = asctile.load %arg0[%c0, %c0] : tensor<128xf16, #asctile.global>, tensor<16x16xf16, #asctile.local<L1>>
  %ca = asctile.copy %la[%c0, %c0] : tensor<16x16xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0A>>
  %lb = asctile.load %arg0[%c0, %c0] : tensor<128xf16, #asctile.global>, tensor<16x16xf16, #asctile.local<L1>>
  %cb = asctile.copy %lb[%c0, %c0] : tensor<16x16xf16, #asctile.local<L1>>, tensor<16x16xf16, #asctile.local<L0B>>
  %res = asctile.matmul %ca, %cb : tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>> -> tensor<16x16xf32, #asctile.local<L0C>>
  return %res : tensor<16x16xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @vector_arith_in_block(
// CHECK-SAME: %[[T1:.*]]: tensor<32xf32, #asctile.local<UB>>, %[[T2:.*]]: tensor<32xf32, #asctile.local<UB>>)
// CHECK-NEXT: %[[BLOCK:.*]] = asctile.vector_group(%[[T1]], %[[T2]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.local<UB>>, %[[ARG2:.*]]: tensor<32xf32, #asctile.local<UB>>):
// CHECK-NEXT:   %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG2]]
// CHECK-NEXT:   asctile.yield %[[ADD]]
// CHECK: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: return %[[BLOCK]]
func.func @vector_arith_in_block(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @store_not_wrapped(
// CHECK-SAME: %[[TENSOR:.*]]: tensor<32xf32, #asctile.global>)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[BLOCK:.*]] = asctile.vector_group(%[[TENSOR]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.global>, %[[ARG2:.*]]: i32):
// CHECK-NEXT:   %[[LOAD:.*]] = asctile.load %[[ARG1]][%[[ARG2]]]
// CHECK-NEXT:   asctile.yield %[[LOAD]]
// CHECK: } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT: asctile.vector_group(%[[BLOCK]], %[[TENSOR]], %[[C0]]
// CHECK-NEXT: ^bb0(%[[ARG1:.*]]: tensor<32xf32, #asctile.local<UB>>, %[[ARG2:.*]]: tensor<32xf32, #asctile.global>, %[[ARG3:.*]]: i32):
// CHECK-NEXT:   asctile.store %[[ARG1]], %[[ARG2]][%[[ARG3]]]
// CHECK: return
func.func @store_not_wrapped(%arg0: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %0 = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  asctile.store %0, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  return
}
