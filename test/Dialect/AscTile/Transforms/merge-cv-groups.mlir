// RUN: ascir-opt -asctile-merge-cv-groups %s | FileCheck %s

// CHECK-LABEL: func.func @merge_cube_groups(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L0A>> {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0:2 = asctile.cube_group {
// CHECK-NEXT:     %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     %2 = asctile.copy %1[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:     %3 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     asctile.yield %1, %2 : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:   } : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:   return %0#1 : tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT: }
func.func @merge_cube_groups(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L0A>> {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  %r2 = asctile.cube_group(%r1 : tensor<128xf16, #asctile.local<L1>>) {
    %cp = asctile.copy %r1[%c0] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
    asctile.yield %cp : tensor<128xf16, #asctile.local<L0A>>
  } : tensor<128xf16, #asctile.local<L0A>>
  %r3 = asctile.cube_group {
    %ld2 = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld2 : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %r2 : tensor<128xf16, #asctile.local<L0A>>
}

// CHECK-LABEL: func.func @merge_vector_groups(%arg0: tensor<32xf32, #asctile.global>) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0:2 = asctile.vector_group {
// CHECK-NEXT:     %1 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:     %2 = asctile.relu %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:     asctile.store %2, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:     asctile.yield %1, %2 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   } : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @merge_vector_groups(%arg0: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.vector_group {
    %ld = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.yield %ld : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  %r2 = asctile.vector_group(%r1 : tensor<32xf32, #asctile.local<UB>>) {
    %rv = asctile.relu %r1 : tensor<32xf32, #asctile.local<UB>>
    asctile.yield %rv : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  asctile.vector_group(%r2 : tensor<32xf32, #asctile.local<UB>>) {
    asctile.store %r2, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
    asctile.yield
  }
  return
}

// CHECK-LABEL: func.func @no_merge_different_types(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = asctile.cube_group {
// CHECK-NEXT:     %2 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     asctile.yield %2 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   %1 = asctile.vector_group {
// CHECK-NEXT:     %2 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:     asctile.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:   return %0 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: }
func.func @no_merge_different_types(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  %r2 = asctile.vector_group {
    %ld2 = asctile.load %arg1[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.yield %ld2 : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  return %r1 : tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @barrier_breaks_run(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = asctile.cube_group {
// CHECK-NEXT:     %3 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     asctile.yield %3 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %1 = arith.addi %c0_i32, %c1_i32 : i32
// CHECK-NEXT:   %2 = asctile.cube_group {
// CHECK-NEXT:     %3 = asctile.load %arg0[%1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     asctile.yield %3 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   return %0 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: }
func.func @barrier_breaks_run(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %c0, %c1 : i32
  %r2 = asctile.cube_group {
    %ld2 = asctile.load %arg0[%sum] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld2 : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %r1 : tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @single_group_not_merged(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = asctile.cube_group {
// CHECK-NEXT:     %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     asctile.yield %1 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   return %0 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT: }
func.func @single_group_not_merged(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %r1 : tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @nested_for(%arg0: tensor<128xf16, #asctile.global>) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c10 = arith.constant 10 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   scf.for %arg1 = %c0 to %c10 step %c1 {
// CHECK-NEXT:     %0 = asctile.cube_group {
// CHECK-NEXT:       %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:       %2 = asctile.copy %1[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:       asctile.yield %1 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @nested_for(%arg0: tensor<128xf16, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %offset = arith.constant 0 : i32
  scf.for %i = %c0 to %c10 step %c1 {
    %r1 = asctile.cube_group {
      %ld = asctile.load %arg0[%offset] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
      asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
    } : tensor<128xf16, #asctile.local<L1>>
    %r2 = asctile.cube_group(%r1 : tensor<128xf16, #asctile.local<L1>>) {
      %cp = asctile.copy %r1[%offset] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
      asctile.yield %cp : tensor<128xf16, #asctile.local<L0A>>
    } : tensor<128xf16, #asctile.local<L0A>>
  }
  return
}

// CHECK-LABEL: func.func @nested_if(%arg0: tensor<128xf16, #asctile.global>, %arg1: i1) {
// CHECK-NEXT:   scf.if %arg1 {
// CHECK-NEXT:     %0 = asctile.cube_group {
// CHECK-NEXT:       %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:       %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:       %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT:       %2 = asctile.copy %1[%c0_i32_0] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:       asctile.yield %1 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:     } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @nested_if(%arg0: tensor<128xf16, #asctile.global>, %cond: i1) {
  scf.if %cond {
    %r1 = asctile.cube_group {
      %c0 = arith.constant 0 : i32
      %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
      asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
    } : tensor<128xf16, #asctile.local<L1>>
    %r2 = asctile.cube_group(%r1 : tensor<128xf16, #asctile.local<L1>>) {
      %c0 = arith.constant 0 : i32
      %cp = asctile.copy %r1[%c0] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
      asctile.yield %cp : tensor<128xf16, #asctile.local<L0A>>
    } : tensor<128xf16, #asctile.local<L0A>>
  }
  return
}
