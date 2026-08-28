// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-merge-cv-groups %s | FileCheck %s

// CHECK-LABEL: func.func @merge_cube_groups(%arg0: tensor<128xf16, #asctile.global>) -> (tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>) {
// CHECK:         %0:3 = asctile.cube_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      %2 = asctile.copy %1[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:      %3 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// TODO: Yield only %2 and %3
// CHECK-NEXT:      asctile.yield %1, %2, %3 : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    return %0#1, %0#2 : tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  }
func.func @merge_cube_groups(%arg0: tensor<128xf16, #asctile.global>) -> (tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>) {
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
  return %r2, %r3 : tensor<128xf16, #asctile.local<L0A>>, tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @merge_vector_groups(%arg0: tensor<32xf32, #asctile.global>) {
// CHECK:         %0:2 = asctile.vector_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      %2 = asctile.relu %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %2, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:      asctile.yield %1, %2 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    } : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
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
// CHECK:         %0 = asctile.cube_group {
// CHECK-NEXT:      %2 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %2 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    %1 = asctile.vector_group {
// CHECK-NEXT:      %2 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    return %0 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  }
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

// CHECK-LABEL: func.func @absorb_enables_merge(%arg0: tensor<128xf16, #asctile.global>, %arg1: i32) -> (tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>) {
// CHECK:         %0:2 = asctile.cube_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      %2 = arith.addi %arg1, %c1_i32 : i32
// CHECK-NEXT:      %3 = asctile.load %arg0[%2] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %1, %3 : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    return %0#0, %0#1 : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  }
func.func @absorb_enables_merge(%arg0: tensor<128xf16, #asctile.global>, %arg1: i32) -> (tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>) {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %arg1, %c1 : i32
  %r2 = asctile.cube_group {
    %ld2 = asctile.load %arg0[%sum] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld2 : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %r1, %r2 : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @single_group_not_merged(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
// CHECK:         %0 = asctile.cube_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %1 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    return %0 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:  }
func.func @single_group_not_merged(%arg0: tensor<128xf16, #asctile.global>) -> tensor<128xf16, #asctile.local<L1>> {
  %c0 = arith.constant 0 : i32
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %r1 : tensor<128xf16, #asctile.local<L1>>
}

// CHECK-LABEL: func.func @absorb_pure_op_before_group(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32) {
// CHECK:         asctile.vector_group {
// CHECK-NEXT:      %0 = arith.addi %arg1, %c1_i32 : i32
// CHECK-NEXT:      %1 = asctile.load %arg0[%0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %1, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @absorb_pure_op_before_group(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %arg1, %c1 : i32
  asctile.vector_group {
    %ld = asctile.load %arg0[%sum] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
    asctile.yield
  }
  return
}

// CHECK-LABEL: func.func @no_absorb_op_with_external_use(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32) -> i32 {
// CHECK:         %0 = arith.addi %arg1, %c1_i32 : i32
// CHECK-NEXT:    asctile.vector_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %1, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
func.func @no_absorb_op_with_external_use(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %arg1, %c1 : i32
  asctile.vector_group {
    %ld = asctile.load %arg0[%sum] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
    asctile.yield
  }
  return %sum : i32
}

// CHECK-LABEL: func.func @barrier_breaks_merge(%arg0: tensor<128xf16, #asctile.global>, %arg1: i32) -> (tensor<128xf16, #asctile.local<L1>>, i32) {
// CHECK-NEXT:    %0 = asctile.cube_group {
// CHECK-NEXT:      %3 = asctile.load %arg0[%arg1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %3 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    %1 = arith.addi %arg1, %arg1 : i32
// CHECK-NEXT:    %2 = asctile.cube_group {
// CHECK-NEXT:      %3 = asctile.load %arg0[%arg1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %3 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    return %0, %1 : tensor<128xf16, #asctile.local<L1>>, i32
// CHECK-NEXT:  }
func.func @barrier_breaks_merge(%arg0: tensor<128xf16, #asctile.global>, %arg1: i32) -> (tensor<128xf16, #asctile.local<L1>>, i32) {
  %r1 = asctile.cube_group {
    %ld = asctile.load %arg0[%arg1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  %sum = arith.addi %arg1, %arg1 : i32
  %r2 = asctile.cube_group {
    %ld2 = asctile.load %arg0[%arg1] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld2 : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return %r1, %sum : tensor<128xf16, #asctile.local<L1>>, i32
}

// CHECK-LABEL: func.func @interchange_with_for(%arg0: tensor<128xf16, #asctile.global>) {
// CHECK:         asctile.cube_group {
// CHECK-NEXT:      scf.for %arg1 = %c0 to %c10 step %c1 {
// CHECK-NEXT:        %0 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:        %1 = asctile.copy %0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:      } {asctile.unroll_factor = 2 : index}
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @interchange_with_for(%arg0: tensor<128xf16, #asctile.global>) {
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
  } {asctile.unroll_factor = 2 : index}
  return
}

// CHECK-LABEL: func.func @interchange_with_if(%arg0: tensor<128xf16, #asctile.global>, %arg1: i1) {
// CHECK:         asctile.cube_group {
// CHECK-NEXT:      scf.if %arg1 {
// CHECK-NEXT:        %0 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:        %1 = asctile.copy %0[%c0_i32] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @interchange_with_if(%arg0: tensor<128xf16, #asctile.global>, %arg1: i1) {
  %c0 = arith.constant 0 : i32
  scf.if %arg1 {
    %r1 = asctile.cube_group {
      %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
      asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
    } : tensor<128xf16, #asctile.local<L1>>
    %c0_1 = arith.constant 0 : i32
    %r2 = asctile.cube_group(%r1 : tensor<128xf16, #asctile.local<L1>>) {
      %cp = asctile.copy %r1[%c0_1] : tensor<128xf16, #asctile.local<L1>>, tensor<128xf16, #asctile.local<L0A>>
      asctile.yield %cp : tensor<128xf16, #asctile.local<L0A>>
    } : tensor<128xf16, #asctile.local<L0A>>
  }
  return
}

// CHECK-LABEL: func.func @no_interchange_mixed_groups_in_for(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) {
// CHECK:         scf.for %arg2 = %c0 to %c10 step %c1 {
// CHECK-NEXT:      %0 = asctile.cube_group {
// CHECK-NEXT:        %2 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:        asctile.yield %2 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      %1 = asctile.vector_group {
// CHECK-NEXT:        %2 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:        asctile.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_interchange_mixed_groups_in_for(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  scf.for %i = %c0 to %c10 step %c1 {
    %r1 = asctile.cube_group {
      %ld = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
      asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
    } : tensor<128xf16, #asctile.local<L1>>
    %r2 = asctile.vector_group {
      %ld2 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
      asctile.yield %ld2 : tensor<32xf32, #asctile.local<UB>>
    } : tensor<32xf32, #asctile.local<UB>>
  }
  return
}

// CHECK-LABEL: func.func @interchange_for_iter_args(%arg0: tensor<32xf32, #asctile.global>) {
// CHECK:         %0:2 = asctile.vector_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      %2 = scf.for %arg1 = %c0 to %c14 step %c1 iter_args(%arg2 = %1) -> (tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:        %3 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:        %4 = arith.addf %arg2, %3 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:        scf.yield %4 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      }
// CHECK-NEXT:      asctile.store %2, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:      asctile.yield %1, %2 : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    } : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @interchange_for_iter_args(%arg0: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c14 = arith.constant 14 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %init = asctile.vector_group {
    %ld = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.yield %ld : tensor<32xf32, #asctile.local<UB>>
  } : tensor<32xf32, #asctile.local<UB>>
  %res = scf.for %i = %c0 to %c14 step %c1 iter_args(%acc = %init) -> (tensor<32xf32, #asctile.local<UB>>) {
    %body = asctile.vector_group {
      %ld = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
      %add = arith.addf %acc, %ld : tensor<32xf32, #asctile.local<UB>>
      asctile.yield %add : tensor<32xf32, #asctile.local<UB>>
    } : tensor<32xf32, #asctile.local<UB>>
    scf.yield %body : tensor<32xf32, #asctile.local<UB>>
  }
  asctile.vector_group {
    asctile.store %res, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
    asctile.yield
  }
  return
}

// CHECK-LABEL: func.func @no_absorb_sync_all_soft_before_vector_group(%arg0: tensor<32xf32, #asctile.global>, %arg1: !ascendc.global_tensor<*xui8>, %arg2: !ascendc.local_tensor<*xui8>, %arg3: i32) {
// CHECK:         ascendc.sync_all_soft %arg1, %arg2, %arg3 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
// CHECK-NEXT:    asctile.vector_group {
// CHECK-NEXT:      %0 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %0, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_absorb_sync_all_soft_before_vector_group(%arg0: tensor<32xf32, #asctile.global>, %arg1: !ascendc.global_tensor<*xui8>, %arg2: !ascendc.local_tensor<*xui8>, %arg3: i32) {
  ascendc.sync_all_soft %arg1, %arg2, %arg3 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
  asctile.vector_group {
    %c0 = arith.constant 0 : i32
    %ld = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @no_absorb_sync_all_hard_before_cube_group(%arg0: tensor<128xf16, #asctile.global>, %arg1: !ascendc.global_tensor<*xui8>, %arg2: !ascendc.local_tensor<*xui8>, %arg3: i32) {
// CHECK:         ascendc.sync_all_soft %arg1, %arg2, %arg3 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
// CHECK-NEXT:    %0 = asctile.cube_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %1 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_absorb_sync_all_hard_before_cube_group(%arg0: tensor<128xf16, #asctile.global>, %arg1: !ascendc.global_tensor<*xui8>, %arg2: !ascendc.local_tensor<*xui8>, %arg3: i32) {
  ascendc.sync_all_soft %arg1, %arg2, %arg3 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
  %0 = asctile.cube_group {
    %c0 = arith.constant 0 : i32
    %ld = asctile.load %arg0[%c0] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  return
}

// CHECK-LABEL: func.func @no_absorb_if_with_barrier(%arg0: tensor<32xf32, #asctile.global>, %arg1: i1, %arg2: !ascendc.global_tensor<*xui8>, %arg3: !ascendc.local_tensor<*xui8>, %arg4: i32) {
// CHECK:         scf.if %arg1 {
// CHECK-NEXT:      ascendc.sync_all_soft %arg2, %arg3, %arg4 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    asctile.vector_group {
// CHECK-NEXT:      %0 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %0, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_absorb_if_with_barrier(%arg0: tensor<32xf32, #asctile.global>, %arg1: i1, %arg2: !ascendc.global_tensor<*xui8>, %arg3: !ascendc.local_tensor<*xui8>, %arg4: i32) {
  scf.if %arg1 {
    ascendc.sync_all_soft %arg2, %arg3, %arg4 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
  }
  asctile.vector_group {
    %c0 = arith.constant 0 : i32
    %ld = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @interchange_if_with_else(%arg0: tensor<32xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>, %arg2: i1) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:         %0 = asctile.vector_group {
// CHECK-NEXT:      %1 = scf.if %arg2 -> (tensor<32xf32, #asctile.local<UB>>) {
// CHECK-NEXT:        %2 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:        scf.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %2 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:        scf.yield %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      }
// CHECK-NEXT:      asctile.yield %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    } : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:    return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  }
func.func @interchange_if_with_else(%arg0: tensor<32xf32, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>, %arg2: i1) -> tensor<32xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %r = scf.if %arg2 -> (tensor<32xf32, #asctile.local<UB>>) {
    %a = asctile.vector_group {
      %ld = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
      asctile.yield %ld : tensor<32xf32, #asctile.local<UB>>
    } : tensor<32xf32, #asctile.local<UB>>
    scf.yield %a : tensor<32xf32, #asctile.local<UB>>
  } else {
    %b = asctile.vector_group {
      %ld = asctile.load %arg1[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
      asctile.yield %ld : tensor<32xf32, #asctile.local<UB>>
    } : tensor<32xf32, #asctile.local<UB>>
    scf.yield %b : tensor<32xf32, #asctile.local<UB>>
  }
  return %r : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @no_merge_vector_with_cube_between(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) {
// CHECK:         asctile.vector_group {
// CHECK-NEXT:      %1 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %1, %arg1[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    %0 = asctile.cube_group {
// CHECK-NEXT:      %1 = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:      asctile.yield %1 : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    } : tensor<128xf16, #asctile.local<L1>>
// CHECK-NEXT:    asctile.vector_group {
// CHECK-NEXT:      %1 = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %1, %arg1[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_merge_vector_with_cube_between(%arg0: tensor<128xf16, #asctile.global>, %arg1: tensor<32xf32, #asctile.global>) {
  asctile.vector_group {
    %c0_i32 = arith.constant 0 : i32
    %ld = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg1[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  }
  asctile.cube_group {
    %c0_i32 = arith.constant 0 : i32
    %ld = asctile.load %arg0[%c0_i32] : tensor<128xf16, #asctile.global>, tensor<128xf16, #asctile.local<L1>>
    asctile.yield %ld : tensor<128xf16, #asctile.local<L1>>
  } : tensor<128xf16, #asctile.local<L1>>
  asctile.vector_group {
    %c0_i32 = arith.constant 0 : i32
    %ld = asctile.load %arg1[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg1[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @absorb_if_before_group(%arg0: tensor<32xf32, #asctile.global>, %arg1: i1) {
// CHECK:         asctile.vector_group {
// CHECK-NEXT:      %0 = scf.if %arg1 -> (i32) {
// CHECK-NEXT:        scf.yield %c0_i32 : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %c1_i32 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %1 = asctile.load %arg0[%0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %1, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @absorb_if_before_group(%arg0: tensor<32xf32, #asctile.global>, %arg1: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %idx = scf.if %arg1 -> (i32) {
    scf.yield %c0 : i32
  } else {
    scf.yield %c1 : i32
  }
  asctile.vector_group {
    %ld = asctile.load %arg0[%idx] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @absorb_for_before_group(%arg0: tensor<32xf32, #asctile.global>) {
// CHECK:         asctile.vector_group {
// CHECK-NEXT:      %0 = scf.for %arg1 = %c0 to %c3 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
// CHECK-NEXT:        %2 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:        %3 = arith.addi %arg2, %2 : i32
// CHECK-NEXT:        scf.yield %3 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %1 = asctile.load %arg0[%0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:      asctile.store %1, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @absorb_for_before_group(%arg0: tensor<32xf32, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %sum = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %step = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %step : i32
    scf.yield %next : i32
  }
  asctile.vector_group {
    %ld = asctile.load %arg0[%sum] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
    asctile.store %ld, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  }
  return
}

// CHECK-LABEL: func.func @no_interchange_if_with_barrier_in_else(%arg0: tensor<32xf32, #asctile.global>, %arg1: i1, %arg2: !ascendc.global_tensor<*xui8>, %arg3: !ascendc.local_tensor<*xui8>, %arg4: i32) {
// CHECK:         scf.if %arg1 {
// CHECK-NEXT:      asctile.vector_group {
// CHECK-NEXT:        %0 = asctile.load %arg0[%c0_i32] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:        asctile.store %0, %arg0[%c0_i32] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:      }
// CHECK-NEXT:    } else {  
// CHECK-NEXT:      ascendc.sync_all_soft %arg2, %arg3, %arg4 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_interchange_if_with_barrier_in_else(%arg0: tensor<32xf32, #asctile.global>, %arg1: i1, %arg2: !ascendc.global_tensor<*xui8>, %arg3: !ascendc.local_tensor<*xui8>, %arg4: i32) {
  scf.if %arg1 {
    asctile.vector_group {
      %c0 = arith.constant 0 : i32
      %ld = asctile.load %arg0[%c0] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
      asctile.store %ld, %arg0[%c0] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
    }
  } else {  
    ascendc.sync_all_soft %arg2, %arg3, %arg4 : !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
  }
  return
}
