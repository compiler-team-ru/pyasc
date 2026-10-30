// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: asctile-opt -ascvf2-raise-vf-for %s | FileCheck %s

func.func @keep_ineligible_loops(%arg0: index) {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  ascvf2.vf_group {
    scf.for %nonzero = %c2 to %c8 step %c1 {
      arith.addi %nonzero, %c1 : index
    }
    scf.for %stepped = %c1 to %c8 step %c2 {
      arith.addi %stepped, %c1 : index
    }
    %result = scf.for %carrier = %c1 to %c8 step %c1 iter_args(%accumulator = %c1) -> (index) {
      %next = arith.addi %accumulator, %carrier : index
      scf.yield %next : index
    }
    arith.addi %result, %c1 : index
  }
  return
}


func.func @keep_loop_outside_vec_scope(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %index = %c0 to %c8 step %c1 {
    arith.addi %index, %c1 : index
  }
  return
}

// CHECK-LABEL: func.func @keep_loop_outside_vec_scope
// CHECK: scf.for
// CHECK-NOT: emitasc.vf_for

func.func @raise_eligible_loops(%arg0: index) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  ascvf2.vf_group {
    scf.for %index = %c0 to %c8 step %c1 {
      scf.for %nested = %c0 to %c4 step %c1 {
        arith.addi %index, %nested : index
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @raise_eligible_loops
// CHECK: ascvf2.vf_group
// CHECK: emitasc.vf_for
// CHECK: emitasc.vf_for
// CHECK: arith.addi
// CHECK-NOT: scf.for
// CHECK-NOT: emitasc.vf_for
