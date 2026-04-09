// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-input-output-tensor %s | FileCheck %s

// CHECK-LABEL: func.func @input_output_tensor_ub_gm
// CHECK: %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
// CHECK: %1 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK: %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
func.func @input_output_tensor_ub_gm(%arg0 : !ascendc.global_tensor<*xf32>, %arg1 : !ascendc.global_tensor<*xf32>, %arg2 : !ascendc.global_tensor<*xf32>) {
    %c64_i32 = arith.constant 64 : i32
    %0 = ascendc.local_tensor_auto vecin() : <64xf32>
    ascendc.data_copy_l2 %0, %arg0, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %1, %arg1, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %2 = ascendc.local_tensor_auto vecout() : <64xf32>
    ascendc.data_copy_l2 %arg2, %2, %c64_i32 : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @input_output_tensor_ub_ub
// CHECK: %0 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK: %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
func.func @input_output_tensor_ub_ub(%arg0 : !ascendc.global_tensor<*xf32>) {
    %c64_i32 = arith.constant 64 : i32
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %0, %arg0, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i32
    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
    ascendc.data_copy_l2 %1, %0, %c64_i32 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
    return
}

// CHECK-LABEL: func.func @scf_for_no_init_args() {
// CHECK:         scf.for %arg0 = %c0 to %c10 step %c1 {
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_no_init_args() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c10 step %c1 {
    }
    return
}

// CHECK-LABEL: func.func @scf_for_no_init_args_use_inside(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c10 step %c1 {
// CHECK-NEXT:      ascendc.data_copy_l2 %arg0, %0, %c64_i64 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_no_init_args_use_inside(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    scf.for %i = %c0 to %c10 step %c1 {
        ascendc.data_copy_l2 %arg0, %0, %c64_i64 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    }
    return
}

// CHECK-LABEL: func.func @scf_for_init_dead_after_loop() {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      scf.yield %arg1 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_init_dead_after_loop() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %tensor : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_init_used_after_loop(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() output : <64xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:      %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:      ascendc.data_copy_l2 %1, %0, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %1) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      scf.yield %arg2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    ascendc.data_copy_l2 %arg0, %0, %c64_i64 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_init_used_after_loop(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        scf.yield %tensor : !ascendc.local_tensor<64xf32>
    }
    ascendc.data_copy_l2 %arg0, %0, %c64_i64 {direction = #ascendc.copy_direction<veccalc, gm>} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<64xf32>, i64
    return
}

// CHECK-LABEL: func.func @scf_for_iter_arg_not_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.muls_l2 %2, %arg2, %cst, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64
// CHECK-NEXT:      scf.yield %2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_iter_arg_not_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f32
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %inner = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.muls_l2 %inner, %tensor, %c2, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, f32, i64
        scf.yield %inner : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_iter_arg_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %2 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK-NEXT:      ascendc.data_copy_l2 %2, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
// CHECK-NEXT:      %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.sub_l2 %3, %2, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:      %4 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.if_aiv {
// CHECK-NEXT:        %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:        ascendc.data_copy_l2 %4, %2, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %4 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_iter_arg_used_after_yielded_dst(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.data_copy_l2 %1, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
        %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.sub_l2 %2, %1, %tensor, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
        scf.yield %1 : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_yielded_from_if_aiv(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %2 = ascendc.if_aiv(%arg2 : !ascendc.local_tensor<64xf32>) -> !ascendc.local_tensor<64xf32> {
// CHECK-NEXT:        %3 = ascendc.local_tensor_auto veccalc() input : <64xf32>
// CHECK-NEXT:        ascendc.data_copy_l2 %3, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
// CHECK-NEXT:        %4 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:        ascendc.sub_l2 %4, %3, %arg2, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:        %5 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:        ascendc.if_aiv {
// CHECK-NEXT:          %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:          ascendc.data_copy_l2 %5, %3, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:        }
// CHECK-NEXT:        ascendc.yield %5 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %2 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_yielded_from_if_aiv(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %inner = ascendc.if_aiv(%tensor : !ascendc.local_tensor<64xf32>) -> !ascendc.local_tensor<64xf32> {
            %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
            ascendc.data_copy_l2 %1, %arg0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.global_tensor<*xf32>, i64
            %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
            ascendc.sub_l2 %2, %1, %tensor, %c64_i64 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
            ascendc.yield %1 : !ascendc.local_tensor<64xf32>
        }
        scf.yield %inner : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_init_arg_used_inside(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:        %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:        ascendc.data_copy_l2 %1, %0, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %1) -> (!ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.mul_l2 %3, %arg2, %0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:      scf.yield %3 : !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_init_arg_used_inside(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f32
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor = %0) -> (!ascendc.local_tensor<64xf32>) {
        %inner = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.mul_l2 %inner, %tensor, %0, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
        scf.yield %inner : !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @scf_for_init_arg_used_twice(%arg0: !ascendc.global_tensor<*xf32>) {
// CHECK:         %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:    ascendc.if_aiv {
// CHECK-NEXT:        %c64_i64_0 = arith.constant 64 : i64
// CHECK-NEXT:        ascendc.data_copy_l2 %1, %0, %c64_i64_0 {direction = #ascendc.copy_direction<veccalc, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %2:2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (!ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>) {
// CHECK-NEXT:      %3 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:      ascendc.mul_l2 %3, %arg2, %arg3, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
// CHECK-NEXT:      scf.yield %3, %arg3 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @scf_for_init_arg_used_twice(%arg0: !ascendc.global_tensor<*xf32>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f32
    %c64_i64 = arith.constant 64 : i64
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
    %result:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%tensor1 = %0, %tensor2 = %0) -> (!ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>) {
        %inner = ascendc.local_tensor_auto veccalc() : <64xf32>
        ascendc.mul_l2 %inner, %tensor1, %tensor2, %c64_i64 {direction = #ascendc.copy_direction<gm, veccalc>} : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i64
        scf.yield %inner, %tensor2 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>
    }
    return
}
