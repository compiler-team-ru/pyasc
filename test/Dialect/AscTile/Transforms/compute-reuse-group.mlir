// RUN: ascir-opt -ascendc-compute-reuse-group %s | FileCheck %s

// CHECK-LABEL: func.func @unroll_nested_loop
// CHECK:       scf.execute_region {
// CHECK-NEXT:    scf.for %arg1 = %c0 to %c4 step %c2 {
// CHECK-NEXT:      scf.execute_region {
// CHECK-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:        }
// CHECK-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK:           } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 0 : i64, asctile.unrolled_loop = 0 : i64}
// CHECK-NEXT:      scf.execute_region {
// CHECK-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:        }
// CHECK:           } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 0 : i64, asctile.unrolled_loop = 1 : i64}
// CHECK-NEXT:      scf.execute_region {
// CHECK-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:        }
// CHECK-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK:           } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 1 : i64, asctile.unrolled_loop = 2 : i64}
// CHECK-NEXT:      scf.execute_region {
// CHECK-NEXT:        scf.for %arg2 = %c0 to %c4 step %c2 {
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:          asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:        }
// CHECK:           } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 1 : i64, asctile.unrolled_loop = 3 : i64}
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.execute_region {
// CHECK-NEXT:      scf.for %arg1 = %c0 to %c4 step %c2 {
// CHECK-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:      }
// CHECK-NEXT:      asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK:         } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 2 : i64, asctile.unrolled_loop = 4 : i64}
// CHECK-NEXT:    scf.execute_region {
// CHECK-NEXT:      scf.for %arg1 = %c0 to %c4 step %c2 {
// CHECK-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:        asctile.set_value %c0_i32, %arg0[%c0_i32] {asc.reuse_group = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:      }
// CHECK:         } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 2 : i64, asctile.unrolled_loop = 5 : i64}
// CHECK:       } {asctile.unroll_factor = 2 : i64, asctile.unrolled_loop = 6 : i64}
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @unroll_nested_loop(%arg0: tensor<32xi32, #asctile.global>) {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  scf.execute_region {
    scf.for %arg1 = %c0 to %c4 step %c2 {
      scf.execute_region {
        scf.for %arg2 = %c0 to %c4 step %c2 {
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
        }
        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
        scf.yield
      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 0 : i64, asctile.unrolled_loop = 0 : i64}
      scf.execute_region {
        scf.for %arg2 = %c0 to %c4 step %c2 {
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
        }
        scf.yield
      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 0 : i64, asctile.unrolled_loop = 1 : i64}
      scf.execute_region {
        scf.for %arg2 = %c0 to %c4 step %c2 {
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
        }
        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
        scf.yield
      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 1 : i64, asctile.unrolled_loop = 2 : i64}
      scf.execute_region {
        scf.for %arg2 = %c0 to %c4 step %c2 {
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
          asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
        }
        scf.yield
      } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 1 : i64, asctile.unrolled_loop = 3 : i64}
    }
    scf.execute_region {
      scf.for %arg1 = %c0 to %c4 step %c2 {
        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
      }
      asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
      scf.yield
    } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 2 : i64, asctile.unrolled_loop = 4 : i64}
    scf.execute_region {
      scf.for %arg1 = %c0 to %c4 step %c2 {
        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
        asctile.set_value %c0_i32, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
      }
      scf.yield
    } {asctile.unroll_factor = 2 : i64, asctile.unroll_iter = 2 : i64, asctile.unrolled_loop = 5 : i64}
    scf.yield
  } {asctile.unroll_factor = 2 : i64, asctile.unrolled_loop = 6 : i64}
  return
}
