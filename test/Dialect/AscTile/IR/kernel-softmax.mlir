// RUN: ascir-opt %s | ascir-opt
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt

module {
  func.func @softmax_kernel(%arg0: memref<*xf32, 22>, %arg1: memref<*xf32, 22>, %arg2: i32, %arg3: i32) attributes {ascendc.aicore, ascendc.global} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
    %1 = asctile.tensor %arg1(%arg2, %arg3) : memref<*xf32, 22>, !asctile.tensor<?x?xf32>
    %2 = ascendc.get_block_idx : i32
    %3 = ascendc.get_block_num : i32
    scf.for %arg4 = %2 to %arg2 step %3  : i32 {
      %4 = asctile.load %0[%arg4, %c0_i32], %cst : !asctile.tensor<?x?xf32>, !asctile.tile<1x1024xf32, UB>
      %5 = asctile.reduce_max_as_1d %4 : !asctile.tile<1x1024xf32, UB>, f32
      %6 = asctile.splat %5 : !asctile.tile<1x1024xf32, UB>
      %7 = arith.subf %4, %6 : !asctile.tile<1x1024xf32, UB>
      %8 = math.exp %7 : !asctile.tile<1x1024xf32, UB>
      %9 = asctile.reduce_sum_as_1d %8 : !asctile.tile<1x1024xf32, UB>, f32
      %10 = asctile.splat %9 : !asctile.tile<1x1024xf32, UB>
      %11 = arith.divf %8, %10 : !asctile.tile<1x1024xf32, UB>
      asctile.store %11, %1[%arg4, %c0_i32] : !asctile.tile<1x1024xf32, UB>, !asctile.tensor<?x?xf32>
    } {asctile.parallel, asctile.unroll_factor = 1 : index}
    return
  }
}
