// RUN: ascir-opt %s | ascir-opt
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt

module {
  func.func @matmul_kernel(%arg0: memref<*xf16, 22>, %arg1: memref<*xf16, 22>, %arg2: memref<*xf32, 22>) attributes {ascendc.aicore, ascendc.global} {
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.tensor %arg0() : memref<*xf16, 22>, !asctile.tensor<64x128xf16>
    %1 = asctile.tensor %arg1() : memref<*xf16, 22>, !asctile.tensor<128x256xf16>
    %2 = asctile.tensor %arg2() : memref<*xf32, 22>, !asctile.tensor<64x256xf32>
    %3 = asctile.accumulator : !asctile.tile<64x256xf32, L0C>
    %4 = asctile.load %0[%c0_i32, %c0_i32], %cst : !asctile.tensor<64x128xf16>, !asctile.tile<64x128xf16, L1>
    %5 = asctile.load %1[%c0_i32, %c0_i32], %cst : !asctile.tensor<128x256xf16>, !asctile.tile<128x256xf16, L1>
    scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %6 = arith.muli %arg3, %c64_i32 : i32
      %7 = asctile.copy %4[%c0_i32, %6] : !asctile.tile<64x128xf16, L1>, !asctile.tile<64x64xf16, L0A>
      %8 = asctile.copy %5[%6, %c0_i32] : !asctile.tile<128x256xf16, L1>, !asctile.tile<64x256xf16, L0B>
      asctile.matmul_acc %7, %8, %3 : <64x64xf16, L0A>, <64x256xf16, L0B>, <64x256xf32, L0C>
    } {asctile.parallel, asctile.unroll_factor = 2 : index}
    asctile.store %3, %2[%c0_i32, %c0_i32] : !asctile.tile<64x256xf32, L0C>, !asctile.tensor<64x256xf32>
    return
  }
}
