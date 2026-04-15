// RUN: ascir-opt %s | ascir-opt
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt

module attributes {asc.kernel_type = "cube", asc.memory_consumed = {}, asc.soc_version = "Ascend950PR_9599"} {
  emitc.include "kernel_operator.h"
  func.func @matmul_kernel(%arg0: memref<*xf16, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}, %arg1: memref<*xf16, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}, %arg2: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}) attributes {ascendc.aicore, ascendc.global} {
    %c32768_i64 = arith.constant 32768 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ascendc.pipe
    %1 = ascendc.global_tensor : !ascendc.global_tensor<64x128xf16>
    ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<64x128xf16>, memref<*xf16, 22>
    %2 = ascendc.global_tensor : !ascendc.global_tensor<128x256xf16>
    ascendc.global_tensor.set_global_buffer %2, %arg1 : !ascendc.global_tensor<128x256xf16>, memref<*xf16, 22>
    %3 = ascendc.global_tensor : !ascendc.global_tensor<64x256xf32>
    ascendc.global_tensor.set_global_buffer %3, %arg2 : !ascendc.global_tensor<64x256xf32>, memref<*xf32, 22>
    %4 = ascendc.tbuf : <co1>
    ascendc.pipe.init_buffer %0, %4, %c65536_i64 : !ascendc.tbuf<co1>, i64
    %5 = ascendc.tbuf.get_tensor %4 : !ascendc.tbuf<co1>, !ascendc.local_tensor<64x256xf32>
    %6 = ascendc.global_tensor.subindex %1[%c0_i32] : !ascendc.global_tensor<64x128xf16>, i32, !ascendc.global_tensor<64x128xf16>
    %7 = ascendc.tbuf : <a1>
    ascendc.pipe.init_buffer %0, %7, %c16384_i64 : !ascendc.tbuf<a1>, i64
    %8 = ascendc.tbuf.get_tensor %7 : !ascendc.tbuf<a1>, !ascendc.local_tensor<64x128xf16>
    %9 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c64_i32, %c128_i32, %c0_i32, %c128_i32, %c64_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32, i32, i32, i32, i32
    ascendc.get_buf pipe_mte2, 0
    ascendc.data_copy_l2 %8, %6, %9 : !ascendc.local_tensor<64x128xf16>, !ascendc.global_tensor<64x128xf16>, !ascendc.nd2nz_params
    ascendc.rls_buf pipe_mte2, 0
    %10 = ascendc.global_tensor.subindex %2[%c0_i32] : !ascendc.global_tensor<128x256xf16>, i32, !ascendc.global_tensor<128x256xf16>
    %11 = ascendc.tbuf : <a1>
    ascendc.pipe.init_buffer %0, %11, %c65536_i64 : !ascendc.tbuf<a1>, i64
    %12 = ascendc.tbuf.get_tensor %11 : !ascendc.tbuf<a1>, !ascendc.local_tensor<128x256xf16>
    %13 = ascendc.construct !ascendc.nd2nz_params(%c1_i32, %c128_i32, %c256_i32, %c0_i32, %c256_i32, %c128_i32, %c1_i32, %c0_i32) : i32, i32, i32, i32, i32, i32, i32, i32
    ascendc.get_buf pipe_mte2, 1
    ascendc.data_copy_l2 %12, %10, %13 : !ascendc.local_tensor<128x256xf16>, !ascendc.global_tensor<128x256xf16>, !ascendc.nd2nz_params
    ascendc.rls_buf pipe_mte2, 1
    %14 = ascendc.local_tensor.subindex %8[%c0_i32] : !ascendc.local_tensor<64x128xf16>, i32, !ascendc.local_tensor<64x128xf16>
    %15 = ascendc.tbuf : <a2>
    ascendc.pipe.init_buffer %0, %15, %c8192_i64 : !ascendc.tbuf<a2>, i64
    %16 = ascendc.tbuf.get_tensor %15 : !ascendc.tbuf<a2>, !ascendc.local_tensor<64x64xf16>
    %17 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c16_i32 : i32, "srcStride" = %c1_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %false : i1)
    ascendc.get_buf pipe_mte1, 2
    ascendc.get_buf pipe_mte1, 0
    ascendc.load_data_g2l %16, %14, %17 : !ascendc.local_tensor<64x64xf16>, !ascendc.local_tensor<64x128xf16>, !ascendc.load_data_2d_params
    ascendc.rls_buf pipe_mte1, 0
    ascendc.rls_buf pipe_mte1, 2
    %18 = ascendc.local_tensor.subindex %12[%c0_i32] : !ascendc.local_tensor<128x256xf16>, i32, !ascendc.local_tensor<128x256xf16>
    %19 = ascendc.tbuf : <b2>
    ascendc.pipe.init_buffer %0, %19, %c32768_i64 : !ascendc.tbuf<b2>, i64
    %20 = ascendc.tbuf.get_tensor %19 : !ascendc.tbuf<b2>, !ascendc.local_tensor<64x256xf16>
    %21 = emitasc.init_struct !ascendc.load_data_2d_params("repeatTimes" = %c16_i32 : i32, "srcStride" = %c8_i32 : i32, "dstGap" = %c0_i32 : i32, "ifTranspose" = %true : i1)
    scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %33 = arith.muli %arg3, %c4096_i32 : i32
      %34 = arith.muli %arg3, %c256_i32 : i32
      %35 = ascendc.local_tensor.subindex %18[%34] : !ascendc.local_tensor<128x256xf16>, i32, !ascendc.local_tensor<128x256xf16>
      %36 = ascendc.local_tensor.subindex %20[%33] : !ascendc.local_tensor<64x256xf16>, i32, !ascendc.local_tensor<64x256xf16>
      ascendc.get_buf pipe_mte1, 3
      ascendc.get_buf pipe_mte1, 1
      ascendc.load_data_g2l %36, %35, %21 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<128x256xf16>, !ascendc.load_data_2d_params
      ascendc.rls_buf pipe_mte1, 1
      ascendc.rls_buf pipe_mte1, 3
    }
    %22 = emitasc.init_struct !ascendc.mmad_params("m" = %c64_i32 : i32, "n" = %c256_i32 : i32, "k" = %c64_i32 : i32, "isBias" = %c1_i32 : i32)
    ascendc.get_buf pipe_m, 4
    ascendc.get_buf pipe_m, 2
    ascendc.get_buf pipe_m, 3
    ascendc.mmad %5, %16, %20, %22 : !ascendc.local_tensor<64x256xf32>, !ascendc.local_tensor<64x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
    ascendc.rls_buf pipe_m, 3
    ascendc.rls_buf pipe_m, 2
    ascendc.rls_buf pipe_m, 4
    %23 = ascendc.local_tensor.subindex %8[%c4096_i32] : !ascendc.local_tensor<64x128xf16>, i32, !ascendc.local_tensor<64x128xf16>
    %24 = ascendc.tbuf : <a2>
    ascendc.pipe.init_buffer %0, %24, %c8192_i64 : !ascendc.tbuf<a2>, i64
    %25 = ascendc.tbuf.get_tensor %24 : !ascendc.tbuf<a2>, !ascendc.local_tensor<64x64xf16>
    ascendc.get_buf pipe_mte1, 5
    ascendc.get_buf pipe_mte1, 0
    ascendc.load_data_g2l %25, %23, %17 : !ascendc.local_tensor<64x64xf16>, !ascendc.local_tensor<64x128xf16>, !ascendc.load_data_2d_params
    ascendc.rls_buf pipe_mte1, 0
    ascendc.rls_buf pipe_mte1, 5
    %26 = ascendc.local_tensor.subindex %12[%c1024_i32] : !ascendc.local_tensor<128x256xf16>, i32, !ascendc.local_tensor<128x256xf16>
    %27 = ascendc.tbuf : <b2>
    ascendc.pipe.init_buffer %0, %27, %c32768_i64 : !ascendc.tbuf<b2>, i64
    %28 = ascendc.tbuf.get_tensor %27 : !ascendc.tbuf<b2>, !ascendc.local_tensor<64x256xf16>
    scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %33 = arith.muli %arg3, %c4096_i32 : i32
      %34 = arith.muli %arg3, %c256_i32 : i32
      %35 = ascendc.local_tensor.subindex %26[%34] : !ascendc.local_tensor<128x256xf16>, i32, !ascendc.local_tensor<128x256xf16>
      %36 = ascendc.local_tensor.subindex %28[%33] : !ascendc.local_tensor<64x256xf16>, i32, !ascendc.local_tensor<64x256xf16>
      ascendc.get_buf pipe_mte1, 6
      ascendc.get_buf pipe_mte1, 1
      ascendc.load_data_g2l %36, %35, %21 : !ascendc.local_tensor<64x256xf16>, !ascendc.local_tensor<128x256xf16>, !ascendc.load_data_2d_params
      ascendc.rls_buf pipe_mte1, 1
      ascendc.rls_buf pipe_mte1, 6
    }
    ascendc.get_buf pipe_m, 4
    ascendc.get_buf pipe_m, 5
    ascendc.get_buf pipe_m, 6
    ascendc.mmad %5, %25, %28, %22 : !ascendc.local_tensor<64x256xf32>, !ascendc.local_tensor<64x64xf16>, !ascendc.local_tensor<64x256xf16>, !ascendc.mmad_params
    ascendc.rls_buf pipe_m, 6
    ascendc.rls_buf pipe_m, 5
    ascendc.rls_buf pipe_m, 4
    %29 = ascendc.global_tensor.subindex %3[%c0_i32] : !ascendc.global_tensor<64x256xf32>, i32, !ascendc.global_tensor<64x256xf32>
    %30 = emitasc.init_struct !ascendc.fixpipe_params_v220("nSize" = %c256_i32 : i32, "mSize" = %c64_i32 : i32, "srcStride" = %c64_i32 : i32, "dstStride" = %c256_i32 : i32)
    %31 = ascendc.construct !ascendc.co2_layout(%c1_i32) constexpr static : i32
    %32 = ascendc.construct !ascendc.fixpipe_config(%31) constexpr static : !ascendc.co2_layout
    ascendc.get_buf pipe_fix, 4
    ascendc.fixpipe %29, %5, %30, %32 : !ascendc.global_tensor<64x256xf32>, !ascendc.local_tensor<64x256xf32>, !ascendc.fixpipe_params_v220, !ascendc.fixpipe_config
    ascendc.rls_buf pipe_fix, 4
    return
  }
}
