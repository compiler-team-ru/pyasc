// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-fixup-mmad-acc-params %s | FileCheck %s

// CHECK-LABEL: func.func @mmad_single
// CHECK-NOT:     emitasc.variable
// CHECK-NOT:     cmatrixInitVal
func.func @mmad_single(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf32>) {
  %c16_i32 = arith.constant 16 : i32
  %0 = ascendc.local_tensor_auto co1() : <16x16xf32>
  %1 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32)
  ascendc.mmad %0, %arg0, %arg1, %1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @mmad_acc_single
// CHECK:         %[[C01ACC:[0-9a-z_]+]] = ascendc.local_tensor_auto co1() : <64x256xf32>
// CHECK-NEXT:    %[[VARPTR:[0-9a-z_]+]] = emitasc.variable true, memref<1xi1>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[VARVAL:[0-9a-z_]+]] = memref.load %[[VARPTR]][%c0] : memref<1xi1>
// CHECK-NEXT:    %[[MMADPARAMS:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c64_i32 : i32, "n" = %c256_i32 : i32, "k" = %c128_i32 : i32, "cmatrixInitVal" = %[[VARVAL]] : i1)
// CHECK-NEXT:    ascendc.mmad %[[C01ACC]], %arg0, %arg1, %[[MMADPARAMS]] : !ascendc.local_tensor<64x256xf32>, !ascendc.local_tensor<64x128xf16>, !ascendc.local_tensor<128x256xf16>, !ascendc.mmad_params
// CHECK-NEXT:    %[[FALSE:[0-9a-z_]+]] = arith.constant false
// CHECK-NEXT:    memref.store %[[FALSE]], %[[VARPTR]][%c0] : memref<1xi1>
// CHECK-NEXT:    return
func.func @mmad_acc_single(%arg0: !ascendc.local_tensor<64x128xf16>, %arg1: !ascendc.local_tensor<128x256xf16>) -> (!ascendc.local_tensor<64x256xf32>) {
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c256_i32 = arith.constant 256 : i32
  %0 = ascendc.local_tensor_auto co1() : <64x256xf32>
  %1 = emitasc.init_struct !ascendc.mmad_params("m" = %c64_i32 : i32, "n" = %c256_i32 : i32, "k" = %c128_i32 : i32, "cmatrixInitVal" = %c1_i32 : i32)
  ascendc.mmad %0, %arg0, %arg1, %1 : !ascendc.local_tensor<64x256xf32>, !ascendc.local_tensor<64x128xf16>, !ascendc.local_tensor<128x256xf16>, !ascendc.mmad_params
  return %0 : !ascendc.local_tensor<64x256xf32>
}

// CHECK-LABEL: func.func @multi_mmad_multi_dst
// CHECK-NOT:     emitasc.variable
// CHECK-NOT:     cmatrixInitVal
func.func @multi_mmad_multi_dst(%arg0: !ascendc.local_tensor<32x16xf16>, %arg1: !ascendc.local_tensor<16x32xf16>) -> (!ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = ascendc.local_tensor_auto co1() : <16x32xf32>
  scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
    %1 = arith.muli %arg2, %c16_i32 : i32
    %2 = arith.muli %1, %c16_i32 : i32
    %3 = ascendc.local_tensor.subindex %arg0[%2] : !ascendc.local_tensor<32x16xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %4 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c32_i32 : i32, "k" = %c16_i32 : i32)
    ascendc.mmad %0, %3, %arg1, %4 : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x32xf16>, !ascendc.mmad_params
  }
  %6 = ascendc.local_tensor_auto co1() : <32x32xf32>
  %7 = emitasc.init_struct !ascendc.mmad_params("m" = %c32_i32 : i32, "n" = %c32_i32 : i32, "k" = %c16_i32 : i32)
  ascendc.mmad %6, %arg0, %arg1, %7 : !ascendc.local_tensor<32x32xf32>, !ascendc.local_tensor<32x16xf16>, !ascendc.local_tensor<16x32xf16>, !ascendc.mmad_params
  return %0, %6 : !ascendc.local_tensor<16x32xf32>, !ascendc.local_tensor<32x32xf32>
}

// CHECK-LABEL: func.func @multi_mmad_acc_multi_dst
// CHECK:         %[[C01ACC1:[0-9a-z_]+]] = ascendc.local_tensor_auto co1() : <16x16xf32>
// CHECK-NEXT:    %[[VAR1PTR:[0-9a-z_]+]] = emitasc.variable true, memref<1xi1>
// CHECK:         scf.for
// CHECK:           %[[IDX1:[0-9a-z_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAR1VAL:[0-9a-z_]+]] = memref.load %[[VAR1PTR]][%[[IDX1]]] : memref<1xi1>
// CHECK-NEXT:      %[[MMADPARAMS1:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %[[VAR1VAL]] : i1)
// CHECK-NEXT:      ascendc.mmad %[[C01ACC1]], %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}, %[[MMADPARAMS1]] : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      %[[FALSE1:[0-9a-z_]+]] = arith.constant false
// CHECK-NEXT:      memref.store %[[FALSE1]], %[[VAR1PTR]][%[[IDX1]]] : memref<1xi1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[C01ACC2:[0-9a-z_]+]] = ascendc.local_tensor_auto co1() : <16x16xf32>
// CHECK-NEXT:    %[[VAR2PTR:[0-9a-z_]+]] = emitasc.variable true, memref<1xi1>
// CHECK-NEXT:    %[[IDX2:[0-9a-z_]+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAR2VAL:[0-9a-z_]+]] = memref.load %[[VAR2PTR]][%[[IDX2]]] : memref<1xi1>
// CHECK-NEXT:    %[[MMADPARAMS2:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c32_i32 : i32, "cmatrixInitVal" = %[[VAR2VAL]] : i1)
// CHECK-NEXT:    ascendc.mmad %[[C01ACC2]], %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}, %[[MMADPARAMS2]] : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    %[[FALSE2:[0-9a-z_]+]] = arith.constant false
// CHECK-NEXT:    memref.store %[[FALSE2]], %[[VAR2PTR]][%[[IDX2]]] : memref<1xi1>
// CHECK-NEXT:    return
func.func @multi_mmad_acc_multi_dst(%arg0: !ascendc.local_tensor<16x32xf16>, %arg1: !ascendc.local_tensor<32x16xf16>) -> (!ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = ascendc.local_tensor_auto co1() : <16x16xf32>
  scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
    %1 = arith.muli %arg2, %c16_i32 : i32
    %2 = ascendc.local_tensor.subindex %arg0[%1] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %3 = arith.muli %1, %c16_i32 : i32
    %4 = ascendc.local_tensor.subindex %arg1[%3] : !ascendc.local_tensor<32x16xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %5 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %c1_i32 : i32)
    ascendc.mmad %0, %2, %4, %5 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
  }
  %6 = ascendc.local_tensor_auto co1() : <16x16xf32>
  %7 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c32_i32 : i32, "cmatrixInitVal" = %c1_i32 : i32)
  ascendc.mmad %6, %arg0, %arg1, %7 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x16xf16>, !ascendc.mmad_params
  return %0, %6 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @multi_mmad_acc_single_dst
// CHECK:         %[[C01ACC:[0-9a-z_]+]] = ascendc.local_tensor_auto co1() : <16x16xf32>
// CHECK-NEXT:    %[[VARPTR:[0-9a-z_]+]] = emitasc.variable true, memref<1xi1>
// CHECK:         scf.for
// CHECK:           %[[IDX1:[0-9a-z_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAR1VAL:[0-9a-z_]+]] = memref.load %[[VARPTR]][%[[IDX1]]] : memref<1xi1>
// CHECK-NEXT:      %[[MMADPARAMS1:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %[[VAR1VAL]] : i1)
// CHECK-NEXT:      ascendc.mmad %[[C01ACC]], %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}, %[[MMADPARAMS1]] : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      %[[FALSE1:[0-9a-z_]+]] = arith.constant false
// CHECK-NEXT:      memref.store %[[FALSE1]], %[[VARPTR]][%[[IDX1]]] : memref<1xi1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[IDX2:[0-9a-z_]+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAR2VAL:[0-9a-z_]+]] = memref.load %[[VARPTR]][%[[IDX2]]] : memref<1xi1>
// CHECK-NEXT:    %[[MMADPARAMS2:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c32_i32 : i32, "cmatrixInitVal" = %[[VAR2VAL]] : i1)
// CHECK-NEXT:    ascendc.mmad %[[C01ACC]], %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}, %[[MMADPARAMS2]] : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    %[[FALSE2:[0-9a-z_]+]] = arith.constant false
// CHECK-NEXT:    memref.store %[[FALSE2]], %[[VARPTR]][%[[IDX2]]] : memref<1xi1>
// CHECK-NEXT:    return
func.func @multi_mmad_acc_single_dst(%arg0: !ascendc.local_tensor<16x32xf16>, %arg1: !ascendc.local_tensor<32x16xf16>) -> (!ascendc.local_tensor<16x16xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = ascendc.local_tensor_auto co1() : <16x16xf32>
  scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
    %1 = arith.muli %arg2, %c16_i32 : i32
    %2 = ascendc.local_tensor.subindex %arg0[%1] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %3 = arith.muli %1, %c16_i32 : i32
    %4 = ascendc.local_tensor.subindex %arg1[%3] : !ascendc.local_tensor<32x16xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %5 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %c1_i32 : i32)
    ascendc.mmad %0, %2, %4, %5 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
  }
  %6 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c32_i32 : i32, "cmatrixInitVal" = %c1_i32 : i32)
  ascendc.mmad %0, %arg0, %arg1, %6 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x16xf16>, !ascendc.mmad_params
  return %0 : !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @mmad_acc_and_mmad_wo_acc
// CHECK:         %[[C01ACC1:[0-9a-z_]+]] = ascendc.local_tensor_auto co1() : <16x16xf32>
// CHECK-NEXT:    %[[VARPTR:[0-9a-z_]+]] = emitasc.variable true, memref<1xi1>
// CHECK:         scf.for
// CHECK:           %[[IDX:[0-9a-z_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VARVAL:[0-9a-z_]+]] = memref.load %[[VARPTR]][%[[IDX]]] : memref<1xi1>
// CHECK-NEXT:      %[[MMADPARAMS1:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %[[VARVAL]] : i1)
// CHECK-NEXT:      ascendc.mmad %[[C01ACC1]], %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}, %[[MMADPARAMS1]] : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:      %[[FALSE:[0-9a-z_]+]] = arith.constant false
// CHECK-NEXT:      memref.store %[[FALSE]], %[[VARPTR]][%[[IDX]]] : memref<1xi1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[C01ACC2:[0-9a-z_]+]] = ascendc.local_tensor_auto co1() : <16x16xf32>
// CHECK-NEXT:    %[[MMADPARAMS2:[0-9a-z_]+]] = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c32_i32 : i32)
// CHECK-NEXT:    ascendc.mmad %[[C01ACC2]], %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}, %[[MMADPARAMS2]] : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x16xf16>, !ascendc.mmad_params
// CHECK-NEXT:    return
func.func @mmad_acc_and_mmad_wo_acc(%arg0: !ascendc.local_tensor<16x32xf16>, %arg1: !ascendc.local_tensor<32x16xf16>) -> (!ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = ascendc.local_tensor_auto co1() : <16x16xf32>
  scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
    %1 = arith.muli %arg2, %c16_i32 : i32
    %2 = ascendc.local_tensor.subindex %arg0[%1] : !ascendc.local_tensor<16x32xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %3 = arith.muli %1, %c16_i32 : i32
    %4 = ascendc.local_tensor.subindex %arg1[%3] : !ascendc.local_tensor<32x16xf16>, i32, !ascendc.local_tensor<16x16xf16>
    %5 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c16_i32 : i32, "cmatrixInitVal" = %c1_i32 : i32)
    ascendc.mmad %0, %2, %4, %5 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>, !ascendc.mmad_params
  }
  %6 = ascendc.local_tensor_auto co1() : <16x16xf32>
  %7 = emitasc.init_struct !ascendc.mmad_params("m" = %c16_i32 : i32, "n" = %c16_i32 : i32, "k" = %c32_i32 : i32)
  ascendc.mmad %6, %arg0, %arg1, %7 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x32xf16>, !ascendc.local_tensor<32x16xf16>, !ascendc.mmad_params
  return %0, %6 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
}
