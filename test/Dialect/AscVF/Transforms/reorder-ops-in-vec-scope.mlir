// RUN: ascir-opt -ascvf-reorder-ops-in-vec-scope %s | FileCheck %s

// CHECK-LABEL: func.func @reorder_supported_ops(%arg0: !ascendc.local_tensor<1xf32>) {
// CHECK:     ascvf.vec_scope {
// CHECK-NEXT:  %cst = arith.constant 0xFF800000 : f32
// CHECK-NEXT:  %c64 = arith.constant 64 : index
// CHECK-NEXT:  %c5 = arith.constant 5 : index
// CHECK-NEXT:  %c7_i64 = arith.constant 7 : i64
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %3 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
// CHECK-NEXT:  ascendc.duplicate %0, %cst : <f32>, f32
// CHECK-NEXT:  ascendc.duplicate %1, %cst : <f32>, f32
// CHECK-NEXT:  %4 = emitasc.variable %c64 : index, memref<1xui32>
// CHECK-NEXT:  %5 = arith.index_cast %c7_i64 : i64 to index
// CHECK-NEXT:  %6 = arith.remsi %c64, %c5 : index
// CHECK-NEXT:  %7 = arith.muli %6, %c5 : index
// CHECK-NEXT:  %8 = emitasc.variable %7 : index, memref<1xui32>
// CHECK-NEXT:  %9 = arith.ceildivsi %7, %5 : index
// CHECK-NEXT:  %10 = arith.cmpi eq, %5, %6 : index
// CHECK-NEXT:  %11 = arith.divsi %9, %c5 : index
// CHECK-NEXT:  %12 = arith.subi %11, %c5 : index
// CHECK-NEXT:  %13 = arith.addi %12, %c5 : index
// CHECK-NEXT:  ascvf.load %2, %arg0[%c64] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1xf32>, index
// CHECK-NEXT:  %14 = ascendc.update_mask f32, %4 : memref<1xui32>
// CHECK-NEXT:  ascendc.select_reg %1, %2, %0, %14 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  ascvf.store %arg0[%13], %1, %3 : !ascendc.local_tensor<1xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:}
func.func @reorder_supported_ops(%arg0: !ascendc.local_tensor<1xf32>) {
  ascvf.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %cst = arith.constant 0xFF800000 : f32
    ascendc.duplicate %0, %cst : <f32>, f32
    %c64 = arith.constant 64 : index
    %1 = emitasc.variable %c64 : index, memref<1xui32>
    %2 = ascendc.reg_tensor : <f32>
    %3 = ascendc.reg_tensor : <f32>
    %c5 = arith.constant 5 : index
    %c7_i64 = arith.constant 7 : i64
    %4 = arith.index_cast %c7_i64 : i64 to index
    %5 = arith.remsi %c64, %c5 : index
    %6 = arith.muli %5, %c5 : index
    %7 = emitasc.variable %6 : index, memref<1xui32>
    %8 = arith.ceildivsi %6, %4 : index
    ascendc.duplicate %2, %cst : <f32>, f32
    %9 = arith.cmpi eq, %4, %5 : index
    ascvf.load %3, %arg0[%c64] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1xf32>, index
    %10 = arith.divsi %8, %c5 : index
    %11 = ascendc.update_mask f32, %1 : memref<1xui32>
    %12 = arith.subi %10, %c5 : index
    ascendc.select_reg %2, %3, %0, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    %13 = ascendc.create_mask f32, VL1 : !ascendc.mask_reg
    %14 = arith.addi %12, %c5 : index
    ascvf.store %arg0[%14], %2, %13 : !ascendc.local_tensor<1xf32>, index, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  }
  return
}

// CHECK-LABEL: func.func @reorder_inside_block(%arg0: !ascendc.local_tensor<1xf32>) {
// CHECK:     ascvf.vec_scope {
// CHECK-NEXT:  %c64 = arith.constant 64 : index
// CHECK-NEXT:  %cst = arith.constant 0xFF800000 : f32
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  ascendc.duplicate %0, %cst : <f32>, f32
// CHECK-NEXT:  %2 = emitasc.variable %c64 : index, memref<1xui32>
// CHECK-NEXT:  ascvf.vf_for %c64 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    %3 = arith.muli %arg1, %c64 : index
// CHECK-NEXT:    %4 = arith.divsi %3, %c64 : index
// CHECK-NEXT:    %5 = arith.subi %4, %c64 : index
// CHECK-NEXT:    %6 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:    ascvf.load %1, %arg0[%5] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1xf32>, index
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @reorder_inside_block(%arg0: !ascendc.local_tensor<1xf32>) {
  ascvf.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %c64 = arith.constant 64 : index
    %1 = emitasc.variable %c64 : index, memref<1xui32>
    %cst = arith.constant 0xFF800000 : f32
    ascendc.duplicate %0, %cst : <f32>, f32
    %2 = ascendc.reg_tensor : <f32>
    ascvf.vf_for %c64 : index {
    ^bb0(%arg1: index):
      %3 = ascendc.update_mask f32, %1 : memref<1xui32>
      %4 = arith.muli %arg1, %c64 : index
      %5 = arith.divsi %4, %c64 : index
      %6 = arith.subi %5, %c64 : index
      ascvf.load %2, %arg0[%6] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1xf32>, index
    }
  }
  return
}
