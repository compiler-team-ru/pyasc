// RUN: ascir-opt -ascvf-fuse-vf-for %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_sequential_same_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:     ascvf.vec_scope {
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  ascvf.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascvf.load %0, %arg0[%arg1] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascvf.load %0, %arg0[%arg1] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:    ascvf.load %0, %arg0[%arg1] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @fuse_sequential_same_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c0 = arith.constant 0 : index
  ascvf.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    ascvf.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascvf.load %0, %arg0[%arg1] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    ascvf.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascvf.load %0, %arg0[%arg1] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    ascvf.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascvf.load %0, %arg0[%arg1] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}

// CHECK-LABEL: func.func @dont_fuse_non_sequential(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:     ascvf.vec_scope {
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  ascvf.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  ascvf.load %0, %arg0[%c0] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
// CHECK-NEXT:  ascvf.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @dont_fuse_non_sequential(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c0 = arith.constant 0 : index
  ascvf.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    ascvf.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    ascvf.load %0, %arg0[%c0] : !ascendc.reg_tensor<f32>, !ascendc.local_tensor<1024xf32>, index
    ascvf.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}

// CHECK-LABEL: func.func @dont_fuse_different_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
// CHECK:     ascvf.vec_scope {
// CHECK-NEXT:  %0 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:  %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
// CHECK-NEXT:  ascvf.vf_for %c0 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:  ascvf.vf_for %c1 : index {
// CHECK-NEXT:  ^bb0(%arg1: index):
// CHECK-NEXT:    ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:  }
// CHECK-NEXT:}
func.func @dont_fuse_different_upper_bound(%arg0: !ascendc.local_tensor<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ascvf.vec_scope {
    %0 = ascendc.reg_tensor : <f32>
    %1 = ascendc.create_mask f32, ALL : !ascendc.mask_reg
    ascvf.vf_for %c0 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
    ascvf.vf_for %c1 : index {
    ^bb0(%arg1: index):
      ascendc.add_reg %0, %0, %0, %1 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
    }
  }
  return
}
