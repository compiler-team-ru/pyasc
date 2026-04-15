// RUN: ascir-opt --asclower-redress-i1-tile -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL:func.func @redress_splat_constant(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:    %cst = arith.constant dense<-1> : !asctile.tile<2xi8, UB>
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %cst : !asctile.tile<2xi8, UB> to !asctile.tile<16xi1, UB>
// CHECK-NEXT:    %1 = "select_like"(%0, %arg0, %arg1) : (!asctile.tile<16xi1, UB>, !asctile.tile<16xf32, UB>, !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB>
// CHECK-NEXT:    return %1 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  }
func.func @redress_splat_constant(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %cst = arith.constant dense<true> : !asctile.tile<16xi1, UB>
  %0 = "select_like"(%cst, %arg0, %arg1) : (!asctile.tile<16xi1, UB>, !asctile.tile<16xf32, UB>, !asctile.tile<16xf32, UB>) -> (!asctile.tile<16xf32, UB>)
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @redress_dense_constant(%arg0: !asctile.tile<16x2xf32, UB>, %arg1: !asctile.tile<16x2xf32, UB>) -> !asctile.tile<16x2xf32, UB> {
// CHECK-NEXT:    %cst = arith.constant dense<[85, -86, 85, -86]> : !asctile.tile<4xi8, UB>
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %cst : !asctile.tile<4xi8, UB> to !asctile.tile<16x2xi1, UB>
// CHECK-NEXT:    %1 = "select_like"(%0, %arg0, %arg1) : (!asctile.tile<16x2xi1, UB>, !asctile.tile<16x2xf32, UB>, !asctile.tile<16x2xf32, UB>) -> !asctile.tile<16x2xf32, UB>
// CHECK-NEXT:    return %1 : !asctile.tile<16x2xf32, UB>
// CHECK-NEXT:  }
func.func @redress_dense_constant(%arg0: !asctile.tile<16x2xf32, UB>, %arg1: !asctile.tile<16x2xf32, UB>) -> !asctile.tile<16x2xf32, UB> {
  %cst = arith.constant dense<[[true, false], [true, false], [true, false], [true, false],
                               [false, true], [false, true], [false, true], [false, true],
                               [true, false], [true, false], [true, false], [true, false],
                               [false, true], [false, true], [false, true], [false, true]]> : !asctile.tile<16x2xi1, UB>
  %0 = "select_like"(%cst, %arg0, %arg1) : (!asctile.tile<16x2xi1, UB>, !asctile.tile<16x2xf32, UB>, !asctile.tile<16x2xf32, UB>) -> (!asctile.tile<16x2xf32, UB>)
  return %0 : !asctile.tile<16x2xf32, UB>
}
