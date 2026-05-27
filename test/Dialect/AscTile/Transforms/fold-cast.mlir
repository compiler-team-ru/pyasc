// RUN: ascir-opt -asctile-fold-cast %s | FileCheck %s

// CHECK-LABEL: func.func @fold_cast_i8_to_i32(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func @fold_cast_i8_to_i32(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi16, UB>
  %1 = asctile.cast %0 : !asctile.tile<32xi16, UB> to !asctile.tile<32xi32, UB>
  return %1 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func @fold_cast_i8_to_f32(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi16, UB>
// CHECK-NEXT:  %1 = asctile.cast %0 : !asctile.tile<32xi16, UB> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @fold_cast_i8_to_f32(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi16, UB>
  %1 = asctile.cast %0 : !asctile.tile<32xi16, UB> to !asctile.tile<32xf16, UB>
  %2 = asctile.cast %1 : !asctile.tile<32xf16, UB> to !asctile.tile<32xf32, UB>
  return %2 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func @fold_cast_f32_to_i32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.cast %arg0 : !asctile.tile<32xf32, UB> to !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func @fold_cast_f32_to_i32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xf32, UB> to !asctile.tile<32xf16, UB>
  %1 = asctile.cast %0 : !asctile.tile<32xf16, UB> to !asctile.tile<32xi32, UB>
  return %1 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func @no_fold_unsupported_cast(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xi64, UB> {
// CHECK-NEXT:  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi32, UB>
// CHECK-NEXT:  %1 = asctile.cast %0 : !asctile.tile<32xi32, UB> to !asctile.tile<32xi64, UB>
// CHECK-NEXT:  return %1 : !asctile.tile<32xi64, UB>
// CHECK-NEXT:}
func.func @no_fold_unsupported_cast(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xi64, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi32, UB>
  %1 = asctile.cast %0 : !asctile.tile<32xi32, UB> to !asctile.tile<32xi64, UB>
  return %1 : !asctile.tile<32xi64, UB>
}

// CHECK-LABEL: func.func @no_fold_non_chained_cast(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func @no_fold_non_chained_cast(%arg0: !asctile.tile<32xi8, UB>) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi16, UB>
  %1 = asctile.cast %arg0 : !asctile.tile<32xi8, UB> to !asctile.tile<32xi32, UB>
  return %1 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func @supported_single_cast_i32_to_i16(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi16, UB> {
// CHECK-NEXT:  %0 = asctile.cast %arg0 : !asctile.tile<32xi32, UB> to !asctile.tile<32xi16, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi16, UB>
// CHECK-NEXT:}
func.func @supported_single_cast_i32_to_i16(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi16, UB> {
  %0 = asctile.cast %arg0 : !asctile.tile<32xi32, UB> to !asctile.tile<32xi16, UB>
  return %0 : !asctile.tile<32xi16, UB>
}
