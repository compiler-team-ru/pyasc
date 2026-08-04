// RUN: ascir-opt -ascendc-insert-subblock-guard -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @kernel_void
// CHECK:       emitc.verbatim "if (AscendC::GetSubBlockIdx() != 0) return;"
// CHECK:       return
module attributes {asc.kernel_type = "mixed"} {
func.func @kernel_void() {
  return
}
}

// -----

// CHECK-LABEL: func.func @vector_kernel
// CHECK-NOT:   emitc.verbatim
// CHECK:       return
module attributes {asc.kernel_type = "vector"} {
func.func @vector_kernel() {
  return
}
}
