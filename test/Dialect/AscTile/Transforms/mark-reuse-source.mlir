// RUN: ascir-opt --asctile-mark-reuse-source %s | FileCheck %s

// CHECK-LABEL: func.func public @test_mark
// CHECK: asctile.reduce <sum> %arg0
// CHECK-SAME: asctile.reuse_source
func.func public @test_mark(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %1 = asctile.reduce <sum> %arg0 {dims= [ 1 : i32]} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
  return %1 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @test_unmark
// CHECK: asctile.reduce <sum> %arg0 {dims = [1 : i32]} :
func.func public @test_unmark(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>) {
  %1 = asctile.reduce <sum> %arg0 {dims= [ 1 : i32]} : tensor<16x16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
  return %1, %arg0 : tensor<16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>
}
