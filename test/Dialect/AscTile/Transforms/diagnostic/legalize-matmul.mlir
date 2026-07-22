// RUN: ascir-opt -asctile-legalize-matmul -split-input-file -verify-diagnostics %s

func.func @valid_matmul_acc(%arg0: tensor<8x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x8xf16, #asctile.local<L0B>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  %acc = asctile.accumulator : tensor<8x8xf32, #asctile.local<L0C>>
  asctile.matmul_acc %acc, %arg0, %arg1 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf16, #asctile.local<L0A>>, tensor<16x8xf16, #asctile.local<L0B>>
  return %acc : tensor<8x8xf32, #asctile.local<L0C>>
}

// -----

func.func @invalid_matmul_acc(%arg0: tensor<8x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x8xf16, #asctile.local<L0B>>, %arg2: tensor<8x8xf32, #asctile.local<L0C>>) -> tensor<8x8xf32, #asctile.local<L0C>> {
  // expected-error@below {{Incorrect use of accumulator in matmul operation.}}
  asctile.matmul_acc %arg2, %arg0, %arg1 : tensor<8x8xf32, #asctile.local<L0C>>, tensor<8x16xf16, #asctile.local<L0A>>, tensor<16x8xf16, #asctile.local<L0B>>
  return %arg2 : tensor<8x8xf32, #asctile.local<L0C>>
}
