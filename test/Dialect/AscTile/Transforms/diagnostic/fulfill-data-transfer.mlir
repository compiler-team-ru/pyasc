// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-fulfill-data-transfer -split-input-file -verify-diagnostics %s

// expected-note@+1 {{source tensor with L0A location defined here}}
func.func @invalid_copy_l0a_to_ub(%arg0: tensor<16x16xf32, #asctile.local<L0A>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{Direct data transfer from L0A to UB is not supported}}
  %0 = asctile.copy %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0A>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}
