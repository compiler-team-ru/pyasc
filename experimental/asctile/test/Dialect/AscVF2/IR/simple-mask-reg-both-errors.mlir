// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not asctile-opt %s 2>&1 | FileCheck %s

// CHECK: 'ascvf2.simple_mask_reg' op cannot specify both static and dynamic count
func.func @both_counts(%count: index) -> !ascendc.mask_reg {
  %mask = ascvf2.simple_mask_reg f32 static 1 dynamic %count : !ascendc.mask_reg
  return %mask : !ascendc.mask_reg
}
