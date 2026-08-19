# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc2

all_dtypes = (asc2.int8, asc2.int16, asc2.int32, asc2.int64, asc2.float16, asc2.bfloat16, asc2.float32, asc2.float64)
all_locations = (asc2.TensorLocation.BT, asc2.TensorLocation.FIX, asc2.TensorLocation.L0A, asc2.TensorLocation.L0B,
                 asc2.TensorLocation.L0C, asc2.TensorLocation.L1, asc2.TensorLocation.UB)
non_ub_locations = tuple(loc for loc in all_locations if loc != asc2.TensorLocation.UB)
non_ub_l0c_locations = tuple(loc for loc in non_ub_locations if loc != asc2.TensorLocation.L0C)
