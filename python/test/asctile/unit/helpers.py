# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asctile

all_dtypes = (asctile.int8, asctile.int16, asctile.int32, asctile.int64, asctile.float16, asctile.bfloat16,
              asctile.float32, asctile.float64)
all_locations = (asctile.TensorLocation.BT, asctile.TensorLocation.FIX, asctile.TensorLocation.L0A,
                 asctile.TensorLocation.L0B, asctile.TensorLocation.L0C, asctile.TensorLocation.L1,
                 asctile.TensorLocation.UB)
non_ub_locations = tuple(loc for loc in all_locations if loc != asctile.TensorLocation.UB)
non_ub_l0c_locations = tuple(loc for loc in non_ub_locations if loc != asctile.TensorLocation.L0C)
