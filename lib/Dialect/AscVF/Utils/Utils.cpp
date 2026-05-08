/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscVF/Utils/Utils.h"

namespace mlir {
namespace ascvf {
ValueVector deduplicate(ArrayRef<Value> values)
{
    ValueVector result;
    ValueSet unique(values.begin(), values.end());
    for (auto value : values) {
        auto it = unique.find(value);
        if (it == unique.end())
            continue;
        result.push_back(value);
        unique.erase(it);
    }
    return result;
}
} // namespace ascvf
} // namespace mlir
