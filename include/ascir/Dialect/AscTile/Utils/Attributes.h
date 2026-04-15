/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASCTILE_UTILS_ATTRIBUTES_H
#define ASCIR_DIALECT_ASCTILE_UTILS_ATTRIBUTES_H
#define LITERAL constexpr const char*

namespace mlir {
namespace asctile {
namespace attr {

LITERAL isMatrixA = "asctile.is_matrix_a";
LITERAL parallel = "asctile.parallel";
LITERAL unrollFactor = "asctile.unroll_factor";
LITERAL unrollGroup = "asctile.unroll_group";

} // namespace attr
} // namespace asctile
} // namespace mlir

#undef LITERAL
#endif // ASCIR_DIALECT_ASCTILE_UTILS_ATTRIBUTES_H
