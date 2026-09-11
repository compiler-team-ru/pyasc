/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "asctile/Dialect/AscVF/IR/AscVF.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace ascendc {

Pipe getOpPipeExt(Operation* op, Pipe defaultPipe)
{
    if (isa<ascvf::VFGroupOp>(op))
        return Pipe::PIPE_V;
    return getOpPipe(op, defaultPipe);
}

} // namespace ascendc
} // namespace mlir
