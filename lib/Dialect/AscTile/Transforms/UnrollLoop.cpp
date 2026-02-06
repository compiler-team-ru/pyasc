/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_UNROLLLOOP
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct UnrollLoopPass : public asctile::impl::UnrollLoopBase<UnrollLoopPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        op.walk([this](scf::ForOp loop) {
            int64_t unrollFactor = 0;
            if (auto a = loop->getAttrOfType<IntegerAttr>(attr::unrollFactor)) {
                unrollFactor = a.getValue().getSExtValue();
            }
            loop->removeAttr(attr::unrollFactor);
            if (unrollFactor <= 1)
                return;
            auto result = loopUnrollByFactor(loop, unrollFactor);
            if (failed(result))
                signalPassFailure();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createUnrollLoopPass()
{
    return std::make_unique<UnrollLoopPass>();
}
