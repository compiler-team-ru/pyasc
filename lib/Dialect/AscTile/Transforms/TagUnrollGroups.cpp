/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_TAGUNROLLGROUPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

template <typename OpT>
void tagOps(Operation *root, int64_t &nextIndex)
{
    Builder builder(root);
    root->walk([&](OpT op) { op->setAttr(attr::unrollGroup, builder.getI64IntegerAttr(nextIndex++)); });
}

struct TagUnrollGroupsPass : public asctile::impl::TagUnrollGroupsBase<TagUnrollGroupsPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        int64_t i = 0;
        op.walk([&i](scf::ForOp loop) {
            if (!loop->hasAttrOfType<IntegerAttr>(attr::unrollFactor))
                return;
            tagOps<asctile::LoadOp>(loop, i);
            tagOps<asctile::StoreOp>(loop, i);
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createTagUnrollGroupsPass()
{
    return std::make_unique<TagUnrollGroupsPass>();
}
