/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_LEGALIZEMATMUL
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

constexpr const char* constZeroAcc = "const_zero_acc";
constexpr const char* unprocessedMatmul = "unprocessed_matmul";

bool checkZeroTile(arith::ConstantOp op)
{
    auto value = op.getValue();
    return matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat());
}

bool processMatmulAcc(asctile::MatmulOp matmulOp)
{
    matmulOp->removeAttr(unprocessedMatmul);
    auto acc = matmulOp.getAcc();
    if (!acc)
        return true;
    if (auto constOp = acc.getDefiningOp<arith::ConstantOp>()) {
        if (checkZeroTile(constOp)) {
            OpBuilder builder(matmulOp);
            auto emptyOp = builder.create<asctile::EmptyOp>(matmulOp.getLoc(), matmulOp.getType());
            matmulOp.getAccMutable().assign(emptyOp);
            return true;
        }
    } else if (auto prevMatmul = acc.getDefiningOp<asctile::MatmulOp>()) {
        // TODO: Add support for bias
        return false;
    } else if (auto blockArg = dyn_cast<BlockArgument>(acc)) {
        auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
        if (!forOp)
            return false;
        auto* loopInit = forOp.getTiedLoopInit(blockArg);
        auto* loopYield = forOp.getTiedLoopYieldedValue(blockArg);
        if (loopYield->get() != matmulOp.getResult())
            return false;
        auto constOp = loopInit->get().getDefiningOp<arith::ConstantOp>();
        if (!constOp || !checkZeroTile(constOp))
            return false;
        matmulOp->setAttr(constZeroAcc, UnitAttr::get(matmulOp->getContext()));
        OpBuilder builder(matmulOp);
        IRMapping mapper;
        SmallVector<Value> newInits(forOp.getInitArgs());
        builder.setInsertionPointAfter(constOp);
        auto emptyOp = builder.create<asctile::EmptyOp>(constOp.getLoc(), matmulOp.getType());
        newInits.push_back(emptyOp);
        builder.setInsertionPointAfter(forOp);
        auto newForOp = builder.create<scf::ForOp>(
            forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), newInits);
        auto* oldForBody = forOp.getBody();
        auto* newForBody = newForOp.getBody();
        newForOp->setAttrs(forOp->getAttrs());
        mapper.map(oldForBody->getArguments(), newForBody->getArguments());
        builder.setInsertionPointToEnd(newForBody);
        for (auto& op : oldForBody->without_terminator()) {
            builder.clone(op, mapper);
        }
        auto mappedMatmulValue = mapper.lookup(blockArg);
        auto newIterArg = newForBody->getArguments().back();
        Value newYieldValue = newIterArg;
        newForBody->walk([&newIterArg, &newYieldValue](asctile::MatmulOp op) {
            if (!op->hasAttrOfType<UnitAttr>(constZeroAcc))
                return;
            op.getAccMutable().assign(newIterArg);
            op->removeAttr(constZeroAcc);
            newYieldValue = op.getResult();
        });
        auto oldYield = cast<scf::YieldOp>(oldForBody->getTerminator());
        SmallVector<Value> newYieldValues;
        for (auto val : oldYield.getResults()) {
            newYieldValues.push_back(mapper.lookup(val));
        }
        newYieldValues.push_back(newYieldValue);
        builder.setInsertionPointToEnd(newForBody);
        builder.create<scf::YieldOp>(newForOp.getLoc(), newYieldValues);
        auto oldRes = forOp.getTiedLoopResult(blockArg);
        for (unsigned i = 0; i < forOp.getNumResults(); i++) {
            auto newRes = (forOp.getResult(i) != oldRes) ? newForOp.getResult(i) : newForOp.getResults().back();
            oldRes.replaceAllUsesWith(newRes);
        }
        forOp.erase();
        return true;
    }
    return false;
}

bool checkCorrectMatmulAcc(func::FuncOp funcOp)
{
    bool correct = true;
    funcOp.walk([](asctile::MatmulOp matmulOp) {
        if (matmulOp.getAcc())
            matmulOp->setAttr(unprocessedMatmul, UnitAttr::get(matmulOp->getContext()));
    });
    do {
        auto result =
            funcOp.walk(
                [&correct](asctile::MatmulOp matmulOp) {
                    if (!matmulOp->hasAttrOfType<UnitAttr>(unprocessedMatmul))
                        return WalkResult::advance();
                    correct &= processMatmulAcc(matmulOp);
                    if (!correct) {
                        matmulOp.emitError()
                            << "Incorrect use of accumulator in Matmul operation. Result should be the same as "
                               "accumulation argument.";
                    }
                    return WalkResult::interrupt();
                });
        if (!result.wasInterrupted())
            break;
    } while (true);
    return correct;
}

struct LegalizeMatmulPass : public asctile::impl::LegalizeMatmulBase<LegalizeMatmulPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (!checkCorrectMatmulAcc(funcOp))
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::asctile::createLegalizeMatmulPass() { return std::make_unique<LegalizeMatmulPass>(); }
