/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscVF/IR/AscVF.h"
#include "asctile/Dialect/AscVF/Transforms/Passes.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_DISPATCHVFFUSION
#include "asctile/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

void insertBarrierOp(func::FuncOp funcOp)
{
    funcOp->walk([](emitasc::VecScopeOp vecScopeOp) {
        OpBuilder builder(vecScopeOp.getContext());
        for (auto& op : vecScopeOp.getBody()->without_terminator()) {
            builder.setInsertionPoint(&op);
            builder.create<ascvf::BarrierOp>(builder.getUnknownLoc());
        }
    });
}

void addPasses(mlir::OpPassManager& pm, int repeatTimes)
{
    pm.addPass(ascvf::createReorderOpsInVecScopePass());
    for (int i = 0; i < repeatTimes; ++i) {
        pm.addPass(ascvf::createFuseVFForPass());
        pm.addPass(ascvf::createReorderOpsInVecScopePass());
    }
    pm.addPass(ascvf::createEliminateCommonMaskPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    for (int i = 0; i < repeatTimes; ++i) {
        pm.addPass(ascvf::createEliminateDataTransferPass());
        pm.addPass(createLoopInvariantCodeMotionPass());
    }
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
}

int calculateCanonicalizationSteps(func::FuncOp funcOp)
{
    int maxNested = 0;
    funcOp->walk([&](ascvf::VFGroupOp vfGroupOp) {
        std::map<emitasc::VFForOp, int> nestedLevel;
        vfGroupOp->walk<WalkOrder::PreOrder>([&](emitasc::VFForOp forOp) {
            if (auto parent = forOp->getParentOfType<emitasc::VFForOp>()) {
                int level = nestedLevel[parent] + 1;
                nestedLevel[forOp] = level;
                maxNested = std::max(maxNested, level);
            } else {
                nestedLevel[forOp] = 0;
            }
        });
    });
    return maxNested + 1;
}

struct DispatchVFFusionPass : public ascvf::impl::DispatchVFFusionBase<DispatchVFFusionPass> {
    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();
        auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
        moduleOp->walk([&](func::FuncOp funcOp) {
            int repeatTimes = calculateCanonicalizationSteps(funcOp);
            OpPassManager pipeline(func::FuncOp::getOperationName(), OpPassManager::Nesting::Explicit);
            addPasses(pipeline, repeatTimes);

            insertBarrierOp(funcOp);
            IRMapping mapping;
            auto cloned = funcOp.clone(mapping);
            builder.insert(cloned);
            cloned.setName((funcOp.getName() + "_cloned").str());
            llvm::DenseMap<Operation*, Operation*> link;
            funcOp->walk([&](ascvf::BarrierOp barrierOp) {
                Operation* origOp = barrierOp.getOperation();
                link[mapping.lookup(origOp)] = origOp;
            });
            if (runPipeline(pipeline, cloned).failed()) {
                signalPassFailure();
                return;
            }

            OpPassManager eraseBarrierPass(func::FuncOp::getOperationName(), OpPassManager::Nesting::Explicit);
            eraseBarrierPass.addPass(ascvf::createEraseBarrierPass());
            if (runPipeline(eraseBarrierPass, cloned).failed()) {
                signalPassFailure();
                return;
            }
            std::set<Operation*> existBarriers;
            cloned->walk([&](ascvf::BarrierOp barrierOp) { existBarriers.insert(link[barrierOp]); });
            funcOp->walk([&](ascvf::BarrierOp barrierOp) {
                if (existBarriers.count(barrierOp)) {
                    OpBuilder builder(barrierOp);
                    builder.create<ascendc::LocalMemBarOp>(
                        builder.getUnknownLoc(), ascendc::MemType::VEC_STORE, ascendc::MemType::VEC_LOAD);
                }
                barrierOp.erase();
            });
            cloned.erase();

            if (runPipeline(pipeline, funcOp).failed())
                signalPassFailure();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createDispatchVFFusionPass() { return std::make_unique<DispatchVFFusionPass>(); }
