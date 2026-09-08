/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Transforms/Passes.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <deque>

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_ERASEBARRIER
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

template <typename T>
std::set<T> setDiff(const std::set<T>& setA, const std::set<T>& setB)
{
    std::set<T> result;
    std::set_difference(setA.begin(), setA.end(), setB.begin(), setB.end(), std::inserter(result, result.begin()));
    return result;
}

struct EraseBarrierPass : public ascvf::impl::EraseBarrierBase<EraseBarrierPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([&](ascvf::VecScopeOp vecScopeOp) {
            ValueMap<std::deque<Operation*>> useTensors;
            SmallVector<Operation*> ops;
            vecScopeOp.walk([&](Operation* op) {
                if (auto loadOp = dyn_cast<ascvf::LoadOp>(op)) {
                    Value tensor = loadOp.getSrcTensor();
                    useTensors[tensor].push_back(loadOp);
                }
                if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
                    Value tensor = storeOp.getDstTensor();
                    useTensors[tensor].push_back(storeOp);
                }
                if (auto barrierOp = dyn_cast<ascvf::BarrierOp>(op)) {
                    ops.push_back(barrierOp);
                }
            });
            for (auto& [tensor, uses] : useTensors) {
                while (!uses.empty() && isa<ascvf::LoadOp>(uses.front()))
                    uses.pop_front();
                while (!uses.empty() && isa<ascvf::StoreOp>(uses.back()))
                    uses.pop_back();
                llvm::copy(uses, std::back_inserter(ops));
            }
            DominanceInfo di;
            std::set<Operation*> activeStores, intersectedStores;
            ascvf::BarrierOp activeBarrier = nullptr;
            llvm::sort(ops, [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
            for (auto* op : ops) {
                if (auto loadOp = dyn_cast<ascvf::LoadOp>(op)) {
                    auto diff = setDiff<Operation*>(activeStores, intersectedStores);
                    if (!diff.empty()) {
                        loadOp.emitWarning("before op absent LocalMemBar");
                    }
                    intersectedStores.clear();
                    activeBarrier = nullptr;
                    activeStores.clear();
                }
                if (auto storeOp = dyn_cast<ascvf::StoreOp>(op)) {
                    activeStores.insert(storeOp);
                }
                if (auto barrierOp = dyn_cast<ascvf::BarrierOp>(op)) {
                    // greedily choose the last available barrier
                    if (activeBarrier) {
                        activeBarrier.erase();
                        activeBarrier = nullptr;
                    }
                    if (activeStores.empty()) {
                        barrierOp.erase();
                    } else {
                        activeBarrier = barrierOp;
                        intersectedStores = activeStores;
                    }
                }
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createEraseBarrierPass() { return std::make_unique<EraseBarrierPass>(); }
