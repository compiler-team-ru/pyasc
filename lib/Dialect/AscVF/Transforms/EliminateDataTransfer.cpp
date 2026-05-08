/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_ELIMINATEDATATRANSFER
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

SmallVector<SmallVector<Operation*>> collectLoadStoreOpsByBlock(ascvf::VFGroupOp groupOp)
{
    SmallVector<SmallVector<Operation*>> loadStoreGroups;
    groupOp.walk([&](Block* block) {
        SmallVector<Operation*> blockOps;
        for (auto& op : *block) {
            if (isa<ascvf::LoadMicroOp, ascvf::StoreMicroOp>(op)) {
                blockOps.emplace_back(&op);
            }
        }
        if (!blockOps.empty()) {
            loadStoreGroups.emplace_back(std::move(blockOps));
        }
    });
    return loadStoreGroups;
}

void eliminateRedundantLoadsAfterStores(ascvf::VFGroupOp groupOp)
{
    auto loadStoreGroups = collectLoadStoreOpsByBlock(groupOp);
    for (auto& blockOps : loadStoreGroups) {
        ValueMap<SmallVector<Operation*>> tensorToLoadStoreOps;
        for (auto* op : blockOps) {
            if (auto loadOp = dyn_cast<ascvf::LoadMicroOp>(op)) {
                tensorToLoadStoreOps[loadOp.getSrcTensor()].emplace_back(loadOp);
            } else if (auto storeOp = dyn_cast<ascvf::StoreMicroOp>(op)) {
                tensorToLoadStoreOps[storeOp.getDstTensor()].emplace_back(storeOp);
            }
        }
        for (auto& pair : tensorToLoadStoreOps) {
            Value lastStoredReg;
            for (auto* op : pair.second) {
                if (auto storeOp = dyn_cast<ascvf::StoreMicroOp>(op)) {
                    lastStoredReg = storeOp.getSrcReg();
                } else if (auto loadOp = dyn_cast<ascvf::LoadMicroOp>(op)) {
                    if (lastStoredReg) {
                        loadOp.getDstReg().replaceAllUsesWith(lastStoredReg);
                        loadOp->erase();
                    }
                }
            }
        }
    }
}

void eliminateOverwrittenStores(ascvf::VFGroupOp groupOp)
{
    ValueSet inputTensors;
    for (auto tensor : groupOp.getSrcList()) {
        inputTensors.insert(tensor);
    }
    ValueSet outputTensors;
    for (auto tensor : groupOp.getDstList()) {
        outputTensors.insert(tensor);
    }
    ValueMap<SmallVector<Operation*>> tensorToLoadStoreOps;
    groupOp.walk([&](Operation* op) {
        if (auto loadOp = dyn_cast<ascvf::LoadMicroOp>(op)) {
            tensorToLoadStoreOps[loadOp.getSrcTensor()].emplace_back(loadOp);
        } else if (auto storeOp = dyn_cast<ascvf::StoreMicroOp>(op)) {
            tensorToLoadStoreOps[storeOp.getDstTensor()].emplace_back(storeOp);
        }
    });
    llvm::DenseSet<Operation*> opsToDelete;
    for (auto& pair : tensorToLoadStoreOps) {
        auto& ops = pair.second;
        // Delete the last store if it's to a non-output tensor
        // (the value won't be used outside the VFGroupOp)
        if (auto lastStore = dyn_cast<ascvf::StoreMicroOp>(ops.back())) {
            if (!outputTensors.count(lastStore.getDstTensor())) {
                opsToDelete.insert(lastStore);
            }
        }
        // Erase overwritten stores
        bool hasSeenStore = false;
        for (auto* op : llvm::make_range(ops.rbegin(), ops.rend())) {
            if (auto storeOp = dyn_cast<ascvf::StoreMicroOp>(op)) {
                if (hasSeenStore) {
                    opsToDelete.insert(storeOp);
                }
                hasSeenStore = true;
            } else {
                hasSeenStore = false;
            }
        }
    }
    for (auto* op : opsToDelete) {
        op->erase();
    }
}

void mergeDuplicateLoadsFromSameAddress(ascvf::VFGroupOp groupOp)
{
    groupOp.walk([&](Block* block) {
        ValueMap<Value> tensorToFirstLoadedReg;
        for (auto& op : llvm::make_early_inc_range(*block)) {
            auto loadOp = dyn_cast<ascvf::LoadMicroOp>(op);
            if (!loadOp)
                continue;
            auto srcTensor = loadOp.getSrcTensor();
            if (tensorToFirstLoadedReg.count(srcTensor)) {
                loadOp.getDstReg().replaceAllUsesWith(tensorToFirstLoadedReg[srcTensor]);
                loadOp.erase();
            } else {
                tensorToFirstLoadedReg[srcTensor] = loadOp.getDstReg();
            }
        }
    });
}

struct EliminateDataTransferPass : public ascvf::impl::EliminateDataTransferBase<EliminateDataTransferPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VFGroupOp vfGroupOp) {
            eliminateRedundantLoadsAfterStores(vfGroupOp);
            eliminateOverwrittenStores(vfGroupOp);
            mergeDuplicateLoadsFromSameAddress(vfGroupOp);
        });
    }
};

} // namespace

namespace mlir {
namespace ascvf {
std::unique_ptr<Pass> createEliminateDataTransferPass() { return std::make_unique<EliminateDataTransferPass>(); }
} // namespace ascvf
} // namespace mlir
