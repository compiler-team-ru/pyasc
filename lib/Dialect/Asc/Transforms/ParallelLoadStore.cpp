/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_PARALLELLOADSTORE
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class UpdateGetRlsBuf {
    using NewBufIdOpsMap = std::unordered_map<Operation *, int32_t>;

    int32_t bufIdMax;
    std::set<Operation *> unusedOps;
    NewBufIdOpsMap newOps;

    void updateBufIdMax() { bufIdMax++; }

    void collectGetBufId(int32_t bufId, Operation *forOp)
    {
        forOp->walk([&](ascendc::GetBufOp op) {
            int32_t currentBufId = op.getBufId();
            if (currentBufId == bufId && op.getPipe() == ascendc::Pipe::PIPE_V) {
                newOps.insert({op, bufIdMax});
            }
        });
    }

    void collectRlsBufId(int32_t bufId, Operation *forOp)
    {
        forOp->walk([&](ascendc::RlsBufOp op) {
            int32_t currentBufId = op.getBufId();
            if (currentBufId == bufId) {
                auto pipe = op.getPipe();
                if (pipe == ascendc::Pipe::PIPE_V || pipe == ascendc::Pipe::PIPE_MTE3) {
                    newOps.insert({op, bufIdMax});
                    if (pipe == ascendc::Pipe::PIPE_MTE3) {
                        unusedOps.insert(op);
                    }
                }
            }
        });
    }

    void updateStoreOp(ascendc::GetBufOp op, scf::ForOp forOp)
    {
        int32_t bufId = op.getBufId();
        collectGetBufId(bufId, forOp);
        newOps.insert({op, bufIdMax});
        unusedOps.insert(op);
        collectRlsBufId(bufId, forOp);
    }

  public:
    explicit UpdateGetRlsBuf(int32_t bufIdMax = 0) : bufIdMax(bufIdMax) {}
    ~UpdateGetRlsBuf() = default;

    void process(scf::ForOp forOp)
    {
        forOp.walk([&](ascendc::GetBufOp op) {
            if (op.getPipe() == ascendc::Pipe::PIPE_MTE3) {
                updateBufIdMax();
                updateStoreOp(op, forOp);
            }
        });
    }

    void insertNewGetRlsBuf()
    {
        for (auto &[bufOp, bufId] : newOps) {
            OpBuilder builder(bufOp);
            if (auto getBufOp = dyn_cast<ascendc::GetBufOp>(bufOp)) {
                auto pipe = getBufOp.getPipe();
                if (pipe == ascendc::Pipe::PIPE_MTE3) {
                    getBufOp.setBufId(bufId);
                } else {
                    builder.create<ascendc::GetBufOp>(getBufOp->getLoc(), getBufOp.getPipe(), bufId, false);
                }
            } else if (auto rlsBufOp = dyn_cast<ascendc::RlsBufOp>(bufOp)) {
                auto pipe = rlsBufOp.getPipe();
                builder.setInsertionPointAfter(bufOp);
                if (pipe == ascendc::Pipe::PIPE_MTE3) {
                    rlsBufOp.setBufId(bufId);
                } else {
                    builder.create<ascendc::RlsBufOp>(rlsBufOp->getLoc(), rlsBufOp.getPipe(), bufId, false);
                }
            }
        }
    }
};

class ParallelLoadStorePass : public ascendc::impl::ParallelLoadStoreBase<ParallelLoadStorePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        int32_t bufIdMax = 0;
        funcOp.walk([&bufIdMax](ascendc::GetBufOp op) {
            int32_t bufId = op.getBufId();
            if (bufIdMax < bufId) {
                bufIdMax = bufId;
            }
        });

        UpdateGetRlsBuf updateGetRlsBuf(bufIdMax);
        funcOp.walk([&updateGetRlsBuf](scf::ForOp forOp) {
            if (!forOp->hasAttrOfType<UnitAttr>(asctile::attr::parallel))
                return;
            updateGetRlsBuf.process(forOp);
            forOp->removeAttr(asctile::attr::parallel);
        });
        updateGetRlsBuf.insertNewGetRlsBuf();
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createParallelLoadStorePass()
{
    return std::make_unique<ParallelLoadStorePass>();
}
} // namespace ascendc
} // namespace mlir
