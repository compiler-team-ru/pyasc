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
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTBUFIDSYNCV2
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using VisitedOpsMap = DenseMap<Operation *, int32_t>;
using VisitedBufIdMap = DenseMap<Operation *, std::set<int32_t>>;

ascendc::Pipe getPipe(Operation *op)
{
    ascendc::Pipe pipe = ascendc::Pipe::PIPE_V;
    if (isa<ascendc::LoadDataG2LOp>(op)) {
        pipe = ascendc::Pipe::PIPE_MTE1;
    } else if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(op)) {
        auto direction = copyOp.getDirection();
        if (direction == ascendc::CopyDirection::gm_ubuf) {
            pipe = ascendc::Pipe::PIPE_MTE2;
        } else if (direction == ascendc::CopyDirection::ubuf_gm) {
            pipe = ascendc::Pipe::PIPE_MTE3;
        }
    } else if (isa<ascendc::FixpipeOp>(op)) {
        pipe = ascendc::Pipe::PIPE_FIX;
    } else if (isa<ascendc::LocalTensorGetValueOp, ascendc::LocalTensorSetValueOp>(op)) {
        pipe = ascendc::Pipe::PIPE_S;
    } else if (isa<ascendc::MmadOp>(op)) {
        pipe = ascendc::Pipe::PIPE_M;
    }
    return pipe;
}

void insertGetRlsBuf(Operation *op, int32_t bufId)
{
    OpBuilder builder(op);
    ascendc::Pipe pipe = getPipe(op);
    builder.create<ascendc::GetBufOp>(op->getLoc(), pipe, bufId, false);
    builder.setInsertionPointAfter(op);
    builder.create<ascendc::RlsBufOp>(op->getLoc(), pipe, bufId, false);
}

Operation *getInitialLocalTensor(Operation *op)
{
    Operation *defOp = op;
    if (isa<ascendc::LocalTensorV3Op, ascendc::TBufGetTensorOp>(defOp))
        return defOp;
    if (auto rCastOp = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(op)) {
        defOp = rCastOp.getIn().getDefiningOp();
    }
    if (auto subTensorOp = dyn_cast<ascendc::LocalTensorSubIndexOp>(op)) {
        defOp = subTensorOp.getTensor().getDefiningOp();
    }
    return getInitialLocalTensor(defOp);
}

class InsertBufIdSyncV2Pass : public ascendc::impl::InsertBufIdSyncV2Base<InsertBufIdSyncV2Pass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        MLIRContext *context = &getContext();
        VisitedOpsMap visitedOps;
        VisitedBufIdMap bufIdMap;
        int32_t bufId = 0;
        funcOp.walk([&visitedOps, &bufIdMap, &bufId](Operation *op) {
            auto copyOp = dyn_cast<ascendc::DataCopyOp>(op);
            if (copyOp && copyOp.getDirection() == ascendc::CopyDirection::ubuf_gm) {
                auto src = copyOp.getSrc().getDefiningOp();
                auto srcBufId = visitedOps[src];
                bufIdMap[op].insert(srcBufId);
                insertGetRlsBuf(op, srcBufId);
                return;
            }
            if (auto opWithDst = dyn_cast<ascendc::OpWithDst>(op)) {
                auto dstTensor = opWithDst.getDst();
                auto dstDefOp = dstTensor.getDefiningOp();
                dstDefOp = getInitialLocalTensor(dstDefOp);
                if (visitedOps.count(dstDefOp)) {
                    auto dstBufId = visitedOps[dstDefOp];
                    bufIdMap[op].insert(dstBufId);
                    insertGetRlsBuf(opWithDst, dstBufId);
                } else {
                    visitedOps[dstDefOp] = bufId;
                    bufIdMap[op].insert(bufId);
                    insertGetRlsBuf(opWithDst, bufId);
                    bufId++;
                }
            }
            if (auto opWithSrc = dyn_cast<ascendc::OpWithSrc>(op)) {
                for (auto &src : opWithSrc.getSrcTensors()) {
                    auto srcDefOp = getInitialLocalTensor(src.getDefiningOp());
                    auto srcBufId = visitedOps[srcDefOp];
                    if (!bufIdMap[op].count(srcBufId)) {
                        bufIdMap[op].insert(srcBufId);
                        insertGetRlsBuf(op, srcBufId);
                    }
                }
            }
            return;
        });
        Builder builder(&getContext());
        for (const auto &[op, bufIdSet] : bufIdMap) {
            SmallVector<int32_t> bufIdVec(bufIdSet.begin(), bufIdSet.end());
            op->setAttr(ascendc::attr::bufId, builder.getI32ArrayAttr(bufIdVec));
        }
        funcOp.walk([](scf::ForOp forOp) {
            if (forOp->hasAttrOfType<UnitAttr>(asctile::attr::parallel)) {
                forOp->removeAttr(asctile::attr::parallel);
                return;
            }
            auto terminator = forOp.getBody()->getTerminator();
            OpBuilder builder(terminator);
            ascir::ConstantOpBuilder consts(builder);
            auto const0 = consts.i32(0);
            builder.create<ascendc::SetFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE2, const0);
            builder.create<ascendc::SetFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE1, const0);
            builder.create<ascendc::WaitFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE2, const0);
            builder.create<ascendc::WaitFlagOp>(terminator->getLoc(), ascendc::HardEvent::MTE3_MTE1, const0);
        });
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createInsertBufIdSyncV2Pass()
{
    return std::make_unique<InsertBufIdSyncV2Pass>();
}
} // namespace ascendc
} // namespace mlir
