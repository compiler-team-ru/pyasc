/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscendC/Transforms/Passes.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

#include <unordered_map>

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_REIFYBUFFERBYID
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using TensorOp = ascendc::LocalTensorAutoOp;
using ReCastOp = ascendc::LocalTensorReinterpretCastOp;

struct ReifyBufferByIdPass : public ascendc::impl::ReifyBufferByIdBase<ReifyBufferByIdPass> {
    void runOnOperation() override
    {
        std::unordered_map<int64_t, SmallVector<TensorOp, 4>> buffers;
        auto func = getOperation();
        func.walk([&](TensorOp op) {
            auto loc = op.getLoc()->findInstanceOf<FusedLocWith<asctile::BufferAttr>>();
            if (loc && loc.getLocations().size() == 1) {
                auto& tensors = buffers[loc.getMetadata().getId()];
                if (tensors.empty() || tensors.front().getPosition() == op.getPosition())
                    tensors.push_back(op);
            }
        });
        auto builder = OpBuilder::atBlockBegin(&func.getFunctionBody().front());
        for (const auto& tensors : llvm::make_second_range(buffers)) {
            if (tensors.size() < 2)
                continue;
            auto largest = *llvm::max_element(tensors, [](TensorOp lhs, TensorOp rhs) {
                return ascendc::getTypeSize(lhs.getType()) < ascendc::getTypeSize(rhs.getType());
            });
            Value base = builder.create<TensorOp>(largest->getLoc(), largest.getType(), largest.getPosition());
            for (auto tensor : tensors) {
                Value newOp;
                if (tensor.getType() == base.getType())
                    newOp = base;
                else
                    newOp = builder.create<ReCastOp>(tensor.getLoc(), tensor.getType(), base);
                tensor.replaceAllUsesWith(newOp);
                tensor.erase();
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createReifyBufferByIdPass() { return std::make_unique<ReifyBufferByIdPass>(); }
