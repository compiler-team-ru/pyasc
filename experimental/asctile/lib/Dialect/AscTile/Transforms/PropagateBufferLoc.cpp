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
#include "asctile/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_PROPAGATEBUFFERLOC
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

struct PropagateBufferLocPass : public asctile::impl::PropagateBufferLocBase<PropagateBufferLocPass> {
    void runOnOperation() override
    {
        getOperation().walk([](tensor::CastOp op) {
            if (auto fused = op.getLoc()->findInstanceOf<FusedLocWith<asctile::BufferAttr>>())
                op.getSource().setLoc(fused);
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createPropagateBufferLocPass()
{
    return std::make_unique<PropagateBufferLocPass>();
}
