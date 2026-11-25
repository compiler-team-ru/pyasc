/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscTile/Utils/Attributes.h"
#include "asctile/Dialect/AscTile/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MARKMATMULSCALE
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

struct MarkMatmulScalePass : public asctile::impl::MarkMatmulScaleBase<MarkMatmulScalePass> {
    void runOnOperation() override
    {
        getOperation().walk([](Operation* op) {
            llvm::TypeSwitch<Operation*>(op).Case<asctile::MatmulOp, asctile::MatmulAccOp>([](auto op) {
                auto bindA =
                    asctile::searchDefiningOp<asctile::CopyScaleOp, tensor::CastOp, arith::BitcastOp>(op.getMatrixA());
                auto bindB =
                    asctile::searchDefiningOp<asctile::CopyScaleOp, tensor::CastOp, arith::BitcastOp>(op.getMatrixB());
                if (bindA && bindB) {
                    op->setAttr(asctile::attr::mmadScale, UnitAttr::get(op.getContext()));
                    (*bindA)->setAttr(asctile::attr::isMatrixA, UnitAttr::get(op.getContext()));
                }
            });
            return;
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMarkMatmulScalePass() { return std::make_unique<MarkMatmulScalePass>(); }
