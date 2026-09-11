/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_EXTENSION_PYOPBUILDER_H
#define ASCIR_EXTENSION_PYOPBUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <optional>
#include <string>

namespace mlir {
namespace ascir {

class PyOpBuilder {
    OpBuilder builder;
    Location loc;

public:
    explicit PyOpBuilder(mlir::MLIRContext* context) : builder(context), loc(builder.getUnknownLoc()) {}
    explicit PyOpBuilder(mlir::Operation* op) : builder(op), loc(op->getLoc()) {}
    ~PyOpBuilder() = default;

    void setLoc(mlir::Location newLoc) { loc = newLoc; }

    void setLoc(const std::string& name, bool reset = false)
    {
        if (reset) {
            setLoc(mlir::NameLoc::get(builder.getStringAttr(name)));
        } else {
            setLoc(mlir::NameLoc::get(builder.getStringAttr(name), loc));
        }
    }

    void setLoc(const std::string& fileName, int line, int column, const std::optional<std::string>& name)
    {
        mlir::Location newLoc = mlir::FileLineColLoc::get(builder.getContext(), fileName, line, column);
        if (name) {
            newLoc = mlir::NameLoc::get(builder.getStringAttr(*name), newLoc);
        }
        setLoc(newLoc);
    }

    mlir::Location getLoc() { return loc; }

    void resetLoc() { loc = builder.getUnknownLoc(); }

    mlir::OpBuilder& getBuilder() { return builder; }

    mlir::OpBuilder* operator->() { return &builder; }

    void setInsertionPointToStart(mlir::Block& block)
    {
        if (!block.empty()) {
            setLoc(block.begin()->getLoc());
        } else {
            resetLoc();
        }
        builder.setInsertionPointToStart(&block);
    }

    void setInsertionPointToEnd(mlir::Block& block)
    {
        if (!block.empty()) {
            setLoc(block.back().getLoc());
        } else {
            resetLoc();
        }
        builder.setInsertionPointToEnd(&block);
    }

    void setInsertionPointAfter(mlir::Operation& op)
    {
        setLoc(op.getLoc());
        builder.setInsertionPointAfter(&op);
    }

    void restoreInsertionPoint(mlir::OpBuilder::InsertPoint pt)
    {
        if (pt.isSet() && pt.getPoint() != pt.getBlock()->end()) {
            setLoc(pt.getPoint()->getLoc());
        } else if (pt.isSet() && !pt.getBlock()->empty()) {
            setLoc(pt.getBlock()->back().getLoc());
        } else {
            resetLoc();
        }
        builder.restoreInsertionPoint(pt);
    }

    mlir::Operation* create(
        llvm::StringRef operationName, mlir::ValueRange operands, mlir::TypeRange types = {},
        llvm::ArrayRef<mlir::NamedAttribute> attributes = {})
    {
        return builder.create(loc, builder.getStringAttr(operationName), operands, types, attributes);
    }

    template <typename OpTy, typename... Args>
    auto create(Args&&... args) -> OpTy
    {
        return builder.create<OpTy>(loc, std::forward<Args>(args)...);
    }

    template <typename OpTy, typename... Args>
    std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(), mlir::Value> createOrFold(Args&&... args)
    {
        return builder.createOrFold<OpTy>(loc, std::forward<Args>(args)...);
    }

    template <typename OpTy, typename... Args>
    std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy> createOrFold(Args&&... args)
    {
        return builder.createOrFold<OpTy>(loc, std::forward<Args>(args)...);
    }

    std::optional<mlir::func::FuncOp> getCurrentFunction()
    {
        mlir::Block* block = builder.getInsertionBlock();
        if (!block) {
            return std::nullopt;
        }
        mlir::Operation* parent = block->getParentOp();
        if (!parent) {
            return std::nullopt;
        }
        if (auto op = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
            return op;
        }
        if (auto op = parent->getParentOfType<mlir::func::FuncOp>()) {
            return op;
        }
        return std::nullopt;
    }
};

} // namespace ascir
} // namespace mlir

#endif // ASCIR_EXTENSION_PYOPBUILDER_H
