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
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/EmitAsc/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTINITDUMP
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

struct InsertInitDumpPass : public ascendc::impl::InsertInitDumpBase<InsertInitDumpPass> {
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        auto walk = funcOp.walk([](Operation* op) {
            if (isa<ascendc::PrintfOp, ascendc::DumpTensorOp>(op))
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
        if (!walk.wasInterrupted())
            return;
        auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
        ascir::ConstantOpBuilder consts(builder);
        auto moduleOp = ascendc::getModule(funcOp);
        auto isMixedVal = false;
        if (auto attr = moduleOp->getAttrOfType<StringAttr>(ascendc::attr::kernelType))
            isMixedVal = attr.getValue() == ascendc::attr::kernelMixed;
        auto isMixed = consts.i1(isMixedVal);
        auto dumpSize = consts.i32(1024 * 1024); // In bytes for each core
        auto loc = builder.getUnknownLoc();
        auto gmAs = builder.getI64IntegerAttr(static_cast<int64_t>(ascendc::AddressSpace::gm));
        auto dumpAddrType = MemRefType::get(ShapedType::kDynamic, builder.getIntegerType(8, false), AffineMap(), gmAs);
        NamedAttribute kernelArg(
            builder.getStringAttr(emitasc::attr::kernelArg),
            builder.getAttr<emitasc::KernelArgumentAttr>(emitasc::KernelArgument::DumpAddr));
        auto idx = funcOp.getNumArguments();
        funcOp.insertArgument(
            idx, dumpAddrType, builder.getDictionaryAttr(kernelArg), NameLoc::get(builder.getStringAttr("dump_addr")));
        auto dumpAddr = funcOp.getArgument(idx);
        builder.create<ascendc::InitDumpOp>(loc, isMixed, dumpAddr, dumpSize);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertInitDumpPass() { return std::make_unique<InsertInitDumpPass>(); }
