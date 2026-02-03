/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::asctile;

ParseResult parseShapeXType(AsmParser &odsParser, SmallVectorImpl<int64_t> &shape, Type &elementType);
void printShapeXType(AsmPrinter &odsPrinter, ArrayRef<int64_t> shape, Type elementType);

ParseResult parseTileLocation(AsmParser &odsParser, TileLocationAttr &loc);
void printTileLocation(AsmPrinter &odsPrinter, TileLocationAttr loc);

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom assembly format
//===----------------------------------------------------------------------===//

ParseResult parseShapeXType(AsmParser &odsParser, SmallVectorImpl<int64_t> &shape, Type &elementType)
{
    if (odsParser.parseOptionalStar()) {
        // No '*' consumed => shaped type is ranked (i.e. has shape)
        if (odsParser.parseDimensionList(shape)) {
            odsParser.emitError(odsParser.getNameLoc(), "either dimension list (ranked) or '*' symbol (unranked) must "
                                                        "be declared");
            return ParseResult::failure();
        }
    } else if (odsParser.parseXInDimensionList()) {
        return ParseResult::failure();
    }
    if (odsParser.parseType(elementType)) {
        return ParseResult::failure();
    }
    return ParseResult::success();
}

void printShapeXType(AsmPrinter &odsPrinter, ArrayRef<int64_t> shape, Type elementType)
{
    if (shape.empty()) {
        odsPrinter << "*x";
    } else {
        for (int64_t dim : shape) {
            if (ShapedType::isDynamic(dim))
                odsPrinter << "?";
            else
                odsPrinter << dim;
            odsPrinter << "x";
        }
    }
    odsPrinter << elementType;
}

ParseResult parseTileLocation(AsmParser &odsParser, TileLocationAttr &loc)
{
    StringRef name;
    if (odsParser.parseKeyword(&name))
        return ParseResult::failure();
    auto maybeLoc = symbolizeTileLocation(name);
    if (!maybeLoc)
        return ParseResult::failure();
    loc = TileLocationAttr::get(odsParser.getContext(), *maybeLoc);
    return ParseResult::success();
}

void printTileLocation(AsmPrinter &odsPrinter, TileLocationAttr loc)
{
    odsPrinter << stringifyTileLocation(loc.getValue());
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

using AscTensorType = mlir::asctile::TensorType;

AscTensorType AscTensorType::get(ArrayRef<int64_t> shape, Type elementType)
{
    return AscTensorType::get(elementType.getContext(), shape, elementType);
}

ShapedType AscTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const
{
    if (shape)
        return AscTensorType::get(*shape, elementType);
    return AscTensorType::get(getShape(), elementType);
}

bool AscTensorType::hasRank() const
{
    return !getShape().empty();
}

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

TileType TileType::get(ArrayRef<int64_t> shape, Type elementType, TileLocation loc)
{
    auto *ctx = elementType.getContext();
    return TileType::get(ctx, shape, elementType, TileLocationAttr::get(ctx, loc));
}

ShapedType TileType::cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const
{
    if (shape)
        return TileType::get(*shape, elementType, getLoc());
    return TileType::get(getShape(), elementType, getLoc());
}

bool TileType::hasRank() const
{
    return !getShape().empty();
}

TileLocation TileType::getLoc() const
{
    return getLocAttr().getValue();
}

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

void AscTileDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "ascir/Dialect/AscTile/IR/AscTileTypes.cpp.inc"
        >();
}
