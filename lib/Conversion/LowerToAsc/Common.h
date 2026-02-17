/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * */

#ifndef LIB_CONVERSION_LOWERTOASC_COMMON_H
#define LIB_CONVERSION_LOWERTOASC_COMMON_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>
#include <numeric>

namespace mlir {
namespace asclower {

constexpr unsigned ubBlockSize = 32;      // In bytes
constexpr unsigned repeatBlockSize = 256; // In bytes
constexpr unsigned bitmaskSize = 64;
constexpr int64_t dstBlkStride = 1;
constexpr int64_t src0BlkStride = 1;
constexpr int64_t src1BlkStride = 1;
constexpr int64_t dstRepStride = 8;
constexpr int64_t src0RepStride = 8;
constexpr int64_t src1RepStride = 8;

inline int64_t getRepeatTimes(const ShapedType &type)
{
    auto sizeType = type.getElementType().getIntOrFloatBitWidth() / CHAR_BIT;
    if (sizeType != 2 && sizeType != 4)
        assert("Unsupported element type");
    auto numElemsPerRepeat = repeatBlockSize / sizeType;
    return llvm::divideCeil(type.getNumElements(), numElemsPerRepeat);
}

inline std::pair<uint64_t, uint64_t> getMask(const ShapedType &type)
{
    auto sizeType = type.getElementTypeBitWidth() / CHAR_BIT;
    if (sizeType != 2 && sizeType != 4)
        assert("Unsupported element type");
    auto mask = llvm::maskTrailingOnes<uint64_t>(bitmaskSize);
    if (sizeType == 2)
        return {mask, mask};
    return {0ul, mask};
}

struct I1ReplacementType {
    static inline constexpr unsigned width = 16U;
    IntegerType type;

    explicit I1ReplacementType(MLIRContext *context) : type(IntegerType::get(context, width)) {}
    ~I1ReplacementType() = default;

    static constexpr uint16_t max()
    {
        constexpr uint16_t base = 1U;
        unsigned shift = width - 1U;
        return (base << shift) | ~(base << shift);
    }
};

struct LoweringTypeConverter : public TypeConverter {
    static Value addUnrealizedCast(OpBuilder &builder, Type type, ValueRange inputs, Location loc)
    {
        auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        return cast.getResult(0);
    }

    LoweringTypeConverter()
    {
        addConversion([](Type type) { return std::optional<Type> {type}; });
        addArgumentMaterialization(addUnrealizedCast);
        addSourceMaterialization(addUnrealizedCast);
        addTargetMaterialization(addUnrealizedCast);
    }
};

struct TensorTypeConverter : public LoweringTypeConverter {
    TensorTypeConverter() : LoweringTypeConverter()
    {
        addConversion([](asctile::TileType type) {
            auto elType = type.getElementType();
            SmallVector<int64_t> shape(type.getShape());
            if (elType.isInteger(1)) {
                I1ReplacementType replType(type.getContext());
                auto numElements = static_cast<int64_t>(llvm::divideCeil(type.getNumElements(), replType.width));
                shape = {numElements};
                elType = replType.type;
            }
            return ascendc::LocalTensorType::get(shape, elType);
        });
        addConversion([](asctile::TensorType type) {
            return ascendc::GlobalTensorType::get(type.getShape(), type.getElementType());
        });
        addConversion([](MemRefType type) -> Type {
            auto addrSpace = ascendc::AddressSpace::gm;
            if (auto attr = dyn_cast_if_present<IntegerAttr>(type.getMemorySpace()))
                addrSpace = static_cast<ascendc::AddressSpace>(attr.getValue().getSExtValue());
            if (addrSpace == ascendc::AddressSpace::Default)
                addrSpace = ascendc::AddressSpace::gm;
            if (addrSpace == ascendc::AddressSpace::gm)
                return ascendc::GlobalTensorType::get(type.getShape(), type.getElementType());
            return ascendc::LocalTensorType::get(type.getShape(), type.getElementType());
        });
    }
};

using ConvertRewriter = ConversionPatternRewriter;

template <typename OpType>
struct ConvertOp : public ConversionPattern {
    ConvertOp(LoweringTypeConverter &converter, MLIRContext *context)
        : ConversionPattern(converter, OpType::getOperationName(), 1, context)
    {}

    const LoweringTypeConverter &converter() const { return *getTypeConverter<LoweringTypeConverter>(); }

    virtual LogicalResult convert(OpType op, [[maybe_unused]] ArrayRef<Value> &operands,
                                  ConvertRewriter &rewriter) const
    {
        return convert(op, rewriter);
    }

    virtual LogicalResult convert(OpType op, ConvertRewriter &rewriter) const { return failure(); }

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConvertRewriter &rewriter) const final
    {
        return convert(cast<OpType>(op), operands, rewriter);
    }

    ascendc::LocalTensorAutoOp createTensorOp(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape,
                                              Type elementType) const
    {
        return builder.create<ascendc::LocalTensorAutoOp>(loc, ascendc::LocalTensorType::get(shape, elementType));
    }

    ascendc::LocalTensorAutoOp createTensorOp(OpBuilder &builder, Location loc, Type convertibleType) const
    {
        auto convertedType = converter().convertType(convertibleType);
        assert(isa<ascendc::LocalTensorType>(convertedType) && "must be convertible");
        auto tensorType = cast<ascendc::LocalTensorType>(convertedType);
        return createTensorOp(builder, loc, tensorType.getShape(), tensorType.getElementType());
    }

    ascendc::LocalTensorReinterpretCastOp createReCastOp(OpBuilder &builder, Location loc, Value convertibleTensor,
                                                         ArrayRef<int64_t> shape, Type elementType) const
    {
        auto type = ascendc::LocalTensorType::get(shape, elementType);
        auto tensor = converter().materializeTargetConversion(builder, loc, type, convertibleTensor);
        return builder.create<ascendc::LocalTensorReinterpretCastOp>(loc, type, tensor);
    }

    ascendc::LocalTensorReinterpretCastOp createReCastOp(OpBuilder &builder, Location loc, Value convertibleTensor,
                                                         Type convertibleType) const
    {
        auto convertedType = converter().convertType(convertibleType);
        assert(isa<ascendc::LocalTensorType>(convertedType) && "must be convertible");
        auto tensorType = cast<ascendc::LocalTensorType>(convertedType);
        return createReCastOp(builder, loc, convertibleTensor, tensorType.getShape(), tensorType.getElementType());
    }

    static int64_t calCount(Value tile) { return calCount(tile.getType()); }

    static int64_t calCount(Type type)
    {
        auto shaped = dyn_cast<ShapedType>(type);
        assert(shaped && shaped.hasStaticShape() && "must be ShapedType with static shape");
        return shaped.getNumElements();
    }
};

} // namespace asclower
} // namespace mlir

#endif // LIB_CONVERSION_LOWERTOASC_COMMON_H
