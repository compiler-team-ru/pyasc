/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===----------------------------------------------------------------------===//
// ReuseTensorAllocation Pass
//===----------------------------------------------------------------------===//
//
// Reduces on-chip memory by reusing freed tensor allocations. Covers VECCALC (UB),
// A1 (L1), A2 (L0A), B2 (L0B), and CO1 (L0C) positions. Reuse is restricted to
// tensors of the same position, since each position maps to a distinct hardware buffer.
//
// Reuse decision:
// 1. Input/output tensor restriction: if both tensors are I/O and both have unroll_iter
//    attributes with different values, they cannot be reused (each iteration loads/stores
//    different data). If either tensor doesn't have unroll_iter, no restriction applies.
// 2. Lifetime overlap: top's lastRead must precede bottom's firstWrite in program order.
//    Uses opPrecedes with DominanceInfo to correctly handle operations in different blocks.
// 3. Same-op reuse: when top's lastRead and bottom's firstWrite are the same compute op,
//    hardware reads all inputs before writing the output — safe to share one allocation.
//
// User collection traverses CastOpInterface and LocalTensorSubIndexOp chains to find
// indirect users (ops that use a view of the allocation). View-producing ops themselves
// are skipped during read/write classification.
//
// Loop-carried values (scf.yield in ForOp/WhileOp, scf.condition in WhileOp) are excluded
// from the reuse list — their data persists across all iterations.
//
//===----------------------------------------------------------------------===//

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ascendc-reuse-tensor-allocation"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_REUSETENSORALLOCATION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr const char* const eraseMeAttr = "asc.erase_me";

using TensorOp = ascendc::LocalTensorAutoOp;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

void appendImplicitUsers(Operation* op, SmallVectorImpl<Operation*>& allUsers)
{
    for (auto* user : op->getUsers()) {
        if (isa<CastOpInterface>(user) || isa<ascendc::LocalTensorSubIndexOp>(user)) {
            auto users = user->getUsers();
            if (!users.empty()) {
                allUsers.append(users.begin(), users.end());
                appendImplicitUsers(user, allUsers);
            }
        }
    }
}

SmallVector<Operation*> collectAllUsers(TensorOp tensorOp)
{
    SmallVector<Operation*> users(tensorOp->getUsers());
    appendImplicitUsers(tensorOp, users);
    return users;
}

// Returns the actual allocation size matching AllocateTensor's sizing rules:
// cube-block-aligned for A1/A2/B2, raw byte size for VECCALC/CO1.
int64_t getAllocationSize(TensorOp op)
{
    auto shapedType = cast<ShapedType>(op.getType());
    auto pos = op.getPosition();
    if (pos == ascendc::TPosition::A1 || pos == ascendc::TPosition::A2 || pos == ascendc::TPosition::B2)
        return ascendc::getTypeSizeCubeBlockAlign(shapedType);
    return ascendc::getTypeSize(op.getType());
}

bool isTensorGreaterOrEqual(TensorOp lhs, TensorOp rhs) { return getAllocationSize(lhs) >= getAllocationSize(rhs); }

bool isReusablePosition(ascendc::TPosition pos)
{
    return pos == ascendc::TPosition::VECCALC || pos == ascendc::TPosition::A1 || pos == ascendc::TPosition::A2 ||
           pos == ascendc::TPosition::B2 || pos == ascendc::TPosition::CO1;
}

bool isReusable(TensorOp op) { return isReusablePosition(op.getPosition()) && op.getType().hasStaticShape(); }

void transferInOutFlags(TensorOp erasedTensor, TensorOp survivingTensor)
{
    if (erasedTensor.getInput())
        survivingTensor.setInput(true);
    if (erasedTensor.getOutput())
        survivingTensor.setOutput(true);
}

void markForErase(TensorOp op, Operation* newTensor)
{
    op->replaceAllUsesWith(newTensor);
    op->setAttr(eraseMeAttr, UnitAttr::get(op.getContext()));
}

std::optional<int64_t> getUnrollIter(Operation* op)
{
    while (op) {
        if (auto iter = op->getAttrOfType<IntegerAttr>(asctile::attr::unrollIter))
            return iter.getValue().getSExtValue();
        op = op->getParentOp();
    }
    return std::nullopt;
}

bool isLoopCarriedYield(Operation* user)
{
    if (isa<scf::YieldOp>(user)) {
        Operation* parent = user->getParentOp();
        return isa<scf::ForOp>(parent) || isa<scf::WhileOp>(parent);
    }
    if (isa<scf::ConditionOp>(user))
        return isa<scf::WhileOp>(user->getParentOp());
    return false;
}

bool isRebornInLoop(Operation* firstWrite, Operation* loopOp)
{
    if (!firstWrite)
        return false;
    // Conservative: uses firstWrite only. A tensor written both outside and inside
    // a loop is effectively re-born, but firstWrite (earliest write) may be outside.
    return loopOp->isAncestor(firstWrite);
}

TensorOp getAllocationRoot(Value v)
{
    auto* defOp = v.getDefiningOp();
    if (!defOp)
        return {};
    if (auto op = dyn_cast<TensorOp>(defOp))
        return op;
    if (auto op = dyn_cast<ascendc::LocalTensorReinterpretCastOp>(defOp))
        return getAllocationRoot(op.getIn());
    if (auto op = dyn_cast<ascendc::LocalTensorSubIndexOp>(defOp))
        return getAllocationRoot(op.getTensor());
    return {};
}

bool isWriteToAllocation(Operation* op, TensorOp root)
{
    if (auto dstOp = dyn_cast<ascendc::OpWithDst>(op)) {
        for (Value dst : dstOp.getDstTensors()) {
            TensorOp dstRoot = getAllocationRoot(dst);
            if (dstRoot && dstRoot == root)
                return true;
        }
    }
    return false;
}

//===----------------------------------------------------------------------===//
// Lifetime Analysis
//===----------------------------------------------------------------------===//

struct LifetimeInfo {
    bool hasUnrollIter = false;
    DenseSet<Operation*> iterativeAncestors;
    Operation* lastRead = nullptr;
    Operation* firstWrite = nullptr;
};

// Computes all lifetime info for a tensor in one collectAllUsers traversal.
// Used for sorting (pre-computed) and for canReuse (on-the-fly to avoid stale data).
LifetimeInfo computeLifetimeInfo(TensorOp tensorOp, DominanceInfo& di)
{
    LifetimeInfo info;
    SmallVector<Operation*> users = collectAllUsers(tensorOp);
    TensorOp root = getAllocationRoot(tensorOp.getResult());

    SmallVector<Operation*> readUsers;
    SmallVector<Operation*> writeUsers;

    for (Operation* user : users) {
        auto iter = getUnrollIter(user);
        if (iter)
            info.hasUnrollIter = true;
        if (auto forOp = user->getParentOfType<scf::ForOp>())
            info.iterativeAncestors.insert(forOp);
        if (auto whileOp = user->getParentOfType<scf::WhileOp>())
            info.iterativeAncestors.insert(whileOp);
        if (isa<scf::YieldOp>(user) || isa<CastOpInterface>(user) || isa<ascendc::LocalTensorSubIndexOp>(user))
            continue;
        if (isWriteToAllocation(user, root)) {
            writeUsers.push_back(user);
            // Same-op reuse: an op writing the allocation via $dst (cast chain) may also
            // read it via source operands. Track as read for accurate lastRead.
            if (auto dstOp = dyn_cast<ascendc::OpWithDst>(user)) {
                auto dstTensors = dstOp.getDstTensors();
                for (Value operand : user->getOperands())
                    if (!llvm::is_contained(dstTensors, operand) && getAllocationRoot(operand) == root) {
                        readUsers.push_back(user);
                        break;
                    }
            }
        } else {
            readUsers.push_back(user);
        }
    }

    if (!readUsers.empty()) {
        if (readUsers.size() == 1)
            info.lastRead = readUsers.front();
        else {
            llvm::stable_sort(
                readUsers, [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
            info.lastRead = readUsers.back();
        }
    }

    if (!writeUsers.empty()) {
        if (writeUsers.size() == 1)
            info.firstWrite = writeUsers.front();
        else {
            llvm::stable_sort(
                writeUsers, [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
            info.firstWrite = writeUsers.front();
        }
    }

    return info;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

bool canReuse(TensorOp bottom, TensorOp top, DominanceInfo& di)
{
    // On-the-fly computation avoids stale data from prior reuses.
    LifetimeInfo topInfo = computeLifetimeInfo(top, di);
    LifetimeInfo bottomInfo = computeLifetimeInfo(bottom, di);

    // Same-write-op restriction:
    // If both tensors are written by the same operation as destinations, they cannot be reused.
    // Operations like softmax write to multiple operands (dst, sumTensor, maxTensor,
    // sharedTmpBuffer) simultaneously via getDstTensors(), so these must have separate buffers.
    if (topInfo.firstWrite && bottomInfo.firstWrite && topInfo.firstWrite == bottomInfo.firstWrite) {
        LLVM_DEBUG(llvm::dbgs() << "  checking same-write-op: " << topInfo.firstWrite << "\n");
        if (auto dstOp = dyn_cast<ascendc::OpWithDst>(topInfo.firstWrite)) {
            auto dstTensors = dstOp.getDstTensors();
            TensorOp topRoot = getAllocationRoot(top.getResult());
            TensorOp bottomRoot = getAllocationRoot(bottom.getResult());

            LLVM_DEBUG(llvm::dbgs() << "    topRoot: " << topRoot << ", bottomRoot: " << bottomRoot << "\n");
            LLVM_DEBUG(llvm::dbgs() << "    dstTensors count: " << dstTensors.size() << "\n");

            bool topIsDst = false;
            bool bottomIsDst = false;
            for (Value dst : dstTensors) {
                TensorOp dstRoot = getAllocationRoot(dst);
                LLVM_DEBUG(llvm::dbgs() << "      dst: " << dst << ", dstRoot: " << dstRoot << "\n");
                if (dstRoot == topRoot)
                    topIsDst = true;
                if (dstRoot == bottomRoot)
                    bottomIsDst = true;
            }

            if (topIsDst && bottomIsDst) {
                LLVM_DEBUG(llvm::dbgs() << "  reject: both tensors are dst operands of same operation\n");
                return false;
            }
        }
    }

    // Cross-iteration write restriction:
    // If tensors have different unroll_iter values (including one having it and the other not)
    // and both are written by operations that write to multiple operands (like softmax),
    // prevent merging to avoid changing tensor identity across iterations.
    // This prevents issues where merging across iterations can lead to incorrect
    // buffer sharing in subsequent merges.
    // However, if the tensors are in different loops (or one is outside any loop),
    // they execute sequentially and can safely be merged.
    auto topIter = getUnrollIter(top);
    auto bottomIter = getUnrollIter(bottom);
    bool differentUnrollIter =
        (topIter && bottomIter && *topIter != *bottomIter) || (topIter && !bottomIter) || (!topIter && bottomIter);
    if (differentUnrollIter) {
        // Only restrict if both tensors share a common loop ancestor
        // (i.e., they're in the same unrolled loop)
        bool shareLoopAncestor = false;
        for (Operation* anc : topInfo.iterativeAncestors) {
            if (bottomInfo.iterativeAncestors.contains(anc)) {
                shareLoopAncestor = true;
                break;
            }
        }

        if (shareLoopAncestor) {
            // Only restrict if both tensors are written by operations with multiple dst operands
            if (topInfo.firstWrite && bottomInfo.firstWrite) {
                bool topHasMultipleDst = false;
                bool bottomHasMultipleDst = false;

                if (auto topDstOp = dyn_cast<ascendc::OpWithDst>(topInfo.firstWrite)) {
                    if (topDstOp.getDstTensors().size() > 1)
                        topHasMultipleDst = true;
                }
                if (auto bottomDstOp = dyn_cast<ascendc::OpWithDst>(bottomInfo.firstWrite)) {
                    if (bottomDstOp.getDstTensors().size() > 1)
                        bottomHasMultipleDst = true;
                }

                if (topHasMultipleDst || bottomHasMultipleDst) {
                    LLVM_DEBUG(
                        llvm::dbgs() << "  reject: different unroll_iters with multi-dst write ops in same loop\n");
                    return false;
                }
            }
        }
    }

    // Loop safety check: if both tensors share a loop ancestor and at least one
    // doesn't have unroll_iter, both must be "reborn" (written) inside the loop.
    // A tensor written outside the loop carries data across iterations.
    bool bothHaveUnrollIter = topInfo.hasUnrollIter && bottomInfo.hasUnrollIter;
    if (!bothHaveUnrollIter) {
        for (Operation* anc : topInfo.iterativeAncestors) {
            if (bottomInfo.iterativeAncestors.contains(anc)) {
                // Check if both tensors are reborn in the shared loop
                bool topReborn = isRebornInLoop(topInfo.firstWrite, anc);
                bool bottomReborn = isRebornInLoop(bottomInfo.firstWrite, anc);

                // If top is not reborn in the shared loop, check if it's reborn in a
                // different loop that completes before the shared loop starts
                if (!topReborn && topInfo.firstWrite) {
                    Operation* topWriteLoop = topInfo.firstWrite->getParentOfType<scf::ForOp>();
                    if (topWriteLoop && topWriteLoop != anc) {
                        // Check if topWriteLoop completes before anc starts
                        if (ascendc::opPrecedes(topWriteLoop, anc, di)) {
                            topReborn = true;
                        }
                    }
                }

                // If bottom is not reborn in the shared loop, check if it's reborn in a
                // different loop that completes before the shared loop starts
                if (!bottomReborn && bottomInfo.firstWrite) {
                    Operation* bottomWriteLoop = bottomInfo.firstWrite->getParentOfType<scf::ForOp>();
                    if (bottomWriteLoop && bottomWriteLoop != anc) {
                        // Check if bottomWriteLoop completes before anc starts
                        if (ascendc::opPrecedes(bottomWriteLoop, anc, di)) {
                            bottomReborn = true;
                        }
                    }
                }

                if (!topReborn || !bottomReborn) {
                    LLVM_DEBUG(llvm::dbgs() << "  reject: not reborn in shared loop\n");
                    return false;
                }
            }
        }
    }

    // Check lifetime overlap: top's lastRead must precede bottom's firstWrite.
    if (!topInfo.lastRead || !bottomInfo.firstWrite) {
        LLVM_DEBUG(llvm::dbgs() << "  reject: missing lastRead or firstWrite\n");
        return false;
    }

    // Same-op reuse (accumulator pattern): hardware reads all inputs before writing output.
    // Only allow for operations with OpWithReusableSrcInterface (whitelist approach).
    if (topInfo.lastRead == bottomInfo.firstWrite) {
        if (top.getType().getElementType() != bottom.getType().getElementType()) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: same-op element type mismatch\n");
            return false;
        }
        // Only allow same-op reuse for operations marked with OpWithReusableSrc.
        // Operations like BroadcastOp that read sources multiple times are not marked
        // and will be rejected here.
        if (!isa<ascendc::OpWithReusableSrc>(topInfo.lastRead)) {
            LLVM_DEBUG(llvm::dbgs() << "  reject: operation not marked with OpWithReusableSrcInterface\n");
            return false;
        }
        return true;
    }

    if (!ascendc::opPrecedes(topInfo.lastRead, bottomInfo.firstWrite, di))
        LLVM_DEBUG(llvm::dbgs() << "  reject: lifetime overlap\n");
    return ascendc::opPrecedes(topInfo.lastRead, bottomInfo.firstWrite, di);
}

void processTensorList(SmallVectorImpl<TensorOp>& tensorList, DominanceInfo& di)
{
    LLVM_DEBUG(llvm::dbgs() << "Processing tensor list with " << tensorList.size() << " tensors\n");
    // Greedy algorithm: pop earliest-freed tensor (top), try to merge with
    // latest-freed remaining tensor (bottom) whose lifetime starts after top's ends.
    // List is sorted latest-freed-first; reverse iteration starts from latest-freed,
    // maximizing reuse by merging short-lived allocations into long-lived ones.
    while (tensorList.size() > 1) {
        TensorOp topTensor = tensorList.pop_back_val();
        LLVM_DEBUG(llvm::dbgs() << "Trying to reuse: " << topTensor << "\n");
        for (auto& bottomTensor : llvm::reverse(tensorList)) {
            LLVM_DEBUG(llvm::dbgs() << "  Checking against: " << bottomTensor << "\n");
            if (!isReusable(bottomTensor) || !isReusable(topTensor) ||
                topTensor.getPosition() != bottomTensor.getPosition() || !canReuse(bottomTensor, topTensor, di))
                continue;

            LLVM_DEBUG(
                llvm::dbgs() << "Reuse: merging " << topTensor << " with " << bottomTensor
                             << " (position=" << topTensor.getPosition() << ")\n");

            if (isTensorGreaterOrEqual(topTensor, bottomTensor)) {
                Block* block = topTensor->getBlock();
                topTensor->moveBefore(block, block->begin());
                OpBuilder builder(topTensor);
                builder.setInsertionPointAfter(topTensor);
                auto castOp = builder.create<ascendc::LocalTensorReinterpretCastOp>(
                    bottomTensor->getLoc(), bottomTensor.getType(), topTensor);
                markForErase(bottomTensor, castOp);
                transferInOutFlags(bottomTensor, topTensor);
                bottomTensor = topTensor;
            } else {
                Block* block = bottomTensor->getBlock();
                bottomTensor->moveBefore(block, block->begin());
                OpBuilder builder(bottomTensor);
                builder.setInsertionPointAfter(bottomTensor);
                auto castOp = builder.create<ascendc::LocalTensorReinterpretCastOp>(
                    topTensor->getLoc(), topTensor.getType(), bottomTensor);
                markForErase(topTensor, castOp);
                transferInOutFlags(topTensor, bottomTensor);
            }
            break;
        }
    }
}

struct ReuseTensorAllocationPass : public ascendc::impl::ReuseTensorAllocationBase<ReuseTensorAllocationPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        DominanceInfo di(funcOp);

        SmallVector<TensorOp> allTensors;
        DenseMap<TensorOp, LifetimeInfo> lifetimeInfoMap;
        funcOp.walk<WalkOrder::PreOrder>([&](TensorOp tensor) {
            if (tensor->getUsers().empty() || !isReusable(tensor))
                return;
            if (llvm::none_of(tensor->getUsers(), isLoopCarriedYield)) {
                lifetimeInfoMap[tensor] = computeLifetimeInfo(tensor, di);
                allTensors.push_back(tensor);
            }
        });

        // Sort by program order: latest-freed first, earliest-freed last.
        // processTensorList pops from the back (earliest-freed) and searches
        // reverse (latest-freed first), maximizing reuse by merging short-lived
        // allocations into long-lived ones.
        llvm::stable_sort(allTensors, [&](TensorOp lhs, TensorOp rhs) {
            Operation* lhsFreed = lifetimeInfoMap[lhs].lastRead;
            if (!lhsFreed)
                lhsFreed = lifetimeInfoMap[lhs].firstWrite;
            Operation* rhsFreed = lifetimeInfoMap[rhs].lastRead;
            if (!rhsFreed)
                rhsFreed = lifetimeInfoMap[rhs].firstWrite;
            if (lhsFreed && rhsFreed)
                return ascendc::opPrecedes(rhsFreed, lhsFreed, di);
            return ascendc::opPrecedes(rhs, lhs, di);
        });

        processTensorList(allTensors, di);
        funcOp.walk([](TensorOp op) {
            if (op->hasAttr(eraseMeAttr))
                op.erase();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createReuseTensorAllocationPass()
{
    return std::make_unique<ReuseTensorAllocationPass>();
}
