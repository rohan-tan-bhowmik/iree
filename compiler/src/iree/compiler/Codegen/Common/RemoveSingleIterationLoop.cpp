// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-remove-trivial-loops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REMOVESINGLEITERATIONLOOPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned dimToIndex(gpu::Dimension dim) {
  switch (dim) {
  case gpu::Dimension::x:
    return 0;
  case gpu::Dimension::y:
    return 1;
  case gpu::Dimension::z:
    return 2;
  default:
    assert(false && "invalid dimension");
    return 0;
  }
}

/// If the value is a threadID return the range [0, workgroupSize-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workgroupCount.
static std::optional<std::pair<AffineExpr, AffineExpr>>
getWorkgroupRange(Value processorValue, SmallVectorImpl<Value> & /*dims*/,
                  SmallVectorImpl<Value> & /*symbols*/,
                  ArrayRef<int64_t> workgroupCount,
                  ArrayRef<int64_t> workgroupSize) {
  if (!workgroupSize.empty()) {
    if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
      unsigned index = dimToIndex(idOp.getDimension());
      OpBuilder b(processorValue.getContext());
      AffineExpr zero = b.getAffineConstantExpr(0);
      AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
      return std::make_pair(zero, ubExpr - 1);
    }
    if (auto dimOp = processorValue.getDefiningOp<gpu::BlockDimOp>()) {
      OpBuilder builder(processorValue.getContext());
      unsigned index = dimToIndex(dimOp.getDimension());
      AffineExpr bound = builder.getAffineConstantExpr(workgroupSize[index]);
      return std::make_pair(bound, bound);
    }
  }

  if (workgroupCount.empty() ||
      llvm::any_of(workgroupCount, ShapedType::isDynamic))
    return std::nullopt;

  if (auto idOp =
          processorValue.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>()) {
    OpBuilder builder(processorValue.getContext());

    // Can't infer the range when workroupCount is unknown.
    unsigned index = idOp.getDimension().getZExtValue();
    if (!workgroupCount[index])
      return std::nullopt;

    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue
                       .getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
    OpBuilder builder(processorValue.getContext());

    // Can't infer the range when workroupCount is unknown.
    unsigned index = dimOp.getDimension().getZExtValue();
    if (!workgroupCount[index])
      return std::nullopt;

    AffineExpr bound = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(bound, bound);
  }
  return std::nullopt;
}

static LogicalResult removeOneTripTiledLoops(mlir::FunctionOpInterface funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             ArrayRef<int64_t> numWorkgroups) {
  auto getWorkgroupRangeFn = [numWorkgroups,
                              workgroupSize](Value processorValue,
                                             SmallVectorImpl<Value> &dims,
                                             SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, numWorkgroups,
                             workgroupSize);
  };
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveSingleIterationLoopPass final
    : public impl::RemoveSingleIterationLoopPassBase<
          RemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      return;
    }
    SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);

    if (failed(removeOneTripTiledLoops(funcOp, workgroupSize.value(),
                                       numWorkgroups))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
