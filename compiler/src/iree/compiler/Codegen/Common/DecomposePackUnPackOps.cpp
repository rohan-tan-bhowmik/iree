// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-decompose-pack-unpack-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSEPACKUNPACKOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// A wrapper pattern that calls linalg::lowerPack on tensor::PackOp. It lowers
/// a tensor.pack op to tensor.pad + tensor.expand_shape + linalg.transpose ops.
struct LowerPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerPackResult> res = linalg::lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

/// A warpper pattern that calls linalg::lowerUnPack on tensor::UnPackOp. It
/// lowers a tensor.unpack op to tensor.empty + linalg.transpose +
/// tensor.collapse_shape + tensor.extract_slice ops.
struct LowerUnPackPattern : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerUnPackOpResult> res =
        linalg::lowerUnPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to empty + transpose + reshape + extract_slice");
    }
    return success();
  }
};

struct DecomposePackUnPackOpsPass final
    : impl::DecomposePackUnPackOpsPassBase<DecomposePackUnPackOpsPass> {
  using impl::DecomposePackUnPackOpsPassBase<
      DecomposePackUnPackOpsPass>::DecomposePackUnPackOpsPassBase;
  explicit DecomposePackUnPackOpsPass(bool tileOuterToOne) {
    this->tileOuterToOne = tileOuterToOne;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void DecomposePackUnPackOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  // Generalization patterns for outer unit dims have higher priority because
  // they do not generate reshape ops.
  {
    RewritePatternSet patterns(ctx);
    patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                 linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "failed to apply generalization patterns on pack/unpack ops for "
          "outer unit dims cases");
      return signalPassFailure();
    }
  }

  // Do not convert pack and unpack ops if outer dims are expected to always be
  // tiled to one.
  if (!tileOuterToOne) {
    RewritePatternSet patterns(ctx);
    patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "failed to apply generalization patterns on pack/unpack ops for "
          "general cases.");
      return signalPassFailure();
    }
  }

  // TODO(hanchung): Below is a fallback solution for tensor.pack/unpack
  // decomposition. They will be retired after lowerPack and lowerUnPack handle
  // all the cases.

  // Apply tiling to make outer dims be all 1s.
  {
    IRRewriter rewriter(ctx);
    auto packOptions = scf::SCFTileAndFuseOptions().setTilingOptions(
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
              auto packOp = cast<tensor::PackOp>(op);

              // Do nothing if any of inner tile sizes is dynamic.
              if (llvm::any_of(packOp.getMixedTiles(), [](OpFoldResult tile) {
                    return tile.is<Value>();
                  })) {
                return {};
              }

              int inputRank = packOp.getSourceRank();
              SmallVector<OpFoldResult> tileSizes(inputRank,
                                                  builder.getIndexAttr(1));
              return tileSizes;
            }));
    funcOp->walk([&](tensor::PackOp op) {
      FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
          scf::tileConsumerAndFuseProducersUsingSCF(
              rewriter, cast<TilingInterface>(op.getOperation()), packOptions);
      if (failed(tileAndFuseResult))
        return signalPassFailure();
      rewriter.replaceOp(op, tileAndFuseResult->replacements[op.getResult()]);
    });

    auto unpackTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) {
              auto unpackOp = cast<tensor::UnPackOp>(op);
              int numLoops = unpackOp.getDestRank();
              auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
              SmallVector<OpFoldResult> tileSizes;
              for (int i = 0; i < numLoops; ++i) {
                if (dimAndTileMapping.count(i)) {
                  tileSizes.push_back(dimAndTileMapping[i]);
                } else {
                  tileSizes.push_back(builder.getIndexAttr(1));
                }
              }
              return tileSizes;
            });
    funcOp->walk([&](tensor::UnPackOp op) {
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, cast<TilingInterface>(op.getOperation()),
                            unpackTilingOptions);
      if (failed(tilingResult))
        return signalPassFailure();
      rewriter.replaceOp(op, tilingResult->replacements);
    });

    LLVM_DEBUG({
      llvm::dbgs()
          << "--- After applying tiling that makes outer dims be all 1s ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  // Canonicalize tiled ops.
  {
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    ctx->getOrLoadDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After canonicalizing tiled ops ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  {
    RewritePatternSet patterns(ctx);
    patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                 linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDecomposePackUnPackOpsPass(bool tileOuterToOne) {
  return std::make_unique<DecomposePackUnPackOpsPass>(tileOuterToOne);
}

} // namespace mlir::iree_compiler
