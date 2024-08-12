// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTEXECUTABLEPREPROCESSINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TestExecutablePreprocessingPass final
    : impl::TestExecutablePreprocessingPassBase<
          TestExecutablePreprocessingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    // Replace i64 constants with whatever we source from the target
    // configuration. A real pipeline would use the target information to do
    // whatever it needed to the executable instead.
    getOperation()->walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
      auto configAttr = variantOp.getTarget().getConfiguration();
      if (!configAttr)
        return;
      auto replacementAttr = configAttr.getAs<IntegerAttr>("replace_i64");
      if (!replacementAttr) {
        // Skip variants that don't request modification.
        return;
      }
      variantOp.walk([&](Operation *op) {
        if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
          if (constantOp.getType() == replacementAttr.getType()) {
            constantOp.setValueAttr(replacementAttr);
          }
        }
      });
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
