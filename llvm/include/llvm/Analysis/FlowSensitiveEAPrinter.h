//===- llvm/Analysis/FlowSensitiveEAPrinter.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/FlowSensitiveEA.h"

namespace llvm {

struct FlowSensitiveEAPrinterPass : PassInfoMixin<FlowSensitiveEAPrinterPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // namespace llvm
