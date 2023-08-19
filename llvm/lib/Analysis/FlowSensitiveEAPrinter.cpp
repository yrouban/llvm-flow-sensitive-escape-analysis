//===- llvm/Analysis/FlowSensitiveEAPrinterPass.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/FlowSensitiveEAPrinter.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;
using namespace llvm::fsea::ExtendedIR;
using namespace llvm::fsea::FlowSensitiveEA;

PreservedAnalyses
FlowSensitiveEAPrinterPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetEA = [&](const Function &Func) -> FlowSensitiveEscapeAnalysis & {
    return FAM.getResult<FlowSensitiveEA>(const_cast<Function &>(Func))
        .getEAUpdater()
        .getFlowSensitiveEA();
  };
  FlowSensitiveEscapeAnalysis::Writer W(GetEA);
  M.print(outs(), &W);
  return PreservedAnalyses::all();
}
