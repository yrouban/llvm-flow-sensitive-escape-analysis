//===- lib/IR/FlowSensitiveAbstractions.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
// This pass implements abstractions related to escape analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/FlowSensitiveAbstractions.h"

#include "llvm/IR/Module.h"

#define DEBUG_TYPE "fsea"

using namespace llvm;
using namespace fsea;

#define DEFINE_ABSTRACTION_IDS(CamelCaseSym, name, numArgs) CamelCaseSym,
enum AbstractionIDs {
  Not_Abstraction = 0,
  DO_FSEA_ABSTRACTIONS(DEFINE_ABSTRACTION_IDS)
  UnspecifiedAbstraction,
};
#undef DEFINE_ABSTRACTION_IDS

#define DEFINE_ABSTRACTION_NAMES(CamelCaseSym, name, numArgs)  name,
static const char* AbstractionNames[] = {
  "",
  DO_FSEA_ABSTRACTIONS(DEFINE_ABSTRACTION_NAMES)
};
#undef DEFINE_ABSTRACTION_NAMES

#define DEFINE_ABSTRACTION(CamelCaseSym, name, numArgs)                        \
  Function *fsea::get##CamelCaseSym(const Module *M) {                         \
    Function *F = M->getFunction(name);                                        \
    assert(F && "could not find " name);                                       \
    return F;                                                                  \
  }                                                                            \
  bool fsea::is##CamelCaseSym(const Function *F) {                             \
    return getAbstractionID(*F) == AbstractionIDs::CamelCaseSym;              \
  }                                                                            \
                                                                               \
  StringRef fsea::get##CamelCaseSym##Name() { return StringRef(name); }

DO_FSEA_ABSTRACTIONS(DEFINE_ABSTRACTION)

#undef DEFINE_ABSTRACTION

static unsigned lookupAbstractionID(StringRef Name) {
  if (!Name.startswith("fsea.") &&
      !Name.startswith("gc.")) {
    return 0;
  }
  // TODO: binary search in sorted array?
  unsigned len = sizeof(AbstractionNames)/sizeof(AbstractionNames[0]);
  for (unsigned i = 0; i < len; ++i)
    if (Name == AbstractionNames[i])
      return i;
  return Not_Abstraction;
}

unsigned fsea::getAbstractionID(const Function &F) {
    // TODO: Should be chashed in Function and accessed as F.getAbstractinID().
    return lookupAbstractionID(F.getName());
}

bool fsea::isNewAllocation(const Function *F) {
  unsigned AbsID = getAbstractionID(*F);
  return AbsID == AbstractionIDs::NewObjectInstance ||
         AbsID == AbstractionIDs::NewArray ||
         AbsID == AbstractionIDs::NewNDimObjectArray;
}

// TODO: SafepointIRVerifier.cpp has the same definition.
bool fsea::isGCPointerType(const Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    // For the sake of this example GC, we arbitrarily pick addrspace(1) as our
    // GC managed heap.  We know that a pointer into this heap needs to be
    // updated and that no other pointer does.
    return (1 == PT->getAddressSpace());
  return false;
}

bool fsea::isGCPointer(const Value *V) {
  return isGCPointerType(V->getType());
}

bool fsea::isGCPointer(const Value &V) {
  return isGCPointerType(V.getType());
}

