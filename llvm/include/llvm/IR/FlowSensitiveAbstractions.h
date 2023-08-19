//===-- llvm/IR/FlowSensitiveAbstractions.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header describes the abstactions that are used by Flow Sensitive Escape
// Analysis.
//===----------------------------------------------------------------------===//

#ifndef FLOWSENSITIVEABSTRACTIONS_H
#define FLOWSENSITIVEABSTRACTIONS_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
namespace fsea {

#define NUM_ARGS_UNKNOWN -1

/// Macro F is expected to have three arguments. The first is the CamelCase name
/// for abstraction. It is usually used in the generated C++ names. The second
/// is the name of the abstraction function. The third is the number of
/// arguments of the abstraction.
#define DO_FSEA_ABSTRACTIONS(F)                                                \
  F(GetKlassID, "fsea.get_klass_id", 1)                                        \
  F(NewObjectInstance, "fsea.new_instance", 2)                                 \
  F(NewArray, "fsea.new_array", NUM_ARGS_UNKNOWN)                              \
  F(NewNDimObjectArray, "fsea.multianewarray", 4)                              \
  F(FinalPublicationBarrier, "fsea.final_publication_barrier", 1)              \
  F(MonitorEnter, "fsea.monitorenter", 2)                                      \
  F(MonitorExit, "fsea.monitorexit", 2)                                        \
  F(MonitorEnterThreadLocal, "fsea.monitorenter.thread_local", 2)              \
  F(MonitorExitThreadLocal, "fsea.monitorexit.thread_local", 2)                \
  F(GCSafepointPoll, "gc.safepoint_poll", 0)                                   \
  F(CompareAndSwapObject, "fsea.compareAndSwapObject", 4)                      \

// This macro defines helper function which check if `Call` is a
// direct call and forwards query \p name to called function object.
// Otherwise (indirect call) returns false.
#define DEFINE_CALLBASE_PROXY(type, name)         \
  inline type name(const CallBase &Call) {        \
    if (const auto *F = Call.getCalledFunction()) \
      return name(F);                             \
    return false;                                 \
  }                                               \

// This macro defines helper function which check if Value `V` is an
// instance of `CallBase` and forwards query \p name to that `CallBase`
// specialization. Otherwise returns false.
#define DEFINE_VALUE_PROXY(type, name)            \
  inline type name(const Value &V) {              \
    if (const auto *CS = dyn_cast<CallBase>(&V))  \
      return name(*CS);                           \
    return false;                                 \
  }                                               \


#define DECL_ABSTRACTION(CamelCaseSym, name, numArgs)                          \
  bool is##CamelCaseSym(const Function *F);                                    \
  Function *get##CamelCaseSym(const Module *);                                 \
  DEFINE_CALLBASE_PROXY(bool, is##CamelCaseSym)                                \
  DEFINE_VALUE_PROXY(bool, is##CamelCaseSym)                                   \
  StringRef get##CamelCaseSym##Name();

DO_FSEA_ABSTRACTIONS(DECL_ABSTRACTION)

#undef DECL_ABSTRACTION

unsigned getAbstractionID(const Function &F);

/// Returns true if and only if the type specified is a
/// pointer to a GC'd object which must be included in
/// barriers and safepoints.
bool isGCPointerType(const llvm::Type *Ty);
bool isGCPointer(const llvm::Value *V);
bool isGCPointer(const llvm::Value &V);

bool isNewAllocation(const Function *F);

DEFINE_CALLBASE_PROXY(bool, isNewAllocation)
DEFINE_VALUE_PROXY(bool, isNewAllocation)

class NewObjectInstance {
  const CallBase &CS;

public:
  NewObjectInstance(const CallBase &CS) : CS(CS) {}
  /// valid only if isNewObjectInstance is true.
  NewObjectInstance(const Value &V) : CS(cast<CallBase>(V)) {}

  Value *getThread() { return CS.getArgOperand(0); }
  Value *getKlassID() { return CS.getArgOperand(1); }
};

class NewArray {
protected:
  const CallBase &CS;

public:
  NewArray(const CallBase &CS) : CS(CS) { }
  /// valid only if isNewArray is true.
  NewArray(const Value &V) : CS(cast<CallBase>(V)) { }

  Value *getThread() { return CS.getArgOperand(ThreadArgIdx); }
  Value *getArrayKlassID() { return CS.getArgOperand(ArrayKlassIDArgIdx); }
  Value *getElementKid() { return CS.getArgOperand(ElementKidArgIdx); }
  Value *getPrimitiveType() { return CS.getArgOperand(PrimitiveTypeArgIdx); }
  Value *getLength() { return CS.getArgOperand(LengthArgIdx); }
  Value *getHeaderSize() { return CS.getArgOperand(HeaderSizeArgIdx); }
  Value *getElementShift() { return CS.getArgOperand(ElementShiftArgIdx); }

  // Returns offset in bytes.
  Value *getZeroInitializeFrom() {
    return CS.arg_size() <= ZeroInitializeFromArgIdx
               ? nullptr
               : CS.getArgOperand(ZeroInitializeFromArgIdx);
  }

  static const int ThreadArgIdx = 0;
  static const int ArrayKlassIDArgIdx = 1;
  static const int ElementKidArgIdx = 2;
  static const int PrimitiveTypeArgIdx = 3;
  static const int LengthArgIdx = 4;
  static const int HeaderSizeArgIdx = 5;
  static const int ElementShiftArgIdx = 6;
  static const int ZeroInitializeFromArgIdx = 7; // Offset in bytes.

  // Pessimistically returns false.
  bool isFullyInitialized() {
    auto *ZeroFrom = getZeroInitializeFrom();
    if (!ZeroFrom)
      return true; // Support tests with fewer arguments.

    if (auto *ZeroFromCI = dyn_cast<ConstantInt>(ZeroFrom))
      if (ZeroFromCI->getZExtValue() == 0)
        return true;

    return false;
  }
};

class FinalPublicationBarrier {
  const CallBase &CS;

public:
  FinalPublicationBarrier(const CallBase &CS) : CS(CS) {}
  /// valid only if isFinalPublicationBarrier is true.
  FinalPublicationBarrier(const Value &V) : CS(cast<CallBase>(V)) {}

  Value *getValueArg() { return CS.getArgOperand(ValueArgIdx); }

  static const int ValueArgIdx = 0;
};

class GetKlassID {
  const CallBase &CS;

public:
  GetKlassID(const CallBase &CS) : CS(CS) {}
  /// valid only if isGetKlassID is true.
  GetKlassID(const Value &V) : CS(cast<CallBase>(V)) {}

  Value *getValueArg() { return CS.getArgOperand(ValueArgIdx); }

  static const int ValueArgIdx = 0;
};

class MonitorBase {
protected:
  const CallBase &CS;

public:
  MonitorBase(const CallBase &CS) : CS(CS) {
    assert((isMonitorEnter(CS) || isMonitorEnterThreadLocal(CS) ||
            isMonitorExit(CS) || isMonitorExitThreadLocal(CS)) &&
           "only valid for "
           "MonitorEnter/MonitorEnterThreadLocal/MonitorExit/"
           "MonitorEnterThreadLocal");
  }
  MonitorBase(const Value &V) : MonitorBase(cast<CallBase>(V)) {}

  Value *getThread() { return CS.getArgOperand(ThreadArgIdx); }
  Value *getObject() { return CS.getArgOperand(ObjectArgIdx); }
  const Use &getObjectUse() { return CS.getArgOperandUse(ObjectArgIdx); }
  const CallBase *getCall() { return &CS; }

  static const int ThreadArgIdx = 0;
  static const int ObjectArgIdx = 1;
};

class MonitorEnter : public MonitorBase {
public:
  MonitorEnter(const CallBase &CS) : MonitorBase(CS) {
    assert((isMonitorEnter(CS) || isMonitorEnterThreadLocal(CS)) &&
           "only valid for MonitorEnter or MonitorEnterThreadLocal");
  }
  MonitorEnter(const Value &V) : MonitorBase(cast<CallBase>(V)) {}
};

class MonitorExit : public MonitorBase {
public:
  MonitorExit(const CallBase &CS) : MonitorBase(CS) {
    assert((isMonitorExit(CS) || isMonitorExitThreadLocal(CS)) &&
           "only valid for MonitorExit or MonitorExitThreadLocal");
  }
  MonitorExit(const Value &V) : MonitorBase(cast<CallBase>(V)) {}
};

class CompareAndSwapObject {
  const CallBase &CS;

public:
  CompareAndSwapObject(const CallBase &CS) : CS(CS) {}
  /// valid only if isCompareAndSwapObject is true.
  CompareAndSwapObject(const Value &V) : CS(cast<CallBase>(V)) {}

  const CallBase &getCall() const { return CS; }

  Value *getObject() const { return CS.getArgOperand(0); }
  Value *getOffset() const { return CS.getArgOperand(1); }
  Value *getExpectedValue() const { return CS.getArgOperand(2); }
  Value *getNewValue() const { return CS.getArgOperand(3); }

  const Use &getObjectUse() const { return CS.getArgOperandUse(0); }
  const Use &getOffsetUse() const { return CS.getArgOperandUse(1); }
  const Use &getExpectedValueUse() const { return CS.getArgOperandUse(2); }
  const Use &getNewValueUse() const { return CS.getArgOperandUse(3); }
};

} // namespace fsea
} // namespace llvm

#endif /* FLOWSENSITIVEABSTRACTIONS_H */
