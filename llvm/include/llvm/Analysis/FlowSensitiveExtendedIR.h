//===- llvm/Analysis/FlowSensitiveExtendedIR.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header defines the extended IR which is used by Flow Sensitive Escape
// Analysis to represent the exact state of unescaped object.
//
// Flow Sensitive Escape Analysis runs an abstract interpreter that iterates
// over CFG and its state keeps track of all allocated objects with their field
// values down to the points where the objects escape. The machine processes
// every instruction of the control flow and updates the state according to
// the instructions' effect on the objects and their fields.
//
// Extended values are calculated as a result of this analysis and cannot
// always be represented as elements of IR. For example, a field can have
// either one of two possible constant values, depending on which of two
// branches executed, but there is no phi node in the IR that merges these
// values. To extend LLVM IR for such merge cases two types are introduced:
// VirtualValue and ExtendedValue. The abstract interpreter operates on the
// extended IR which consists of the LLVM IR and extended values. The process
// of conversion of the extended IR to LLVM IR is called materialization.
//===----------------------------------------------------------------------===//

#ifndef FLOWSENSITIVEEXTENDEDIR_H
#define FLOWSENSITIVEEXTENDEDIR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>

namespace llvm {
class AtomicCmpXchgInst;
class AtomicRMWInst;
class Instruction;
class Type;

namespace fsea {
class CompareAndSwapObject;
namespace FlowSensitiveEA {
class State;
class FlowSensitiveEAUpdater;
} // namespace FlowSensitiveEA

namespace ExtendedIR {

using namespace llvm;

// We use AllocationID as an opaque identifier of a tracked allocation.
using AllocationID = uint64_t;
AllocationID createAllocationID(const Instruction *NewI);

class VirtualValue;
class ExtendedValue {
  friend struct DenseMapInfo<ExtendedValue>;

  using ValuePtrUnion = PointerUnion<const Value *, const VirtualValue *>;
  ValuePtrUnion V;
  ExtendedValue(ValuePtrUnion V) : V(V) {}

public:
  ExtendedValue(const Value *V) : V(V) { assert(V && "Can't be null!"); }
  ExtendedValue(const VirtualValue *V) : V(V) { assert(V && "Can't be null!"); }

  ExtendedValue() = default;
  ExtendedValue(const ExtendedValue &) = default;
  ExtendedValue(ExtendedValue &&) = default;
  ExtendedValue &operator=(const ExtendedValue &) = default;
  ExtendedValue &operator=(ExtendedValue &&) = default;

  std::optional<const Value *> asValue() const {
    if (V.is<const Value *>())
      return V.get<const Value *>();
    return std::nullopt;
  }
  std::optional<const VirtualValue *> asVirtualValue() const {
    if (V.is<const VirtualValue *>())
      return const_cast<VirtualValue *>(V.get<const VirtualValue *>());
    return std::nullopt;
  }

  Value *
  materialize(fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const;
  /// Returns the materialized Value * if it exists, or std::nullopt otherwise.
  std::optional<Value *> getMaterialized() const;
  void print(raw_ostream &ROS) const;
  void printAsOperand(raw_ostream &ROS, bool PrintType = true) const;
  std::string getName() const;
  Type *getType() const;

  bool operator==(const ExtendedValue &Other) const { return V == Other.V; }
  bool operator!=(const ExtendedValue &Other) const {
    return !(*this == Other);
  }
  bool operator<(const ExtendedValue &Other) const { return V < Other.V; };

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); dbgs() << "\n"; }
#endif
};

class VirtualPHI;
class CASStoredValue;
class VirtualLoad;
class AtomicRMWStoredValue;

class VirtualValue {
protected:
  mutable std::optional<Value *> Materialized;

public:
  virtual ~VirtualValue() {}

  virtual Value *materialize(
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const = 0;
  virtual std::string getName() const = 0;
  virtual Type *getType() const = 0;
  std::optional<Value *> getMaterialized() const {
    return Materialized;
  }
  virtual void print(raw_ostream &ROS) const {
    if (Materialized) {
      ROS << ", materialized as ";
      Materialized.value()->printAsOperand(ROS);
    }
  }
  virtual void printAsOperand(raw_ostream &ROS, bool PrintType = true) const {
    if (PrintType) {
      getType()->print(ROS);
      ROS << " ";
    }
    ROS << getName();
  }

  // Dynamic downcast (ala dyn_cast_or_null<Type>()).
  // List all subclasses here and override only two in its subclass.
  // TODO: Consider reusing LLVM casting.
  virtual VirtualPHI *asVirtualPHI() { return nullptr; }
  virtual const VirtualPHI *asVirtualPHI() const { return nullptr; }
  virtual CASStoredValue *asCASStoredValue() { return nullptr; }
  virtual const CASStoredValue *asCASStoredValue() const { return nullptr; }
  virtual VirtualLoad *asVirtualLoad() { return nullptr; }
  virtual const VirtualLoad *asVirtualLoad() const { return nullptr; }
  virtual AtomicRMWStoredValue *asAtomicRMWStoredValue() { return nullptr; }
  virtual const AtomicRMWStoredValue *asAtomicRMWStoredValue() const {
    return nullptr;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); dbgs() << "\n"; }
#endif
};

/// VirtualPHI represents a PHI node which is not present in the IR.
///
/// We use virtual PHIs to represent field values when the field is initialized
/// with different values across different paths.
class VirtualPHI : public VirtualValue {
  const unsigned ID;
  const BasicBlock *Parent;
  Type *Ty;

  SmallVector<ExtendedValue, 4> Values;
  SmallVector<const BasicBlock *, 4> Blocks;

public:
  VirtualPHI(unsigned ID, const BasicBlock *Parent, Type *Ty)
      : ID(ID), Parent(Parent), Ty(Ty) {}
  ~VirtualPHI() override {}

  VirtualPHI *asVirtualPHI() override { return this; }
  const VirtualPHI *asVirtualPHI() const override { return this; }

  void addIncoming(ExtendedValue V, const BasicBlock *BB) {
    Values.push_back(V);
    Blocks.push_back(BB);
    assert(V.getType() == Ty && "Incoming type mismatch!");
  }

  Value *materialize(
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const override;

  void print(raw_ostream &ROS) const override;
  std::string getName() const override { return "$vphi." + std::to_string(ID); }
  Type *getType() const override { return Ty; }
  const BasicBlock *getParent() const { return Parent; }

  unsigned getNumIncomingValues() const { return Values.size(); }
  const BasicBlock *getIncomingBlock(unsigned i) const { return Blocks[i]; }
  ExtendedValue getIncomingValue(unsigned i) const { return Values[i]; }

  void removeAllIncomingValues() {
    Blocks.clear();
    Values.clear();
  }

  auto incoming_values() const
      -> decltype(make_range(Values.begin(), Values.end())) {
    return make_range(Values.begin(), Values.end());
  }
};

/// CASStoredValue represents a field value that is calculated as a result of
/// call to fsea.compareAndSwapObject(object, offset, expected, newValue) or
/// cmpxchg object, expected, newValue.
class CASStoredValue : public VirtualValue {
public:
  const Instruction &I;
  const Value *Ptr;
  const Value *Offset; // nullptr for cmpxchg.
  const Value *ExpectedValue;
  const Value *NewValue;
  ExtendedValue CurrentFieldValue;

protected:
  mutable Value *MaterializedCmp;

public:
  CASStoredValue(const fsea::CompareAndSwapObject &CAS);
  CASStoredValue(const AtomicCmpXchgInst &ACXI);
  ~CASStoredValue() override {}

  CASStoredValue *asCASStoredValue() override { return this; }
  const CASStoredValue *asCASStoredValue() const override { return this; }

  void setCurrentFieldValue(ExtendedValue CurrentFieldValue) {
    assert(ExpectedValue->getType() == CurrentFieldValue.getType());
    this->CurrentFieldValue = CurrentFieldValue;
  }

  bool equalsIgnoreCurrentFieldValue(const CASStoredValue &CAS) const {
    return &this->I == &CAS.I && this->Ptr == CAS.Ptr &&
           this->Offset == CAS.Offset &&
           this->ExpectedValue == CAS.ExpectedValue &&
           this->NewValue == CAS.NewValue;
  }

  /// Check that the CASStoredValue still refers to a filed value of a tracked
  /// object.
  /// When created or updated the CASStoredValue object captures a field value
  /// of a tracked object which might escape at later analysis iterations. This
  /// method allows a simple check for conditions that guarantee that the
  /// captured field value can be calculated and the calculation results in
  /// the same value.
  bool isValid(const FlowSensitiveEA::State &S) const;

  bool isMaterialized() const { return Materialized.has_value(); }
  Value *materialize(
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const override;

  void print(raw_ostream &ROS) const override {
    ROS << getName() << " = CASStoredValue(";
    I.printAsOperand(ROS, false);
    ROS << ", obj: ";
    Ptr->printAsOperand(ROS, false);
    ROS << ", fieldValue: ";
    CurrentFieldValue.printAsOperand(ROS, false);
    ROS << ")";
    VirtualValue::print(ROS);
  }

  std::string getName() const override {
    return "$vcas." + std::string(I.getName());
  }

  Type *getType() const override { return ExpectedValue->getType(); }

  Value *getMaterializedCmp(
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
    materialize(EAUpdater);
    assert(MaterializedCmp);
    return MaterializedCmp;
  }
};

/// AtomicRMWStoredValue represents a field value that is calculated as a result
/// of atomicrmw.
class AtomicRMWStoredValue : public VirtualValue {
public:
  const AtomicRMWInst &ARMWI;
  ExtendedValue CurrentFieldValue;

protected:
  // Stores materialized CurrentFieldValue possibly bitcasted to the type of
  // ARMWI.
  mutable Value *MaterializedCurrentFieldValue = nullptr;

public:
  AtomicRMWStoredValue(const AtomicRMWInst &ARMWI) : ARMWI(ARMWI) {}
  ~AtomicRMWStoredValue() override {}

  AtomicRMWStoredValue *asAtomicRMWStoredValue() override { return this; }
  const AtomicRMWStoredValue *asAtomicRMWStoredValue() const override {
    return this;
  }

  bool isValid(const FlowSensitiveEA::State &S) const;
  bool isMaterialized() const { return Materialized.has_value(); }
  Value *materialize(
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const override;
  Value *getMaterializedCurrentFieldValue() const {
    return MaterializedCurrentFieldValue;
  }

  void setCurrentFieldValue(ExtendedValue FieldValue) {
    assert(!MaterializedCurrentFieldValue ||
           *CurrentFieldValue.getMaterialized() ==
               *FieldValue.getMaterialized());
    this->CurrentFieldValue = FieldValue;
  }

  Value *materializeField(
      Instruction *InsertBefore,
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const;

  void print(raw_ostream &ROS) const override;

  std::string getName() const override;

  Type *getType() const override;
};

/// VirtualLoad represents field values loaded from untracked objects.
class VirtualLoad : public VirtualValue {
  const Instruction &InsertBefore;
  Type *const Ty;         // Loaded result type.
  const Value *const Src; // Source pointer to load from.
  const int64_t Offset;   // Offset within the source pointer.

public:
  VirtualLoad(const Instruction &InsertBefore, Type *Ty, const Value *Src,
              int64_t Offset)
      : InsertBefore(InsertBefore), Ty(Ty), Src(Src), Offset(Offset) {}
  ~VirtualLoad() override {}

  bool operator==(const VirtualLoad &VL) const {
    return &InsertBefore == &VL.InsertBefore && Ty == VL.Ty && Src == VL.Src &&
           Offset == VL.Offset;
  }

  VirtualLoad *asVirtualLoad() override { return this; }
  const VirtualLoad *asVirtualLoad() const override { return this; }

  Value *materialize(
      fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const override;

  void print(raw_ostream &ROS) const override;

  std::string getName() const override;

  Type *getType() const override { return Ty; }
};

struct SingleValueInstructionModel;
struct LoadedFields;

/// InstructionModel owns some virtual values that will be materialized at an
/// instruction and stored in VirtualContext.
struct InstructionModel {
public:
  virtual ~InstructionModel() {}

  virtual void print(raw_ostream &ROS) const = 0;

  virtual LoadedFields *asLoadedFields() { return nullptr; }
  virtual const LoadedFields *asLoadedFields() const { return nullptr; }
  virtual SingleValueInstructionModel *asSingleValueInstructionModel() {
    return nullptr;
  }
  virtual const SingleValueInstructionModel *
  asSingleValueInstructionModel() const {
    return nullptr;
  }
};

/// Stores exactly one VirtualValue.
struct SingleValueInstructionModel : public InstructionModel {
  std::unique_ptr<VirtualValue> VValue;

  SingleValueInstructionModel(VirtualValue *VValue) : VValue(VValue) {
    assert(!VValue->asVirtualPHI());
  }
  ~SingleValueInstructionModel() override {}

  SingleValueInstructionModel *asSingleValueInstructionModel() override {
    return this;
  }
  const SingleValueInstructionModel *
  asSingleValueInstructionModel() const override {
    return this;
  }

  void print(raw_ostream &ROS) const override {
    ROS << "\n  ; ";
    VValue->print(ROS);
  }
};

struct LoadedFields : public InstructionModel {
  // Field offset to owned VirtualLoad map.
  std::map<unsigned, std::unique_ptr<VirtualLoad>> Entries;

  ~LoadedFields() override {}

  LoadedFields *asLoadedFields() override { return this; }
  const LoadedFields *asLoadedFields() const override { return this; }

  void print(raw_ostream &ROS) const override;
};

/// VirtualContext maintains the "extended" part of the extended IR used by
/// the analysis.
///
/// Currently the extended part consists of virtual PHI nodes. VirtualContext
/// manages and owns VirtualPHI objects. It should be used to create new virtual
/// PHIs and iterate over existing virtual PHIs.
class VirtualContext {
  unsigned NumVirtualPHIs = 0;

  struct BlockContext {
    /// The owning map of virtual PHIs per basic block.
    using VirtualPHIOwningVector = SmallVector<std::unique_ptr<VirtualPHI>, 8>;
    VirtualPHIOwningVector VirtualPHIs;
    /// A map between a field (described as a pair of
    /// AllocationID and Offset) and the corresponding virtual PHI.
    using FieldDesc = std::pair<AllocationID, int64_t>;
    using FieldToVPHIMap = DenseMap<FieldDesc, VirtualPHI *>;
    FieldToVPHIMap FieldToVPHIs;
    DenseMap<const Instruction *, std::unique_ptr<InstructionModel>>
        InstructionModels;

    /// Check that all stored instructions in this BlockContext
    /// are in the specified block.
    void verify(const BasicBlock *BB) {
#ifndef NDEBUG
      for (auto &It : InstructionModels)
        assert(BB == It.first->getParent() &&
               "Instruction is from another block!");
#endif
    }
  };

  DenseMap<const BasicBlock *, BlockContext> BlocksContext;

  BlockContext &getOrCreateBlockContext(const BasicBlock *BB) {
    auto It = BlocksContext.find(BB);
    if (It != BlocksContext.end())
      return It->second;
    auto Emplaced = BlocksContext.try_emplace(BB);
    assert(Emplaced.second && "Must have been emplaced");
    return Emplaced.first->second;
  }

  VirtualPHI *createVirtualPHI(const BasicBlock *Parent, Type *Ty) {
    auto *VPHI = new VirtualPHI(NumVirtualPHIs++, Parent, Ty);
    auto &Context = getOrCreateBlockContext(Parent);
    Context.VirtualPHIs.emplace_back(VPHI);
    return VPHI;
  }

public:
  void clear() {
    NumVirtualPHIs = 0;
    BlocksContext.clear();
  }

  void forgetForBlock(const BasicBlock *BB) {
    auto It = BlocksContext.find(BB);
    if (It == BlocksContext.end())
      return;
    NumVirtualPHIs -= It->second.VirtualPHIs.size();
    BlocksContext.erase(It);
  }

  class vphi_const_iterator {
    // vphi_const_iterator is just a wrapper over owning vector iterator.
    BlockContext::VirtualPHIOwningVector::const_iterator Current;

  public:
    vphi_const_iterator(BlockContext::VirtualPHIOwningVector::const_iterator i)
        : Current(i) {}

    bool operator!=(const vphi_const_iterator &other) const {
      return !(*this == other);
    }

    bool operator==(const vphi_const_iterator &Other) const {
      return Current == Other.Current;
    }

    vphi_const_iterator &operator++() {
      Current++;
      return *this;
    }

    VirtualPHI *operator*() const { return Current->get(); }
  };

  VirtualValue *getVirtualValue(const Instruction &I) const {
    if (auto *VC = getInstructionModel(I)) {
      auto VCV = VC->asSingleValueInstructionModel();
      assert(VCV);
      return VCV->VValue.get();
    }
    return nullptr;
  }

  InstructionModel *getInstructionModel(const Instruction &I) const {
    auto It = BlocksContext.find(I.getParent());
    if (It == BlocksContext.end())
      return nullptr;
    auto &Context = It->second;
    auto VC = Context.InstructionModels.find(&I);
    return VC == Context.InstructionModels.end() ? nullptr : VC->second.get();
  }

  void setInstructionModel(const Instruction &I,
                           std::unique_ptr<InstructionModel> &&IM) {
    auto &Context = getOrCreateBlockContext(I.getParent());
    auto Emplaced = Context.InstructionModels.try_emplace(&I, std::move(IM));
    assert(Emplaced.second);
    (void)Emplaced;
  }

  VirtualPHI *getOrCreateVirtualPHI(const BasicBlock *Parent, Type *Ty,
                                    AllocationID ID, int64_t Offset) {
    auto &Context = getOrCreateBlockContext(Parent);
    auto Desc = std::make_pair(ID, Offset);
    auto &ParentVPHIs = Context.FieldToVPHIs;
    auto It = ParentVPHIs.find(Desc);
    if (It != ParentVPHIs.end()) {
      assert(It->second->getType() == Ty && "Type mismatch!");
      return It->second;
    }
    auto *NewVPHI = createVirtualPHI(Parent, Ty);
    ParentVPHIs[Desc] = NewVPHI;
    return NewVPHI;
  }

  void printBlockVirtualPHIs(const BasicBlock *BB, raw_ostream &ROS) const {
    for (auto *PHI : vphis(BB)) {
      ROS << "  ; ";
      PHI->print(ROS);
      ROS << "\n";
    }
  }

  vphi_const_iterator vphis_begin(const BasicBlock *BB) const {
    auto It = BlocksContext.find(BB);
    if (It == BlocksContext.end())
      return nullptr;
    return It->second.VirtualPHIs.begin();
  }

  vphi_const_iterator vphis_end(const BasicBlock *BB) const {
    auto It = BlocksContext.find(BB);
    if (It == BlocksContext.end())
      return nullptr;
    return It->second.VirtualPHIs.end();
  }

  auto vphis(const BasicBlock *BB) const
      -> decltype(make_range(vphis_begin(BB), vphis_end(BB))) {
    return make_range(vphis_begin(BB), vphis_end(BB));
  }

  void verify() {
#ifndef NDEBUG
    for (auto &It : BlocksContext)
      It.second.verify(It.first);
#endif
  }
};

} // namespace ExtendedIR
} // namespace fsea

template <> struct DenseMapInfo<fsea::ExtendedIR::ExtendedValue> {
  static inline fsea::ExtendedIR::ExtendedValue getEmptyKey() {
    return DenseMapInfo<decltype(fsea::ExtendedIR::ExtendedValue::V)>
              ::getEmptyKey();
  }
  static inline fsea::ExtendedIR::ExtendedValue getTombstoneKey() {
    return DenseMapInfo<decltype(fsea::ExtendedIR::ExtendedValue::V)>
              ::getTombstoneKey();
  }
  static unsigned getHashValue(const fsea::ExtendedIR::ExtendedValue &Val) {
    return DenseMapInfo<decltype(fsea::ExtendedIR::ExtendedValue::V)>
              ::getHashValue(Val.V);
  }

  static bool isEqual(const fsea::ExtendedIR::ExtendedValue &LHS,
                      const fsea::ExtendedIR::ExtendedValue &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif /* FLOWSENSITIVEEXTENDEDIR_H */
