//===- lib/Analysis/FlowSesnistiveExtendedIR.cpp - --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FlowSensitiveExtendedIR.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/FlowSensitiveEA.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/FlowSensitiveAbstractions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"


#define DEBUG_TYPE "fsea"

using namespace llvm;

STATISTIC(NumVPHIsMaterialized, "Number of virtual PHIs materialized");

namespace llvm {
namespace fsea {

namespace FlowSensitiveEA {
class State;
class FlowSensitiveEAUpdater;
}

namespace ExtendedIR {

AllocationID createAllocationID(const Instruction *NewI) {
  return reinterpret_cast<uint64_t>(NewI);
}

Value *VirtualPHI::materialize(
    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
  if (Materialized)
    return *Materialized;

  NumVPHIsMaterialized++;
  IRBuilder<> B(&*const_cast<BasicBlock *>(Parent)->begin());
  PHINode *MaterializedPHI =
      B.CreatePHI(Ty, getNumIncomingValues(), getName());
  Materialized = MaterializedPHI;
  for (unsigned i = 0; i < getNumIncomingValues(); i++)
    MaterializedPHI->addIncoming(Values[i].materialize(EAUpdater),
                                 const_cast<BasicBlock *>(Blocks[i]));
  EAUpdater.invalidateBlock(Parent);
  return *Materialized;
}

void VirtualPHI::print(raw_ostream &ROS) const {
  ROS << getName() << " = vphi " << *Ty << " ";
  for (unsigned i = 0; i < getNumIncomingValues(); i++) {
    if (i != 0)
      ROS << ", ";
    ROS << "[ ";
    Values[i].printAsOperand(ROS, false);
    ROS << ", ";
    Blocks[i]->printAsOperand(ROS, false);
    ROS << " ]";
  }
  VirtualValue::print(ROS);
}

CASStoredValue::CASStoredValue(const fsea::CompareAndSwapObject &CAS)
    : I(CAS.getCall()), Ptr(CAS.getObject()), Offset(CAS.getOffset()),
      ExpectedValue(CAS.getExpectedValue()), NewValue(CAS.getNewValue()) {}

CASStoredValue::CASStoredValue(const AtomicCmpXchgInst &ACXI)
    : I(ACXI), Ptr(ACXI.getPointerOperand()), Offset(nullptr),
      ExpectedValue(ACXI.getCompareOperand()),
      NewValue(ACXI.getNewValOperand()) {}

bool CASStoredValue::isValid(const FlowSensitiveEA::State &S) const {
  // The CASStoredValue is valid if the underlying allocation is
  // unescaped and has known exact state.
  auto TP = S.getTrackedPointer(Ptr);
  if (!TP)
    return false;
  auto *Alloc = S.getAllocation(TP->AllocID);
  if (!Alloc)
    return false;
  if (!Alloc->ExactState.has_value())
    return false;
#ifndef NDEBUG
  // Verify that the allocation actually has this CASStoredValue as the field
  // value at the corresponding offset.
  auto GetOffset = [&] () {
    if (!Offset) {
      assert(TP->Offset &&
             "Must have a constant offset if exact state is known!");
      return *TP->Offset;
    }
    assert(TP->isAlias() && "Ptr must be an alias!");
    auto *OffsetCI = dyn_cast<ConstantInt>(Offset);
    assert(OffsetCI && "Must have a constant offset if exact state is known!");
    return OffsetCI->getSExtValue();
  };
  auto FieldValue = Alloc->ExactState->getFieldValue(GetOffset());
  assert(FieldValue && *FieldValue == this &&
         "Must have this CASStoredValue as field value!");
#endif
  return true;
}

bool AtomicRMWStoredValue::isValid(const FlowSensitiveEA::State &S) const {
  // The AtomicRMWStoredValue is valid if the underlying allocation is
  // unescaped and has known exact state.
  auto TP = S.getTrackedPointer(ARMWI.getPointerOperand());
  if (!TP)
    return false;
  auto *Alloc = S.getAllocation(TP->AllocID);
  if (!Alloc)
    return false;
  if (!Alloc->ExactState.has_value())
    return false;
#ifndef NDEBUG
  // Verify that the allocation actually has this AtomicRMWStoredValue as the
  // field value at the corresponding offset.
  assert(TP->Offset && "Must have a constant offset if exact state is known!");
  auto FieldValue = Alloc->ExactState->getFieldValue(*TP->Offset);
  assert(FieldValue && *FieldValue == this &&
         "Must have this AtomicRMWStoredValue as field value!");
#endif
  return true;
}

Value *AtomicRMWStoredValue::materializeField(
    Instruction *InsertBefore,
    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
  if (!MaterializedCurrentFieldValue) {
    MaterializedCurrentFieldValue = CurrentFieldValue.materialize(EAUpdater);
    if (MaterializedCurrentFieldValue->getType() != getType()) {
      MaterializedCurrentFieldValue = CastInst::Create(
          Instruction::BitCast, MaterializedCurrentFieldValue, getType(),
          "$atomicrmw.bc", InsertBefore);
      EAUpdater.invalidateBlock(InsertBefore->getParent());
    }
  }
  return MaterializedCurrentFieldValue;
}

void AtomicRMWStoredValue::print(raw_ostream &ROS) const {
  ROS << getName() << " = AtomicRMWStoredValue(";
  ARMWI.printAsOperand(ROS);
  ROS << ", fieldValue: ";
  CurrentFieldValue.printAsOperand(ROS);
  ROS << ")";
  VirtualValue::print(ROS);
}

std::string AtomicRMWStoredValue::getName() const {
  return "$atomicrmw." + std::string(ARMWI.getName());
}

Type *AtomicRMWStoredValue::getType() const { return ARMWI.getType(); }

// Updates InsertBefore to the top most generated instruction.
static Value *
performAtomicOp(AtomicRMWInst::BinOp Op, Instruction *&InsertBefore,
                Value *Loaded, Value *Arg,
                fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater,
                const Twine &Name = "") {
  // Note: Do not try using IRBuilder here as it tries to const fold the new
  // instruction but the operands will be changed later.
  Instruction *NewInstr1;
  Instruction *NewInstr2;
  if (Op == AtomicRMWInst::Xchg)
    return Arg;
  EAUpdater.invalidateBlock(InsertBefore->getParent());
  switch (Op) {
  case AtomicRMWInst::Add:
    return InsertBefore = BinaryOperator::Create(Instruction::Add, Loaded, Arg,
                                                 Name + ".add", InsertBefore);
  case AtomicRMWInst::Sub:
    return InsertBefore = BinaryOperator::Create(Instruction::Sub, Loaded, Arg,
                                                 Name + ".sub", InsertBefore);
  case AtomicRMWInst::And:
    return InsertBefore = BinaryOperator::Create(Instruction::And, Loaded, Arg,
                                                 Name + ".and", InsertBefore);
  case AtomicRMWInst::Nand:
    NewInstr1 = BinaryOperator::Create(Instruction::And, Loaded, Arg,
                                       Name + ".and", InsertBefore);
    NewInstr2 =
        BinaryOperator::Create(Instruction::Xor, NewInstr1,
                               Constant::getAllOnesValue(NewInstr1->getType()),
                               Name + ".nand", InsertBefore);
    InsertBefore = NewInstr1;
    return NewInstr2;
  case AtomicRMWInst::Or:
    return InsertBefore = BinaryOperator::Create(Instruction::Or, Loaded, Arg,
                                                 Name + ".or", InsertBefore);
  case AtomicRMWInst::Xor:
    return InsertBefore = BinaryOperator::Create(Instruction::Xor, Loaded, Arg,
                                                 Name + ".xor", InsertBefore);
  case AtomicRMWInst::Max:
    NewInstr1 = ICmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_SGT, Loaded,
                                 Arg, Name + ".cmp", InsertBefore);
    NewInstr2 =
        SelectInst::Create(NewInstr1, Loaded, Arg, Name + ".max", InsertBefore);
    InsertBefore = NewInstr1;
    return NewInstr2;
  case AtomicRMWInst::Min:
    NewInstr1 = ICmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_SLE, Loaded,
                                 Arg, Name + ".sel", InsertBefore);
    NewInstr2 =
        SelectInst::Create(NewInstr1, Loaded, Arg, Name + ".min", InsertBefore);
    InsertBefore = NewInstr1;
    return NewInstr2;
  case AtomicRMWInst::UMax:
    NewInstr1 = ICmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_UGT, Loaded,
                                 Arg, Name + ".cmp", InsertBefore);
    NewInstr2 = SelectInst::Create(NewInstr1, Loaded, Arg, Name + ".umax",
                                   InsertBefore);
    InsertBefore = NewInstr1;
    return NewInstr2;
  case AtomicRMWInst::UMin:
    NewInstr1 = ICmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_ULE, Loaded,
                                 Arg, Name + ".cmp", InsertBefore);
    NewInstr2 = SelectInst::Create(NewInstr1, Loaded, Arg, Name + ".umin",
                                   InsertBefore);
    InsertBefore = NewInstr1;
    return NewInstr2;
  case AtomicRMWInst::FAdd:
    return InsertBefore = BinaryOperator::CreateFAdd(
               Loaded, Arg, Name + ".fadd", InsertBefore);
  case AtomicRMWInst::FSub:
    return InsertBefore = BinaryOperator::CreateFSub(
               Loaded, Arg, Name + ".fsub", InsertBefore);
  default:
    llvm_unreachable("Unknown atomic op");
  }
}

static void setAtomicOpField(AtomicRMWInst::BinOp Op, Value *Materialized,
                             Value *FieldValue) {
  // Materialized must be generated by performAtomicOp().
  switch (Op) {
  case AtomicRMWInst::Xchg:
    return;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::FAdd:
  case AtomicRMWInst::FSub: {
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Materialized);
    assert(BO && "BinaryOperator expected");
    BO->setOperand(0, FieldValue);
    return;
  }
  case AtomicRMWInst::Nand: {
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Materialized);
    assert(BO && "BinaryOperator expected");
    assert(BO->getOpcode() == Instruction::Xor && "Xor expected");
    BinaryOperator *BO2 = dyn_cast<BinaryOperator>(BO->getOperand(0));
    assert(BO2 && "BinaryOperator expected");
    assert(BO2->getOpcode() == Instruction::And && "And expected");
    BO2->setOperand(0, FieldValue);
    return;
  }
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin: {
    SelectInst *Sel = dyn_cast<SelectInst>(Materialized);
    assert(Sel && "SelectInst expected");
    ICmpInst *Cmp = dyn_cast<ICmpInst>(Sel->getOperand(0));
    assert(Cmp && "ICmpInst expected");
    Cmp->setOperand(0, FieldValue);
    Sel->setOperand(1, FieldValue);
    return;
  }
  default:
    llvm_unreachable("Unknown atomic op");
  }
}

Value *AtomicRMWStoredValue::materialize(
    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
  if (Materialized)
    return *Materialized;

  // To prevent recursive materialization cycle we must create
  // the materialized node before calling materialize() recursively.
  auto *DummyValue = ARMWI.getOperation() == AtomicRMWInst::Xchg
                         ? nullptr
                         : UndefValue::get(ARMWI.getType());
  AtomicRMWInst *ConstCastedARMWI = const_cast<AtomicRMWInst *>(&ARMWI);
  Instruction *InsertBefore = ConstCastedARMWI;
  Materialized =
      performAtomicOp(ARMWI.getOperation(), InsertBefore, DummyValue,
                      ConstCastedARMWI->getValOperand(), EAUpdater, "$atomicrmw");
  assert(ARMWI.getType() == Materialized.value()->getType());
  Value *FV = materializeField(InsertBefore, EAUpdater);
  setAtomicOpField(ARMWI.getOperation(), Materialized.value(), FV);

  return Materialized.value();
}

Value *CASStoredValue::materialize(
    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
  if (Materialized)
    return *Materialized;

  Instruction *I = const_cast<Instruction *>(&this->I);

  // To prevent recursive materialization cycle we must create
  // the materialized node before calling materialize() recursively.
  auto *DummyCondition = UndefValue::get(Type::getInt1Ty(I->getContext()));
  auto *DummyNew = UndefValue::get(NewValue->getType());
  auto *DummyOld = UndefValue::get(CurrentFieldValue.getType());
  SelectInst *Sel =
      SelectInst::Create(DummyCondition, DummyNew, DummyOld, "$cas.select", I);
  Materialized = Sel;

  IRBuilder<> Builder(Sel);
  auto *Cmp =
      Builder.CreateICmpEQ(CurrentFieldValue.materialize(EAUpdater),
                           const_cast<Value *>(ExpectedValue), "$cas.cmp");
  Sel->setCondition(Cmp);
  Sel->setTrueValue(const_cast<Value *>(NewValue));
  Sel->setFalseValue(CurrentFieldValue.materialize(EAUpdater));
  MaterializedCmp = Cmp;
  EAUpdater.invalidateBlock(I->getParent());
  return Sel;
}

Value *VirtualLoad::materialize(
    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
  if (Materialized)
    return *Materialized;

  IRBuilder<> Builder(const_cast<Instruction *>(&InsertBefore));
  auto AS = Src->getType()->getPointerAddressSpace();
  Value *Ptr = const_cast<Value *>(Src);
  if (Offset != 0) {
    auto *I8Ty = Type::getInt8Ty(InsertBefore.getContext());
    auto *I8PtrTy = I8Ty->getPointerTo(AS);

    // These changes are reported to EAUpdater below.
    auto *BitCast = Builder.CreateBitCast(Ptr, I8PtrTy);
    Ptr = Builder.CreateGEP(
        I8Ty, BitCast,
        ConstantInt::getSigned(Type::getInt64Ty(InsertBefore.getContext()),
                               Offset),
        "$load.gep");
  }

  auto *PtrTy = Ty->getPointerTo(AS);
  if (Ptr->getType() != PtrTy)
    Ptr = Builder.CreateBitCast(Ptr, PtrTy, "$load.bc");

  unsigned Align =
      PowerOf2Ceil(InsertBefore.getModule()->getDataLayout().getTypeStoreSize(
          const_cast<Type *>(Ty)));
  Materialized =
      Builder.CreateAlignedLoad(Ty, Ptr, MaybeAlign(Align), getName());
  EAUpdater.invalidateBlock(InsertBefore.getParent());
  return *Materialized;
}

void LoadedFields::print(raw_ostream &ROS) const {
  for (auto &VL : Entries) {
    ROS << "\n  ; ";
    VL.second->print(ROS);
  }
}

void VirtualLoad::print(raw_ostream &ROS) const {
  ROS << getName() << " = VirtualLoad ";
  Src->printAsOperand(ROS, false);
  ROS << " +" << Offset << "; ";
  VirtualValue::print(ROS);
}

std::string VirtualLoad::getName() const {
  return "$vload." + std::string(Src->getName()) + "." +
         std::to_string(Offset);
}

Value *ExtendedValue::materialize(
    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &EAUpdater) const {
  if (auto OptV = asValue())
    return const_cast<Value *>(*OptV);
  return asVirtualValue().value()->materialize(EAUpdater);
}

std::optional<Value *> ExtendedValue::getMaterialized() const {
  if (auto OptV = asValue())
    return const_cast<Value *>(*OptV);
  return asVirtualValue().value()->getMaterialized();
}

void ExtendedValue::print(raw_ostream &ROS) const {
  if (auto OptV = asValue())
    OptV.value()->print(ROS);
  else
    asVirtualValue().value()->print(ROS);
}

void ExtendedValue::printAsOperand(raw_ostream &ROS, bool PrintType) const {
  if (auto OptV = asValue())
    OptV.value()->printAsOperand(ROS, PrintType);
  else
    asVirtualValue().value()->printAsOperand(ROS, PrintType);
}

std::string ExtendedValue::getName() const {
  if (auto OptV = asValue())
    return std::string(OptV.value()->getName());
  else
    return asVirtualValue().value()->getName();
}

Type *ExtendedValue::getType() const {
  if (auto OptV = asValue())
    return OptV.value()->getType();
  else
    return asVirtualValue().value()->getType();
}
} // namespace ExtendedIR
} // namespace fsea
} // namespace llvm
