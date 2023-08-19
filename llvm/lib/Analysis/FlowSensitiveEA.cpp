//===- lib/Analysis/FlowSensitiveEA.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
// This pass implements flow-sensitive escape analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FlowSensitiveEA.h"

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fsea"

using namespace llvm;

// Describes the meanings of the address spaces used.
enum AddressSpaceLayout : unsigned  {
  // Anything not in java heap or another known space.  Aliases w/ the TLS
  // address space, but not the JavaHeap ones.
  CHeapAddrSpace = 0,

  // Contains only objects managed by the GC; references are 64 bits.  May
  // alias with CompressedJavaHeapAddrSpace, but not other two.
  JavaHeapAddrSpace = 1,

  // Contains only objects managed by the GC; references are 32 bits.  May
  // alias with JavaHeapAddrSpace, but not other two.
  CompressedJavaHeapAddrSpace = 2,

  // Contains objects in the C-Heap which do not require LVBs and SVBs. No
  // guarantees on what it may alias with.
  CHeapNonLVBSVBAddrSpace = 3,

  // TLS Access (mapped to the JavaThread structure, analogous to the
  // standard usage).  Access through this address space are mapped to GS
  // register.  May alias structures in the CHeapAddressSpace.
  JavaThreadTLSAddrSpace = 256
};

static cl::opt<bool> MergeAllocationsAtPHIs("opt-alloc-merge-at-phis",
                                            cl::Hidden, cl::init(true));
/// Makes sense only with opt-alloc-track-exact-state enabled.
/// If enabled and exact state can't be tracked the analysis will fall back
/// to non-exact state.
/// If disabled and exact state can't be tracked the analysis will consider the
/// allocation escaped.
static cl::opt<bool> TrackNonExactState("opt-alloc-track-non-exact-state",
                                        cl::Hidden, cl::init(true));
/// Track symbolic allocation state as a sequence of instructions required to
/// produce initialized allocation state.
static cl::opt<bool>
    TrackSymbolicAllocationState("opt-alloc-track-symbolic-state", cl::Hidden,
                                 cl::init(true));
/// Track the exact state (i.e. values of all fields) of unescaped allocations.
/// This enables more accurate escape analysis as we can track pointers to
/// unescaped allocaions in fields of unescaped allocations.
static cl::opt<bool> TrackExactAllocationState("opt-alloc-track-exact-state",
                                               cl::Hidden, cl::init(true));
static cl::opt<bool> PrintTrackedPointers("opt-alloc-fs-print-tracked-pointers",
                                          cl::Hidden, cl::init(false));
static cl::opt<bool> PrintSymbolicState("opt-alloc-fs-print-symbolic-state",
                                        cl::Hidden, cl::init(false));
static cl::opt<bool> PrintDeoptState("opt-alloc-fs-print-deopt-state",
                                     cl::Hidden, cl::init(false));
/// Should long running instructions invalidate the available deopt state?
///
/// It might be legal to reexecute such instructions in the interpreter, but
/// this might have a negative impact on the deoptimization latencies.
///
/// NOTE: this is a purely theoretical concern. We don't have evidence that
/// this is problematic in practice.
static cl::opt<bool> CanReexecuteExpensiveInstuctionsOnDeopt(
    "opt-alloc-can-reexecute-expensive-intrs-on-deopt", cl::Hidden,
    cl::init(true));
static cl::opt<unsigned> IterationLimit("opt-alloc-fs-iteration-limit",
                                        cl::Hidden, cl::init(10));
static cl::opt<bool> OptAllocOptimisticMerge("opt-alloc-fs-optimistic-merge",
                                             cl::Hidden, cl::init(true));

/// Model effects of memcpys on tracked allocations up to this length.
static cl::opt<unsigned> ModelMemcpyMaxElements(
    "opt-alloc-fs-model-max-memcpy-elements", cl::Hidden, cl::init(16));
static cl::opt<bool> ModelCAS("opt-alloc-fs-model-cas", cl::Hidden,
                              cl::init(true));
static cl::opt<bool> ModelMemcpy("opt-alloc-fs-model-memcpy", cl::Hidden,
                                 cl::init(true));

namespace llvm {
namespace fsea {
namespace FlowSensitiveEA {

Allocation::Allocation(AllocationID ID, const Value *KlassID,
                       std::optional<ExtendedValue> ArrayLength,
                       std::optional<ExtendedValue> ZeroInitializeFrom,
                       const Instruction *I)
    : ID(ID), KlassID(KlassID), ArrayLength(ArrayLength),
      ZeroInitializeFrom(ZeroInitializeFrom), NewInstruction(I) {
  if (TrackExactAllocationState)
    ExactState = ExactAllocationState();
  if (TrackSymbolicAllocationState)
    SymbolicState = SymbolicAllocationState();
}

bool Allocation::isTrackedField(int64_t Offset, Type *Ty) const {
  return getFieldInfo(Offset, Ty).has_value();
}

bool Allocation::isTrackedField(int64_t Offset, unsigned SizeInBytes) const {
  return getFieldInfo(Offset, SizeInBytes).has_value();
}

void Allocation::print(raw_ostream &ROS, const State *S) const {
  LLVMContext &C = NewInstruction->getContext();
  NewInstruction->printAsOperand(ROS, false);
  ROS << ", kid=";
  KlassID->printAsOperand(ROS, false);

  std::optional<TypeUtils::JavaType> T;
  if (auto MaybeKID = TypeUtils::runTimeToCompileTimeKlassID(KlassID)) {
    T = TypeUtils::JavaType(*MaybeKID, true);
    if (auto JTI = VMInterface::getJavaTypeInfo(C, *T, std::nullopt))
      if (!JTI->getName().empty())
        ROS << " (" << JTI->getName() << ")";
  }

  if (!isArray())
    assert(!ZeroInitializeFrom && "Doesn't make sense for non-arrays");
  else {
    ROS << ", arraylength=";
    ArrayLength->printAsOperand(ROS, false);

    // This is an array that may have a non-zero ZeroInitializeFrom.
    if (!ZeroInitializeFrom)
      ROS << ", zerofrom=undef";
    else
      [&]() { // Print ZeroInitializeFrom.
        if (auto V = ZeroInitializeFrom->asValue())
          if (auto *CI = dyn_cast<ConstantInt>(*V))
            if (CI->isZero())
              return; // Don't print zerofrom=0

        ROS << ", zerofrom=";
        ZeroInitializeFrom->printAsOperand(ROS, false);
      }();
  }

  if (NeedsPublicationBarrier)
    ROS << ", needs publication barrier";

  if (ContributingAllocations.size() != 1)
    ROS << "\n  ;    contributing allocations: "
        << ContributingAllocations.size();
  if (ExactState)
    ExactState->print(C, T, S, ROS);
  else
    ROS << "\n  ;    exact state unknown\n";

  if (PrintSymbolicState) {
    if (SymbolicState)
      SymbolicState->print(ROS);
    else
      ROS << "  ;    symbolic state unknown\n";
  }
}

void ExactAllocationState::print(LLVMContext &C,
                                 std::optional<TypeUtils::JavaType> T,
                                 const State *S, raw_ostream &ROS) const {
  if (LockCount)
    ROS << ", lockcount=" << LockCount;
  ROS << "\n";
  // Collect both initialized fields and known invariant locations
  SmallSet<int64_t, 16> KnownOffsetsSet;
  KnownOffsetsSet.insert(InvariantFields.begin(), InvariantFields.end());
  for (auto &FV : FieldValues)
    KnownOffsetsSet.insert(FV.first);

  // Sort the offsets and print everything we know about them
  SmallVector<int64_t, 16> KnownOffsets(KnownOffsetsSet.begin(),
                                        KnownOffsetsSet.end());
  std::sort(KnownOffsets.begin(), KnownOffsets.end());
  for (int64_t Offset : KnownOffsets) {
    ROS << "  ;    +" << Offset;
    if (T)
      if (auto FI = VMInterface::getFieldInfoAtOffset(C, *T, true, Offset))
        if (!FI->getName().empty())
          ROS << " (" << FI->getName() << ")";
    ROS << ": ";
    auto FieldValue = getFieldValue(Offset);
    if (FieldValue) {
      FieldValue->printAsOperand(ROS);
      if (S && S->getTrackedPointer(*FieldValue))
        ROS << " (tracked allocation)";
    }
    else
      ROS << "uninitialized";

    if (isInvariantField(Offset))
      ROS << ", invariant";

    ROS << "\n";
  }
}

State::DeoptState::DeoptState(CallBase *CB, const State &S) : Call(CB) {
  auto OBU = Call->getOperandBundle(LLVMContext::OB_deopt);
  assert(OBU && "Must have a deopt state!");
  SmallVector<ExtendedValue, 8> Worklist;
  for (const Use &U : OBU->Inputs)
    if (fsea::isGCPointer(U))
      Worklist.push_back(static_cast<Value*>(U));
  ReferredAllocIDs = S.getAllocationClosure(std::move(Worklist));
}

Allocation &State::addTrackedAllocation(
    AllocationID NewID, const Instruction *Alloc, const Value *KlassID,
    std::optional<ExtendedValue> ArrayLength,
    std::optional<ExtendedValue> ZeroInitializeFrom,
    const SmallSet<AllocationID, 4> &ContributingAllocations) {
  assert(isa<ConstantInt>(KlassID) && "KlassID must be const");
  assert((fsea::isNewObjectInstance(*Alloc) || fsea::isNewArray(*Alloc) ||
          isa<PHINode>(Alloc)) &&
         "Must be a new allocation or a phi");
  assert((!fsea::isNewObjectInstance(*Alloc) || !ArrayLength.has_value()) &&
         "instance allocations can't have the array length specified");
  auto Inserted = Allocations.insert(
      std::make_pair(NewID, Allocation(NewID, KlassID, ArrayLength,
                                       ZeroInitializeFrom, Alloc)));
  addTrackedPointer(Alloc, TrackedPointer(NewID, 0));
  assert(getAllocation(NewID));
  assert(getTrackedPointer(Alloc));
  assert(getAllocation(getTrackedPointer(Alloc)));
  Allocation &NewAllocation = Inserted.first->second;
  NewAllocation.ContributingAllocations = ContributingAllocations;
  return NewAllocation;
}

/// Prints tracked allocations sorted by their names.
void State::print(raw_ostream & ROS) const {
  SmallVector<const Allocation *, 8> SortedAllocations;
  for (auto &A : Allocations)
    SortedAllocations.emplace_back(&A.second);
  std::sort(SortedAllocations.begin(), SortedAllocations.end(),
            [](const Allocation *A1, const Allocation *A2) {
              return A1->NewInstruction->getName().compare(
                         A2->NewInstruction->getName()) < 0;
            });
  for (auto &A : SortedAllocations) {
    ROS << "  ;  alloc: ";
    A->print(ROS, this);
  }
  if (PrintTrackedPointers) {
    SmallVector<std::pair<ExtendedValue, std::string>, 8> SortedTPs;
    for (auto &It : TrackedPointers)
      SortedTPs.emplace_back(It.first, It.first.getName());
    std::sort(SortedTPs.begin(), SortedTPs.end(),
              [](const std::pair<ExtendedValue, std::string> &V1,
                 const std::pair<ExtendedValue, std::string> &V2) {
                return V1.second.compare(V2.second) < 0;
              });
    for (auto &Ptr : SortedTPs) {
      ROS << "  ;  tracked pointer: ";
      Ptr.first.printAsOperand(ROS, false);
      ROS << " - ";
      getTrackedPointer(Ptr.first)->print(ROS, *this);
      ROS << "\n";
    }
  }
  if (PrintDeoptState) {
    ROS << "  ;  deopt state: ";
    if (LastDeoptState)
      ROS << *LastDeoptState->getCall() << "\n";
    else
      ROS << "none\n";
  }
}

bool SymbolicStateInstVisitor::mayClobber(
    const Instruction *I, SymbolicAllocationState::MemorySource MS) {
  // First, try to determine if allocation may be clobbered using State
  auto MayClobberUsingState = [&]() -> bool {
    // Check that this is a recognized instruction modifying a tracked
    // allocation. Give up if it's unrecognized.
    std::optional<TrackedPointer> ModTP;
    if (auto *SI = dyn_cast<StoreInst>(I)) {
      ModTP = S.getTrackedPointer(SI->getPointerOperand());
    } else if (auto *AMI = dyn_cast<AtomicMemCpyInst>(I)) {
      ModTP = S.getTrackedPointer(AMI->getRawDest());
    } else { // TODO: Add support for MemMove, CmpXchg, RMW and etc.
      return true; // Unsupported memory writing instruction.
    }
    auto PtrTP = S.isUnescapedAllocation(MS.TP) ? MS.TP : std::nullopt;
    // If one of the pointer is tracked and the other is not the instruction
    // can't clobber.
    if (ModTP.has_value() != PtrTP.has_value())
      return false;
    // If both of the pointers are tracked check that the allocation referenced
    // by PtrTP is is not one of the contributing allocations for the ModTP.
    if (ModTP && PtrTP)
      if (!S.getAllocation(ModTP)
          ->ContributingAllocations.contains(PtrTP->AllocID))
        return false;
    return true;
  };
  // If we can't figure out using EA, ask generic AA
  return MayClobberUsingState() && isModSet(AA.getModRefInfo(
      I, MemoryLocation(MS.Ptr, LocationSize::beforeOrAfterPointer())));
}

bool SymbolicStateInstVisitor::visit(Instruction *I) {
  bool ChangeMade = false;
  if (I->mayWriteToMemory())
    for (auto &A : S.Allocations)
      if (auto &State = A.second.SymbolicState)
        for (auto &MS : State->MemorySources)
          if (mayClobber(I, MS)) {
            LLVM_DEBUG(dbgs() << "Instruction " << *I
                              << " invalidates memory which is used in "
                                 "symbolic state of allocation "
                              << *A.second.NewInstruction << "\n");
            State = std::nullopt;
            ChangeMade = true;
            break;
          }
  ChangeMade |= InstVisitor::visit(I);
  return ChangeMade;
}

bool StateInstVisitor::applyPublicationBarrier(
    const CallBase *PublicationBarrierCall) {
  fsea::FinalPublicationBarrier Barrier(*PublicationBarrierCall);
  auto TP = S.getTrackedPointer(Barrier.getValueArg());
  if (!TP)
    return false;
  auto Allocation = S.getAllocation(TP);
  if (Allocation->NeedsPublicationBarrier)
    return false;
  Allocation->NeedsPublicationBarrier = true;
  return true;
}

bool StateInstVisitor::visit(Instruction *I) {
  LLVM_DEBUG(dbgs() << "Visiting " << *I << "\n");
  bool ChangeMade = false;
  if (fsea::isFinalPublicationBarrier(*I))
    ChangeMade |= applyPublicationBarrier(cast<CallBase>(I));
  ChangeMade |= ExactStateVisitor.visit(I);
  ChangeMade |= SymbolicStateVisitor.visit(I);
  ChangeMade |= DeoptStateVisitor.visitInstruction(*I);
  S.verify();
  return ChangeMade;
}

FlowSensitiveEscapeAnalysis::FlowSensitiveEscapeAnalysis(
    const Function &F, const DominatorTree &DT, AAResults &AA,
    LazyValueInfo &LVI)
    : F(F), DT(DT), DL(F.getParent()->getDataLayout()),
      BatchAA(AA), LVI(LVI) {
  calculateBBStates();
}

/// Returns a VirtualValue bound to the instruction \p I in the state \p S.
const VirtualValue *
FlowSensitiveEscapeAnalysis::getVirtualValue(const State &S,
                                             const Instruction &I) const {
  if (auto *VV = VContext.getVirtualValue(I)) {
    if (auto *CAS = VV->asCASStoredValue())
      return CAS->isValid(S) ? CAS : nullptr;
    if (auto *ARMW = VV->asAtomicRMWStoredValue())
      return ARMW->isValid(S) ? ARMW : nullptr;
    return VV;
  }
  return nullptr;
}

/// Returns the set of allocations which escape on the given edge.
/// These are the allocations from From Out state which escaped during merge
/// of To predecessors.
SetVector<std::pair<AllocationID, const Allocation *>>
FlowSensitiveEscapeAnalysis::getEscapedAllocationsForEdge(
    const BasicBlock *From, const BasicBlock *To) const {
  SetVector<std::pair<AllocationID, const Allocation *>> Escaped;

  auto FromIt = BlockStates.find(From);
  if (FromIt == BlockStates.end())
    return Escaped;
  const State &FromState = FromIt->second.Out;

  auto ToIt = BlockStates.find(To);
  if (ToIt == BlockStates.end())
    return Escaped;
  const State &ToState = ToIt->second.In;

  auto AllocationIDs = ToState.getEscapedAllocations(FromState);
  for (auto ID : AllocationIDs)
    Escaped.insert(std::make_pair(ID, FromState.getAllocation(ID)));
  return Escaped;
}

void FlowSensitiveEscapeAnalysis::Writer::emitState(
    FlowSensitiveEscapeAnalysis &EA, const BasicBlock *BB,
    const State BasicBlockState::*S, const char *T,
    formatted_raw_ostream &ROS) {
  auto Mapped = EA.BlockStates.find(BB);
  if (Mapped == EA.BlockStates.end())
    return;

  if ((Mapped->second.*S).isEmpty())
    return;

  ROS << "  ;  " << T << ":\n";
  (Mapped->second.*S).print(ROS);
}

void FlowSensitiveEscapeAnalysis::Writer::emitEscapedAtMerge(
    FlowSensitiveEscapeAnalysis &EA, const BasicBlock *BB,
    formatted_raw_ostream &ROS) {
  SmallSet<AllocationID, 4> Visited;
  std::string AllocationsList;
  raw_string_ostream AllocationsListStream(AllocationsList);
  unsigned Count = 0;
  for (auto *Pred : predecessors(BB)) {
    auto Escaped = EA.getEscapedAllocationsForEdge(Pred, BB);
    for (auto A : Escaped)
      if (Visited.insert(A.first).second) {
        if (Count++)
          AllocationsListStream << ", ";
        A.second->NewInstruction->printAsOperand(AllocationsListStream, false);
      }
  }
  if (!Visited.empty())
    ROS << "  ;  escaped allocations: " << AllocationsList << "\n";
}

void FlowSensitiveEscapeAnalysis::Writer::collectInfoCommentsForBlock(
    FlowSensitiveEscapeAnalysis &EA, const BasicBlock &BB) {
  auto SaveInfo = [&] (State &S, const Instruction &I,
                       const State *PrevState = nullptr) {
    if (auto TP = S.getTrackedPointer(&I)) {
      raw_string_ostream RSO(InfoComments[&I]);
      if (&I == S.getAllocation(TP->AllocID)->NewInstruction)
        RSO << "; Allocation";
      else {
        RSO << "; Tracked pointer: ";
        TP->print(RSO, S);
      }
    }
    if (auto *VC = EA.VContext.getInstructionModel(I)) {
      raw_string_ostream RSO(InfoComments[&I]);
      VC->print(RSO);
    }
    // Report allocations escaped at this instruction.
    if (!PrevState)
      // Can't report escaped allocations without previous state
      return;
    auto Escaped = S.getEscapedAllocations(*PrevState);
    if (Escaped.empty())
      return;
    raw_string_ostream RSO(InfoComments[&I]);
    RSO << "  ;  escaped allocations: ";
    unsigned Count = 0;
    for (auto ID : Escaped) {
      if (Count++)
        RSO << ", ";
      auto *A = PrevState->getAllocation(ID);
      A->NewInstruction->printAsOperand(RSO, false);
    }
  };

  auto BS = EA.BlockStates.find(&BB);
  if (BS == EA.BlockStates.end())
    return;
  State S = BS->second.In;
  for (const PHINode &Phi : BB.phis())
    SaveInfo(S, Phi);

  StateInstVisitor Visitor(S, EA);
  for (auto &I : make_range(BB.getFirstNonPHI()->getIterator(), BB.end())) {
    auto PrevState = S;
    bool Changed = Visitor.visit(const_cast<Instruction *>(&I));
    SaveInfo(S, I, Changed ? &PrevState : nullptr);
  }
  if (S != BS->second.Out)
    // We should not print analysis while running the fixed point iteration
    // as new vphi nodes may be generated as a result of state calculation
    // in this method. These new vphis are printed but stay unused.
    // If this message is printed after the fixed point iteration is
    // finished then there is a bug the fixed point iteration.
    InfoComments[BB.getTerminator()] +=
        "\n ; WARNING: Out state is unstable";
}

void FlowSensitiveEscapeAnalysis::Writer::emitBasicBlockStartAnnot(
    const BasicBlock *BB, formatted_raw_ostream &ROS) {
  auto &EA = GetEA(*BB->getParent());

  collectInfoCommentsForBlock(EA, *BB);
  emitEscapedAtMerge(EA, BB, ROS);
  EA.getVirtualContext().printBlockVirtualPHIs(BB, ROS);
  emitState(EA, BB, &BasicBlockState::In, "In", ROS);
}

void FlowSensitiveEscapeAnalysis::Writer::emitBasicBlockEndAnnot(
    const BasicBlock *BB, formatted_raw_ostream &ROS) {
  auto &EA = GetEA(*BB->getParent());

  emitState(EA, BB, &BasicBlockState::Out, "Out", ROS);
  InfoComments.clear();
}

void FlowSensitiveEscapeAnalysis::Writer::printInfoComment(
    const Value &V, formatted_raw_ostream &ROS) {
  auto InfoComment = InfoComments.find(&V);
  if (InfoComment == InfoComments.end())
    return;

  ROS.PadToColumn(50);
  ROS << InfoComment->second;
}

void FlowSensitiveEscapeAnalysis::print(raw_ostream &ROS) {
  auto GetEA = [&](const Function &Func) -> FlowSensitiveEscapeAnalysis & {
    assert(&Func == &F && "Can't handle any other function");
    return *this;
  };
  Writer W(GetEA);
  F.print(ROS, &W);
}

void TrackedPointer::print(raw_ostream &ROS, const State &S) const {
  S.getAllocation(AllocID)->NewInstruction->printAsOperand(ROS, false);
  if (Offset.has_value())
    ROS << " +" << Offset.value();
}

void FlowSensitiveEscapeAnalysis::print(raw_ostream &ROS,
                                        bool PrintType) const {
  for (auto &BR : BlockRPON) {
    const BasicBlock *BB = BR.first;
    auto BS = BlockStates.find(BB);
    if (BS == BlockStates.end())
      continue;
    ROS << BB->getName() << ":\n";
    ROS << ";  In:\n";
    BS->second.In.print(ROS);
    getVirtualContext().printBlockVirtualPHIs(BB, ROS);
    auto &S = BS->second.Out;
    for (auto &I : *BB)
      if (auto TP = S.getTrackedPointer(&I)) {
        I.printAsOperand(ROS, true);
        if (&I == S.getAllocation(TP->AllocID)->NewInstruction)
          ROS << ": Allocation";
        else {
          ROS << ": Tracked pointer: ";
          TP->print(ROS, S);
        }
        ROS << "\n";
      }
    ROS << ";  Out:\n";
    BS->second.Out.print(ROS);
  }
}

std::optional<ExtendedValue> ExactAllocationState::getMergedFieldValue(
    VirtualContext &VContext, AllocationID ID,
    GetAllocationIDForBlock GetAllocID, int64_t Offset, const BasicBlock *BB,
    GetBlockOutState GetState) {
  // (1) Collect incoming values for all predecessors.
  //
  // We need to traverse through the predecessors twice because the field might
  // be uninitialized for some of the them. In this case we should use the
  // initial field value, but we need to know the field Type to use
  // Allocation::getInitialFieldValue. So, in the first loop collect initialized
  // values and the Type. In the second loop fill non-initialized values.
  SmallDenseMap<const BasicBlock *, ExtendedValue, 8> IncomingValues;
  SmallDenseMap<const BasicBlock *, State::FieldValueType, 8> IncomingTypes;
  SmallDenseMap<const BasicBlock *, const Allocation *, 8> IncomingAllocs;
  Type *FieldTy = nullptr;
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (PredBlockState.isBackedgeUnknown())
      IncomingTypes[Pred] = State::BackedgeUnknown;
    else if (PredBlockState.isUnreachableUnknown())
      IncomingTypes[Pred] = State::UnreachableUnknown;
    else {
      auto Alloc = PredBlockState.getValue()->getAllocation(GetAllocID(Pred));
      assert(Alloc && "Should have the allocation available!");
      IncomingAllocs[Pred] = Alloc;
      assert(Alloc->ExactState && "Must have exact allocation state!");
      auto FieldValue = Alloc->ExactState->getFieldValue(Offset);
      if (FieldValue) {
        IncomingTypes[Pred] = State::Initialized;
        IncomingValues[Pred] = *FieldValue;
        FieldTy = FieldValue->getType();
      } else
        IncomingTypes[Pred] = State::NotInitialized;
    }
  }

  assert(FieldTy && "There should be at least one initialized field!");
  for (auto It : IncomingTypes)
    if (It.second == State::NotInitialized) {
      if (auto FieldValue =
              IncomingAllocs[It.first]->getInitialFieldValue(Offset, FieldTy))
        IncomingValues[It.first] = *FieldValue;
      else {
        LLVM_DEBUG(
            dbgs() << "Cannot merge unknown non-initialized field at offset "
                   << Offset << "\n";);
        return std::nullopt; // Merge of unknown non-initialized field
      }
    }

  // (2) Check if all values are the same.
  auto GetSingleValue = [&]() -> std::optional<ExtendedValue> {
    std::optional<ExtendedValue> SingleValue;
    for (auto It : IncomingValues) {
      if (!SingleValue)
        SingleValue = It.second;
      else if (SingleValue != It.second)
        return std::nullopt;
    }
    return SingleValue;
  };

  if (auto SingleValue = GetSingleValue())
    return *SingleValue; // No phi needed

  for (auto It : IncomingValues)
    if (It.second.getType() != FieldTy) {
      LLVM_DEBUG(dbgs() << "Cannot merge field values at offset " << Offset
                        << " with different types: " << *It.second.getType()
                        << " != " << *FieldTy << "\n";);
      return std::nullopt; // Incoming types mismatch!
    }

  // (3) Create or update an existing virtual PHI
  return State::getOrCreateVirtualPHIForField(VContext, BB, FieldTy, ID, Offset,
                                              IncomingValues, IncomingTypes);
}

std::optional<ExactAllocationState>
ExactAllocationState::getMergedAllocationState(
    VirtualContext &VContext, AllocationID ID,
    GetAllocationIDForBlock GetAllocID, const BasicBlock *BB,
    GetBlockOutState GetState) {
  ExactAllocationState MergedState;

  // (1) Collect all known incoming states
  SmallVector<const ExactAllocationState *, 4> IncomingStates;
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    auto Alloc = PredBlockState.getValue()->getAllocation(GetAllocID(Pred));
    assert(Alloc && "Should have the allocation available!");
    if (!Alloc->ExactState) {
      LLVM_DEBUG(dbgs() << "Exact allocation state is not known in one of the "
                        << "predecessors\n";);
      return std::nullopt;
    }
    IncomingStates.push_back(&Alloc->ExactState.value());
  }

  // (2) Merge the lock count, can only merge the same lock count across all
  // incoming states.
  std::optional<unsigned> SingleLockCount;
  for (auto *ExactState : IncomingStates)
    if (!SingleLockCount)
      SingleLockCount = ExactState->LockCount;
    else if (SingleLockCount != ExactState->LockCount) {
      LLVM_DEBUG(dbgs() << "Different lock counts: " << *SingleLockCount
                        << " != " << ExactState->LockCount << "\n";);
      return std::nullopt;
    }
  MergedState.LockCount = *SingleLockCount;

  // (3) Collect field offsets
  SmallSet<int64_t, 16> Offsets;
  for (auto *ExactState : IncomingStates)
    for (auto It : ExactState->FieldValues) {
      int64_t Offset = It.first;
      if (!MergedState.getFieldValue(Offset)) {
        if (auto FieldValue =
                getMergedFieldValue(VContext, ID, GetAllocID,
                                    Offset, BB, GetState))
          MergedState.FieldValues[Offset] = *FieldValue;
        else
          return std::nullopt;
      }
    }

  // (4) Collect fields invariant in all incoming states
  DenseMap<int64_t, unsigned> InvariantCounts;
  for (auto *ExactState : IncomingStates)
    for (int64_t Offset : ExactState->InvariantFields)
      InvariantCounts[Offset]++;
  for (auto It : InvariantCounts)
    if (It.second == IncomingStates.size())
      MergedState.markInvariantField(It.first);

  return MergedState;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &OS,
    const SymbolicAllocationState::InitializingInstruction &II) {
  II.print(OS);
  return OS;
}

void SymbolicAllocationState::print(raw_ostream &ROS) const {
  for (auto &I : InitializingInstructions)
    ROS << "  ; " << I << "\n";
}

std::optional<SymbolicAllocationState>
SymbolicAllocationState::getMergedAllocationState(
    GetAllocationIDForBlock GetAllocID, const BasicBlock *BB,
    GetBlockOutState GetState) {
  // Produce the merged state if the symbolic state is the same across all
  // incoming paths.
  const SymbolicAllocationState *SingleSymbolicState = nullptr;
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    auto Alloc = PredBlockState.getValue()->getAllocation(GetAllocID(Pred));
    assert(Alloc && "Should have the allocation available!");
    if (!Alloc->SymbolicState) {
      LLVM_DEBUG(dbgs() << "Symbolic allocation state is not known in one"
                        << " of the predecessors\n";);
      return std::nullopt;
    }
    if (!SingleSymbolicState)
      SingleSymbolicState = &*Alloc->SymbolicState;
    else if (*SingleSymbolicState != Alloc->SymbolicState) {
      LLVM_DEBUG(dbgs() << "Different incoming symbolic state\n";);
      return std::nullopt;
    }
  }

  assert(SingleSymbolicState != nullptr && "Must have some state!");
  return *SingleSymbolicState;
}

bool SymbolicAllocationState::isModifyingAllocationContent(
    const SymbolicAllocationState::InitializingInstruction &II) {
  // The only kind of non-IR instruction is publication barrier and it doesn't
  // modify the allocation content.
  if (!II.isIRInstruction())
    return false;

  const Instruction *I = II.getIRInstruction();

  if (fsea::isFinalPublicationBarrier(*I))
    return false;

  if (auto *II = dyn_cast<IntrinsicInst>(I))
    if (II->getIntrinsicID() == Intrinsic::invariant_start)
      return false;

  return true;
}

ExtendedValue State::getOrCreateVirtualPHIForField(
    VirtualContext &VContext, const BasicBlock *BB, Type *FieldTy,
    AllocationID ID, int64_t Offset,
    SmallDenseMap<const BasicBlock *, ExtendedValue, 8> &IncomingValues,
    SmallDenseMap<const BasicBlock *, FieldValueType, 8> &IncomingTypes) {
  // Create or update an existing virtual PHI
  auto *VPHI = VContext.getOrCreateVirtualPHI(BB, FieldTy, ID, Offset);

  // If we are merging a loop header it's possible to have a backedge unknown
  // state in some of the predecessors. In this case we treat the backedge
  // unknown states as optimistic "TOP" state. This way we produce a specualtive
  // state which is propagated down the CFG. Once the backedge states becomes
  // available (these are still speculative states, because they are computed
  // based on the specualtive merge result) the caller re-merges the header
  // again and continues the analysis until the fixed point is reached, i.e.
  // until the merged state matches the specualtive state of the previous merge.
  //
  // When we revisit the merge it's possible to have an existing VPHI for this
  // field at this block. In this case we just update the incoming values for
  // this VPHI. Note that the incoming values for VPHIs don't participate in
  // the state comparison.
  VPHI->removeAllIncomingValues();

  for (auto *Pred : predecessors(BB)) {
    ExtendedValue IncomingV;
    if (IncomingTypes[Pred] == UnreachableUnknown)
      // If the incoming block has unknown state because it's dead, take undef
      // as the incoming value.
      IncomingV = UndefValue::get(FieldTy);
    else if (IncomingTypes[Pred] == BackedgeUnknown)
      // If the incoming block has unknown state and it's a backedge which
      // haven't yet been processed, optimistically assume the value to be
      // unchanged in the loop. If this is not true we'll come here again and
      // will update the incoming with the new value.
      IncomingV = VPHI;
    else {
      assert(IncomingValues.find(Pred) != IncomingValues.end());
      IncomingV = IncomingValues[Pred];
    }
    VPHI->addIncoming(IncomingV, Pred);
  }

  return ExtendedValue(VPHI);
}

/// Returns true if the allocation was merged successfully and the merge has an
/// exact state.
bool State::mergeAllocation(VirtualContext &VContext, AllocationID ID,
                            const BasicBlock *BB, GetBlockOutState GetState) {
  // (1) Check that the allocation is present in all incoming states.
  const Allocation *SomeAlloc = nullptr;
  const BasicBlock *MissedInPredBlock = nullptr;
  // We assume that the following properties are the same for incoming allocs
  auto HaveEqualInvariantProperties = [](const Allocation &A,
                                         const Allocation &B) {
    if (!(A.ID == B.ID && A.NewInstruction == B.NewInstruction &&
          A.KlassID == B.KlassID))
      return false;
    if (A.isPHIMergedAllocation()) {
      // either alloc state is equal, or one is subset ("older") of another
      return (A.ArrayLength == B.ArrayLength &&
              A.ContributingAllocations == B.ContributingAllocations) ||
             llvm::set_is_subset(A.ContributingAllocations,
                                 B.ContributingAllocations) ||
             llvm::set_is_subset(B.ContributingAllocations,
                                 A.ContributingAllocations);
    } else {
      return A.ArrayLength == B.ArrayLength &&
             A.ContributingAllocations == B.ContributingAllocations;
    }
  };
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    if (auto Alloc = PredBlockState.getValue()->getAllocation(ID)) {
      assert((!SomeAlloc || HaveEqualInvariantProperties(*SomeAlloc, *Alloc)) &&
             "Inconsistent incoming alloc state");
      if (SomeAlloc && SomeAlloc->isPHIMergedAllocation()) {
        if (SomeAlloc->ContributingAllocations.size() >=
            Alloc->ContributingAllocations.size())
          continue;
      }
      SomeAlloc = Alloc;
    } else
      MissedInPredBlock = Pred;
    if (MissedInPredBlock && SomeAlloc)
      break;
  }

  assert(SomeAlloc &&
         "There should be at least one pred with this allocation!");

  if (MissedInPredBlock) {
    LLVM_DEBUG(dbgs() << "Merging allocation %"
                      << SomeAlloc->NewInstruction->getName() << " at block "
                      << BB->getName() << " found it untracked in predecessor "
                      << MissedInPredBlock->getName() << ":\n"
                      << "  escaped allocation: ";
               SomeAlloc->dumpInstruction(););
    return false;
  }

  // (2) Merge the allocation state
  auto MergedState =
    ExactAllocationState::getMergedAllocationState(VContext, ID,
      [ID] (const BasicBlock *BB) { return ID; },
      BB, GetState);

  if (!MergedState) {
    LLVM_DEBUG(dbgs() << "Merging allocation %"
                      << SomeAlloc->NewInstruction->getName() << " at block "
                      << BB->getName() << " could not merge its state:\n");
    if (TrackExactAllocationState && !TrackNonExactState) {
      LLVM_DEBUG(dbgs() << "  escaped allocation: ";
                 SomeAlloc->dumpInstruction(););
      return false;
    }
    LLVM_DEBUG(dbgs() << "  escaped allocation content: ";
               SomeAlloc->dumpInstruction(););
  }

  // (3) Add the allocation
  auto &MergedAllocation = addTrackedAllocation(
      SomeAlloc->ID, SomeAlloc->NewInstruction, SomeAlloc->KlassID,
      SomeAlloc->ArrayLength, SomeAlloc->ZeroInitializeFrom,
      SomeAlloc->ContributingAllocations);
  MergedAllocation.ExactState = MergedState;
  MergedAllocation.SymbolicState =
    SymbolicAllocationState::getMergedAllocationState(
      [ID] (const BasicBlock *BB) { return ID; },
      BB, GetState);

  // (4) Take an intersection of all tracked pointers from all incoming blocks
  // and set NeedsPublicationBarrier.
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    auto *Alloc = PredBlockState.getValue()->getAllocation(ID);
    assert(Alloc && "Should have the allocation in all incoming states!");
    // If at least one of the incoming states needs publication barrier
    // set the flag for the merged allocation.
    MergedAllocation.NeedsPublicationBarrier |= Alloc->NeedsPublicationBarrier;
    for (auto Ptr : Alloc->TrackedPointers)
      addTrackedPointer(Ptr,
                        *PredBlockState.getValue()->getTrackedPointer(Ptr));
  }

  return MergedState.has_value();
}

// Check that individual allocations can't be accessed after the merge.
//
// For example:
//   b = new A
//   c = new A
//   br %flag, label %blockB, lable %blockC
//
// blockB:
//   br label %merge
//
// blockC:
//   br label %merge
//
// merge:
//   p = phi %b, %c
//   p.f = valP
//
// If either b or c can be accessed after the merge we can no longer track the
// state p as a new allocation.
bool State::isPHIMergeLegal(const PHINode *ZeroOffsetPHI,
                            PHIMergeCandidateInfo &CandidateInfo,
                            const DominatorTree &DT,
                            GetBlockOutState GetState) {
  auto *BB = ZeroOffsetPHI->getParent();

  // (1) Collect all tracked pointers pointing to the allocations being merged
  SmallSet<ExtendedValue, 16> PointersToCheck;
  for (auto ID : *CandidateInfo.IncomingAllocations)
    for (auto *Pred : predecessors(BB)) {
      auto PredBlockState = GetState(Pred);
      if (PredBlockState.isKnownState())
        if (auto *Alloc = PredBlockState.getValue()->getAllocation(ID))
          for (auto Ptr : Alloc->TrackedPointers)
            PointersToCheck.insert(Ptr);
    }

  // (2) If a tracked pointer to one of the allocations is stored in some
  // tracked allocation - bail out. This pointer can be loaded and one of
  // the merged allocations can be accessed after the PHI.
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (PredBlockState.isKnownState())
      for (auto &A : PredBlockState.getValue()->Allocations) {
        // If there is no exact state then all pointers that were stored to
        // this allocation must have been escaped.
        if (A.second.ExactState)
          for (auto It : A.second.ExactState->FieldValues)
            if (PointersToCheck.count(It.second)) {
              LLVM_DEBUG(dbgs() << "Pointer to allocation is stored to memory."
                                   " Can't PHI-merge!\n");
              return false;
            }
      }
  }

  // (3) Check that none of the users collected on the previous step is reachable
  // from the PHI.
  SmallPtrSet<const User *, 16> Users;
  for (auto Ptr : PointersToCheck) {
    auto PtrV = Ptr.asValue();
    if (!PtrV)
      // Can't handle virtual value pointers for now.
      return false;
    auto *PtrI = dyn_cast<Instruction>(*PtrV);
    assert(PtrI && "We only expect to see Instructions here");
    if (!fsea::FlowSensitiveEAUtils::isPointerDeadThroughBlockEntry(
            PtrI, BB, [&](User *U) -> bool {
              // Skip PHIs which are candidates for the PHI-merge
              return CandidateInfo.PHIOffsets.count(dyn_cast<PHINode>(U));
            }))
      return false;
  }

  return true;
}

std::optional<ExtendedValue> State::getVPHIForMergedArrayLength(
    VirtualContext &VContext, AllocationID ID,
    const PHIMergeCandidateInfo::AllocationList *IncomingAllocations,
    const BasicBlock *BB, GetBlockOutState GetState) {
  auto ArrayLengthOffset = fsea::VMInterface::getVMIntegerConstant(
      BB->getContext(), "fsea.array_length_offset_in_bytes");
  if (!ArrayLengthOffset)
    return std::nullopt;

  SmallDenseMap<const BasicBlock *, ExtendedValue, 8> IncomingValues;
  SmallDenseMap<const BasicBlock *, FieldValueType, 8> IncomingTypes;
  Type *FieldTy = nullptr;

  auto AllocationsIt = IncomingAllocations->begin();
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (PredBlockState.isBackedgeUnknown())
      IncomingTypes[Pred] = BackedgeUnknown;
    else if (PredBlockState.isUnreachableUnknown())
      IncomingTypes[Pred] = UnreachableUnknown;
    else {
      IncomingTypes[Pred] = Initialized;
      auto *InState = PredBlockState.getValue();
      auto *A = InState->getAllocation(*AllocationsIt);
      // We should only come here if all incoming allocations share the same
      // KlassID and have different array lengths across different paths.
      // We can't have some inputs be arrays while others are not.
      assert(A->ArrayLength &&
             "All incoming allocations should have an array length!");
      auto ArrayLength = *A->ArrayLength;
      IncomingValues[Pred] = ArrayLength;
      if (!FieldTy)
        FieldTy = ArrayLength.getType();
      else if (FieldTy != ArrayLength.getType()) {
        // Incoming types mismatch!
        LLVM_DEBUG(dbgs() << "Cannot merge array lengths with different types: "
                          << *ArrayLength.getType() << " != " << *FieldTy
                          << "\n";);
        return std::nullopt;
      }
      AllocationsIt++;
    }
  }
  assert(FieldTy && "There should be at least one array length!");

  // Create or update an existing virtual PHI
  return getOrCreateVirtualPHIForField(VContext, BB, FieldTy, ID,
                                       *ArrayLengthOffset, IncomingValues,
                                       IncomingTypes);
}

bool State::tryMergeAllocationsAtPHI(
    VirtualContext &VContext, PHIMergeCandidateInfo &CandidateInfo,
    const DominatorTree &DT, GetBlockOutState GetState,
    SmallVectorImpl<AllocationID> &EscapeContent) {
  LLVM_DEBUG(dbgs() << "tryMergeAllocationsAtPHI, number of candidate PHIs = "
                    << CandidateInfo.PHIOffsets.size() << "\n");

  // We have a PHI-merge candidate - a list of allocations and PHI nodes which
  // merge tracked pointers to these allocations. In case of a successful merge
  // the result of the merge will be represented as a new tracked allocation in
  // the state. All PHI nodes from CandidateInfo will be registered as tracked
  // pointers to the new "allocation".

  // (1) If all incoming allocations are the same allocation we don't need to
  // threat this PHI as a merged allocation. It will become a tracked pointer
  // to this allocation during normal merge.
  auto FirstID = (*CandidateInfo.IncomingAllocations)[0];
  if (std::all_of(CandidateInfo.IncomingAllocations->begin(),
                  CandidateInfo.IncomingAllocations->end(),
                  [&](AllocationID ID) { return ID == FirstID; })) {
    LLVM_DEBUG(dbgs() << "All incoming values are the same allocation, "
                         "don't merge\n");
    return false;
  }

  // (2) Find a PHI which merges base pointers, i.e. tracked pointers with zero
  // offset. This PHI will be the new "allocation instruction". All other
  // PHIs will be marked as "derived" tracked pointers.
  auto FindZeroOffsetPHI = [&]() -> const PHINode * {
    for (auto It : CandidateInfo.PHIOffsets)
      if (It.second.value_or(-1) == 0)
        return It.first;
    return nullptr;
  };
  const PHINode *ZeroOffsetPHI = FindZeroOffsetPHI();
  if (!ZeroOffsetPHI) {
    LLVM_DEBUG(dbgs() << "There is no zero-offset PHI, can't merge\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "ZeroOffsetPHI = " << *ZeroOffsetPHI << "\n");

  // (3) To consider this PHI for PHI-merging we need the same klass ID
  // and array length across all incoming tracking pointers.
  const Value *SingleKlassID = nullptr;
  std::optional<std::optional<ExtendedValue>> SingleArrayLength;
  std::optional<std::optional<ExtendedValue>> SingleZeroInitializeFrom;

  auto AllocationsIt = CandidateInfo.IncomingAllocations->begin();
  for (auto *Pred : predecessors(ZeroOffsetPHI->getParent())) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    auto *InState = PredBlockState.getValue();
    AllocationID ID = *AllocationsIt;
    auto *A = InState->getAllocation(ID);
    if (AllocationsIt == CandidateInfo.IncomingAllocations->begin()) {
      SingleKlassID = A->KlassID;
      SingleArrayLength = A->ArrayLength;
      SingleZeroInitializeFrom = A->ZeroInitializeFrom;
    } else {
      if (SingleKlassID != A->KlassID) {
        LLVM_DEBUG(dbgs() << "Incoming allocations are incompatible\n");
        return false;
      }
      if (SingleArrayLength && *SingleArrayLength != A->ArrayLength)
        SingleArrayLength.reset();
      if (SingleZeroInitializeFrom &&
          *SingleZeroInitializeFrom != A->ZeroInitializeFrom)
        SingleZeroInitializeFrom.reset();
    }
    AllocationsIt++;
  }
  assert(AllocationsIt == CandidateInfo.IncomingAllocations->end() &&
         "Should have processed all allocations from the list!");
  assert(SingleKlassID && "There must be a klass ID!");

  AllocationID MergedAllocationID = createAllocationID(ZeroOffsetPHI);

  std::optional<ExtendedValue> ArrayLength;
  if (SingleArrayLength)
    ArrayLength = *SingleArrayLength;
  // If we come here all incoming allocations have the same KlassID, but
  // don't have the same array length. Generate a virtual PHI for the merged
  // length.
  else if (auto PHIMergedArrayLength = getVPHIForMergedArrayLength(
            VContext, MergedAllocationID, CandidateInfo.IncomingAllocations,
            ZeroOffsetPHI->getParent(), GetState))
    ArrayLength = PHIMergedArrayLength;
  else
    return false;

  // TODO: In case incoming allocations have different ZeroInitializeFrom
  // we can generate a virtual PHI for the merged value as it is done for array
  // length. For now it is left undefined preventing loading uninitialized
  // fields.
  std::optional<ExtendedValue> ZeroInitializeFrom;
  if (SingleZeroInitializeFrom)
    ZeroInitializeFrom = *SingleZeroInitializeFrom;

  // (4) PHI-merge is legal only if the individual allocations can't be
  // accessed after the merge point.
  if (!isPHIMergeLegal(ZeroOffsetPHI, CandidateInfo, DT, GetState)) {
    LLVM_DEBUG(dbgs() << "PHI-merging is not legal\n");
    return false;
  }

  // (5) Create merged allocation state
  auto GetAllocID = [&] (const BasicBlock *BB) {
    auto *InV = ZeroOffsetPHI->getIncomingValueForBlock(BB);
    auto PredBlockState = GetState(BB);
    assert(PredBlockState.isKnownState());
    auto *InState = PredBlockState.getValue();
    auto TP = InState->getTrackedPointer(InV);
    assert(TP);
    return TP->AllocID;
  };

  std::optional<ExactAllocationState> MergedState;
  // We can't express exact allocation state if the length is different across
  // different incoming paths:
  //
  //  a_bb:
  //    a = new char[10]
  //
  //  b_bb:
  //    b = new char[100]
  //    b[99] = 42;              ; Can't express in exact merged state
  //
  //  merge_bb:
  //    merge = phi a, b         ; Merged allocation
  //
  if (SingleArrayLength)
    MergedState = ExactAllocationState::getMergedAllocationState(
        VContext, MergedAllocationID, GetAllocID, ZeroOffsetPHI->getParent(),
        GetState);
  if (!MergedState) {
    LLVM_DEBUG(dbgs() << "Can't create merged allocation state\n");
    for (auto ID : *CandidateInfo.IncomingAllocations)
      EscapeContent.push_back(ID);
    if (!TrackNonExactState)
      return false;
  }

  // Compute the set of contributing allocations.
  SmallSet<AllocationID, 4> ContributingAllocations;
  ContributingAllocations.insert(MergedAllocationID);
  bool NeedsPublicationBarrier = false;

  // Add a union of ContributingAllocations for all incoming allocations.
  AllocationsIt = CandidateInfo.IncomingAllocations->begin();
  for (auto *Pred : predecessors(ZeroOffsetPHI->getParent())) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    auto *InState = PredBlockState.getValue();
    auto *A = InState->getAllocation(*AllocationsIt);
    ContributingAllocations.insert(A->ContributingAllocations.begin(),
                                   A->ContributingAllocations.end());
    NeedsPublicationBarrier |= A->NeedsPublicationBarrier;
    AllocationsIt++;
  }
  assert(AllocationsIt == CandidateInfo.IncomingAllocations->end() &&
         "Should have processed all allocations from the list!");

  // (6) Add the merged allocation into the state
  auto &MergedAllocation = addTrackedAllocation(
      MergedAllocationID, ZeroOffsetPHI, SingleKlassID, ArrayLength,
      ZeroInitializeFrom, ContributingAllocations);
  MergedAllocation.ExactState = MergedState;
  MergedAllocation.NeedsPublicationBarrier = NeedsPublicationBarrier;
  MergedAllocation.SymbolicState =
    SymbolicAllocationState::getMergedAllocationState(
      GetAllocID, ZeroOffsetPHI->getParent(), GetState);
  for (auto It : CandidateInfo.PHIOffsets)
    addTrackedPointer(It.first, TrackedPointer(MergedAllocationID, It.second));
  assert(getTrackedPointer(ZeroOffsetPHI)->AllocID == MergedAllocationID);
  return true;
}

SetVector<AllocationID> State::tryMergeAllocationsAtPHIs(
    VirtualContext &VContext, const BasicBlock *BB, const DominatorTree &DT,
    GetBlockOutState GetState, SmallVectorImpl<AllocationID> &EscapeContent) {
  LLVM_DEBUG(dbgs() << "tryMergeAllocationsAtPHIs " << BB->getName() << "\n");

  SetVector<AllocationID> MergedIDs;

  // Iterate over all PHIs in the BB and collect candidates for PHI-merging.
  // PHI-merge candidate is a set of PHI nodes which merge tracked pointers to
  // the allocations. As a result of a successful merge all PHIs in the
  // candidate will become tracked pointers to the merged allocation.
  //
  // For example:
  //
  //  a_bb:
  //    a = new A()
  //    a_8 = gep a, 8
  //    a_16 = gep a, 16
  //
  //  b_bb:
  //    b = new A()
  //    b_8 = gep b, 8
  //
  //  merge_bb:
  //    merge = phi a, b         ; Merged allocation
  //    merge_8 = phi a_8, b_8   ; tracked pointer, merge +8
  //    merge_f = phi a_16, b_8  ; tracked pointer, merge +unknown
  //
  // Here merge, merge_8, merge_f form one PHI-merge candidate.

  // Uniqify the lists of allocations we encounter. We'll use the pointer in
  // this set as a key for CandidatePHIs map. Note, we rely on the fact that
  // insertion into a set doesn't invalidate references to the existing
  // elements.
  std::set<PHIMergeCandidateInfo::AllocationList> AllocationLists;
  DenseMap<const PHIMergeCandidateInfo::AllocationList *, PHIMergeCandidateInfo>
      CandidatePHIs;

  for (const PHINode &PN : BB->phis()) {
    if (!fsea::isGCPointer(PN))
      continue;

    // Look for the PHIs with all inputs being tracked pointers.
    // Collect the list of all incoming allocations.
    PHIMergeCandidateInfo::AllocationList IncomingAllocationIDs;
    // If all incoming tracked pointers share the same offset
    // remember this offset. In the example above:
    //
    //  merge_bb:
    //    merge = phi a, b ; merged allocations: a, b; single offset = 0
    //    merge_8 = phi a_8, b_8 ; merged allocations: a, b; single offset = 8
    //    merge_f = phi a_16, b_8 ; merged allocations: a, b; no single offset
    //
    std::optional<int64_t> SingleOffset;
    // Lamda is used to break from the outer loop (over PHIs) from inside
    // the inner loop (over predecessors) via return.
    auto ProcessPHI = [&]() {
      bool First = true;
      for (auto *Pred : predecessors(BB)) {
        auto PredBlockState = GetState(Pred);
        if (!PredBlockState.isKnownState())
          continue;
        auto *InState = PredBlockState.getValue();
        auto *InV = PN.getIncomingValueForBlock(Pred);
        auto TP = InState->getTrackedPointer(InV);
        if (!TP)
          return false;
        IncomingAllocationIDs.push_back(TP->AllocID);
        if (First) {
          SingleOffset = TP->Offset;
          First = false;
        } else if (SingleOffset != TP->Offset)
          SingleOffset = std::nullopt;
      }
      return !IncomingAllocationIDs.empty();
    };
    if (!ProcessPHI())
      continue; // This PHI is not a good candidate for merging...

    const PHIMergeCandidateInfo::AllocationList *Ref =
        &*AllocationLists.insert(IncomingAllocationIDs).first;
    if (CandidatePHIs.find(Ref) == CandidatePHIs.end())
      CandidatePHIs[Ref].IncomingAllocations = Ref;
    assert(CandidatePHIs[Ref].IncomingAllocations == Ref);
    CandidatePHIs[Ref].PHIOffsets[&PN] = SingleOffset;
  }

  assert(CandidatePHIs.size() == AllocationLists.size());
  LLVM_DEBUG(dbgs() << "\tnumber of candidates = "
                    << CandidatePHIs.size() << "\n");
  for (auto &Candidates : CandidatePHIs)
    if (tryMergeAllocationsAtPHI(VContext, Candidates.second, DT, GetState,
                                 EscapeContent))
      for (auto ID : *Candidates.second.IncomingAllocations)
        MergedIDs.insert(ID);

  return MergedIDs;
}

State State::merge(FlowSensitiveEscapeAnalysis &EA, const BasicBlock *BB,
                   const DominatorTree &DT, GetBlockOutState GetState) {
  State MergedState;

  // (1) Collect a union of allocations from all incoming states.
  SmallSet<AllocationID, 8> Allocations;
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    if (PredBlockState.getValue()->isEmpty())
      // If one of the incoming state is empty then we cannot get any better.
      return State();
    for (const auto &It : PredBlockState.getValue()->Allocations)
      Allocations.insert(It.first);
  }

  // (2) Model unescaped allocations merged at PHIs
  SmallVector<AllocationID, 8> EscapeContentAllocations;
  if (MergeAllocationsAtPHIs) {
    auto MergedIDs =
      MergedState.tryMergeAllocationsAtPHIs(EA.VContext, BB, DT, GetState,
                                            EscapeContentAllocations);
    for (auto ID : MergedIDs)
      Allocations.erase(ID);
  }

  // (3) For every allocation from the union try merging the allocation into
  // the resulting state.
  for (auto ID : Allocations)
    if (!MergedState.mergeAllocation(EA.VContext, ID, BB, GetState))
      // for allocations that failed to merge we need to escape all their
      // contents yet we need to wait until all other allocations are merged in
      EscapeContentAllocations.push_back(ID);

  // (4) As a last step escape all the allocations that were "transitively
  // escaped".
  for (auto *Pred : predecessors(BB)) {
    auto PredBlockState = GetState(Pred);
    if (!PredBlockState.isKnownState())
      continue;
    for (auto ID : EscapeContentAllocations)
      if (const Allocation *A =
              PredBlockState.getValue()->getAllocation(ID)) {
        auto Closure =
            PredBlockState.getValue()->getAllocationContentClosure(A);
        for (auto EscapeID : Closure)
          MergedState.escape(EscapeID);
      }
  }

  // (5) Apply IR and virtual phis in no particular order.
  for (const PHINode &PN : BB->phis())
    if (fsea::isGCPointer(PN))
      MergedState.applyPhi(&PN, *BB, GetState);
  for (const auto *VPHI : EA.VContext.vphis(BB))
    if (fsea::isGCPointerType(VPHI->getType()))
      MergedState.applyPhi(VPHI, *BB, GetState);

  // (6) Find the last available deopt state.
  auto GetLastAvailableDeoptState = [&]() -> std::optional<DeoptState> {
    std::optional<DeoptState> SingleDeoptState;
    for (auto *Pred : predecessors(BB)) {
      auto PredBlockState = GetState(Pred);
      if (!PredBlockState.isKnownState()) {
        if (CanReexecuteExpensiveInstuctionsOnDeopt)
          continue;
        else
          return std::nullopt;
      }
      auto &PredBlockDeoptState =
        PredBlockState.getValue()->getLastAvailableState();
      if (!PredBlockDeoptState)
        return std::nullopt;
      if (!SingleDeoptState)
        SingleDeoptState = PredBlockDeoptState;
      else if (SingleDeoptState != PredBlockDeoptState)
        return std::nullopt;
    }
    return SingleDeoptState;
  };
  MergedState.LastDeoptState = GetLastAvailableDeoptState();

  return MergedState;
}

bool FlowSensitiveEscapeAnalysis::calculateBBStatesImpl(
    SmallPtrSetImpl<const BasicBlock *> &InitialWorklist,
    std::optional<unsigned> OptimisticIterations) {
  auto RPON = [&](const BasicBlock *BB) {
    auto BlockState = BlockRPON.find(BB);
    assert(BlockState != BlockRPON.end() && "Block must be reachable");
    return BlockState->second;
  };

  auto CMP = [&](const BasicBlock *B1, const BasicBlock *B2) {
    return RPON(B1) < RPON(B2);
  };

  auto IsVisited = [&](const BasicBlock *BB) {
    return BlockStates.find(BB) != BlockStates.end();
  };

  DenseMap<const BasicBlock *, unsigned> VisitedCounts;
  // This set is used instead of a priority queue because we want to avoid
  // scheduling same blocks twice.
  std::set<const BasicBlock *, decltype(CMP)> Worklist(CMP);

  auto PushWork = [&](const BasicBlock *BB) { Worklist.emplace(BB); };

  auto PopWork = [&]() -> const BasicBlock * {
    if (Worklist.empty())
      return nullptr;
    const BasicBlock *Top = *Worklist.begin();
    Worklist.erase(Top);
    return Top;
  };

  // By default the method calculateBBStates() runs the mode defined by the
  // OptimisticIterations args. In the optimistic mode to get the fixed point
  // states correct we must complete iterations. That is because the result
  // calculation optimistically ignores unknown or accepts speculative states
  // from some edges and only at the fixed point we know that all possible
  // state effects are processed. Number of optimistic mode iterations may be
  // very big as every back edge can bring a new state change that can make
  // the whole loop be re-iterated.
  // To cope with the non-linear complexity of the optimistic mode state
  // calculation a pessimistic mode is introduced. In this mode all unknown
  // states are treated as having the worst possible effect. This results
  // in the most pessimistic state taken at merge points if at least one edge
  // brings unknown state.
  // Running in the pessimistic mode from the beginning does not produce
  // speculative states, so all states are correct once computed.
  // The pessimistic mode guarantees that calculateBBStates() finishes when
  // each reachable basic block is processed once.
  auto GetBlockState = [&](const BasicBlock *BB) -> BlockOutState {
    const auto BlockStateIt = BlockStates.find(BB);
    if (BlockStateIt != BlockStates.end())
      return BlockOutState::getKnownState(&BlockStateIt->second.Out);

    if (BlockRPON.find(BB) == BlockRPON.end())
      return BlockOutState::getUnreachableUnknownState();

    // State has not been calculated yet (back branch).
    // In pessimistic mode return empty State (that is bottom - the worst
    // possible state). In optimistic mode return None (that is top - the most
    // compatible state).
    if (OptimisticIterations)
      return BlockOutState::getBackedgeUnknownState();

    static State EmptyState;
    return BlockOutState::getKnownState(&EmptyState);
  };

  // Start from the given blocks.
  Worklist.insert(InitialWorklist.begin(), InitialWorklist.end());

  unsigned BlocksProcessedCount = 0;
  LLVM_DEBUG(dbgs() << BlocksProcessedCount << ": Started "
                    << (OptimisticIterations ? "optimistic" : "pessimistic")
                    << " mode.\n");

  // Iterate till a fixed point is reached.
  while (const BasicBlock *BB = PopWork()) {
    ++BlocksProcessedCount;
    assert((OptimisticIterations || BlocksProcessedCount <= BlockRPON.size()) &&
           "Pessimistic merge mode must finish once each block is processed");

    LLVM_DEBUG(dbgs() << BlocksProcessedCount
                      << ": Calculating allocation state for block "
                      << BB->getName() << "\n");

    auto BBS = BlockStates.find(BB);
    bool Visited = BBS != BlockStates.end();
    assert((OptimisticIterations || !Visited) &&
           "In pessimistic mode each block must be visited once");

    unsigned &VisitedCount = VisitedCounts[BB];
    if (OptimisticIterations && ++VisitedCount > *OptimisticIterations) {
      // We failed to compute the analysis within the given limit of iterations.
      // As described in BlockOutState comment the States we compute during the
      // analysis form a lattice, so the analysis should converge eventually.
      // The number of iterations it will take for analysis to converge is
      // limited by the height of the lattice. We expect that in practice for
      // most cases we only need a few iterations to converge. The problem is
      // that theoretically the height of the lattice can be really large.
      // So the iteration limit is a protections from an adversarial scenario.
      // Our fallback strategy is simple - drop everything and recompute the
      // analysis in pessimistic mode.
      //
      // We can be more accurate in this scenario. For example, if the function
      // has several outer loop and only one of them exhibits adversarial
      // behavior we can do pessimistic analysis only for the loops which fail
      // to converge within the given iteration limit. We don't known if this
      // matters in practice, so for now just reset everything and return false
      // indicating a failure.
      LLVM_DEBUG(dbgs() << BlocksProcessedCount
                        << ": Iteration limit reached for block "
                        << BB->getName() << "\n");
      clear();
      return false;
    }

    // Merge inputs.
    State IS = State::merge(*this, BB, DT, GetBlockState);
    if (Visited && BBS->second.In == IS) {
      LLVM_DEBUG(
          dbgs() << BlocksProcessedCount
                 << ":   no need to re-calculate allocation state for block "
                 << BB->getName() << ".\n");
      continue;
    }

    if (!Visited)
      // Create empty states.
      BBS = BlockStates.try_emplace(BB, BasicBlockState()).first;

    assert(IsVisited(BB));

    BasicBlockState &BlockState = BBS->second;
    BlockState.In = IS; // Make a copy.

    StateInstVisitor Visitor(IS, *this, false);
    for (auto &I : make_range(BB->getFirstNonPHI()->getIterator(), BB->end()))
      Visitor.visit(const_cast<Instruction *>(&I));

    // The first iteration or a new output state might give something new.
    if (Visited && BlockState.Out == IS) {
      LLVM_DEBUG(
          dbgs()
          << BlocksProcessedCount
          << ":   no need to schedule successors re-calculation for block "
          << BB->getName() << ".\n");
      continue;
    }

    BlockState.Out = std::move(IS);

    // Schedule successors.
    for (auto *Succ : successors(BB))
      // In optimistic mode all successors are scheduled.
      // In pessimistic mode every block is scheduled once.
      if (OptimisticIterations || !IsVisited(Succ)) {
        PushWork(&*Succ);
        LLVM_DEBUG(dbgs() << BlocksProcessedCount << ":   scheduled successor "
                          << BB->getName() << " -> " << Succ->getName()
                          << "\n");
      }
  }

#ifndef NDEBUG
  for (const auto &B : BlockRPON)
    assert(IsVisited(B.first) && "All reachable blocks must be visited.");
#endif

  return true;
}

void FlowSensitiveEscapeAnalysis::calculateBlockRPON() {
  assert(BlockRPON.empty() && "Must be empty");
  ReversePostOrderTraversal<const Function *> RPOT(&F);
  for (auto BI = RPOT.begin(); BI != RPOT.end(); ++BI)
    BlockRPON.insert(std::make_pair(*BI, BlockRPON.size()));
}

void FlowSensitiveEscapeAnalysis::calculateBBStates() {
  SmallPtrSet<const BasicBlock *, 1> InitialWorklist = {&F.getEntryBlock()};
  calculateBBStates(InitialWorklist);
}

void FlowSensitiveEscapeAnalysis::calculateBBStates(
    SmallPtrSetImpl<const BasicBlock *> &InitialWorklist) {
  if (BlockRPON.empty())
    calculateBlockRPON();
  if (OptAllocOptimisticMerge) {
    if (calculateBBStatesImpl(InitialWorklist, IterationLimit.getValue()))
      return;
    LLVM_DEBUG(dbgs() << "Retry in globally pessimistic mode\n");
  }
  bool Result =
      calculateBBStatesImpl(InitialWorklist, /*OptimisticIterations=*/std::nullopt);
  assert(Result && "Pessimistic mode must succeed!");
  (void) Result;
}


void FlowSensitiveEscapeAnalysis::verify(bool AfterMaterialization) {
#ifndef NDEBUG
  for (auto &BB : F) {
    auto It = BlockStates.find(&BB);
    if (It == BlockStates.end())
      continue;

    auto S = It->second.In;
    StateInstVisitor Visitor(S, *this);
    for (auto &I : make_range(BB.getFirstNonPHI()->getIterator(), BB.end()))
      Visitor.visit(const_cast<Instruction *>(&I));

    if (AfterMaterialization)
      // Check that analysis state is a fixed point and the first
      // transformation stage did not break the analysis results.
      //
      // Note that we have materialized some of the virtual values as this point.
      // This should not have affected the state in a meaningful way, but we could
      // have added some extra tracked pointers.
      //
      // E.g. let's say we materialized a virtual load from a tracked allocation.
      // We added a gep and abitcast to compute the address to the loaded value.
      // These instructions are now tracked pointers!
      //
      // So, use a special equality check which considers states equal even if the
      // other state has extra tracked pointers.
      assert(It->second.Out.isEquivalentAfterMaterialize(S) &&
             "Unexpected output state");
    else
      assert(It->second.Out == S && "Unexpected output state");
  }
  VContext.verify();
#endif // NDEBUG
}

bool FlowSensitiveEscapeAnalysis::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &Inv) {
  // We need to invalidate if we have either failed to preserve this analyses
  // result directly or if any of its dependencies have been invalidated.
  auto PAC = PA.getChecker<llvm::FlowSensitiveEA>();
  if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>())
    return true;

  return Inv.invalidate<LazyValueAnalysis>(F, PA) ||
         Inv.invalidate<DominatorTreeAnalysis>(F, PA) ||
         Inv.invalidate<AAManager>(F, PA);
}

bool State::escape(SmallVectorImpl<ExtendedValue> &&Values) {
  bool Changed = false;

  auto Closure = getAllocationClosure(std::move(Values));
  for (AllocationID ID : Closure)
    if (Allocation *A = getAllocation(ID)) {
      LLVM_DEBUG(dbgs() << "  escaped allocation: "; A->dumpInstruction());
      remove(ID);
      Changed = true;
    }

  return Changed;
}

bool State::escape(AllocationID ID) {
  auto *A = getAllocation(ID);
  if (!A)
    return false;

  SmallVector<ExtendedValue, 16> Worklist;
  Worklist.emplace_back(A->NewInstruction);
  bool Escaped = escape(std::move(Worklist));
  assert(Escaped && "Must escape the given allocation");
  return Escaped;
}

bool State::escapeContent(Allocation *A) {
  if (!A->ExactState)
    // Can't hold tracked pointers if exact state is unknown, so there is
    // nothing to escape.
    return false;
  SmallVector<ExtendedValue, 16> Worklist;
  for (auto &FV : A->ExactState->FieldValues)
    Worklist.emplace_back(FV.second);
  return escape(std::move(Worklist));
}

bool State::markContentUntrackable(AllocationID ID) {
  auto *A = getAllocation(ID);
  if (!A->ExactState)
    return false;
  if (!TrackNonExactState)
    return escape(ID);

  bool ChangeMade = escapeContent(ID);
  // Requery the allocations because escapeContent may have escaped the
  // allocation itself (if the allocation contains a self-reference as a field).
  A = getAllocation(ID);
  if (!A)
    return ChangeMade;

  A->ExactState = std::nullopt;
  return true;
}

std::optional<fsea::FieldInfo>
Allocation::getFieldInfo(int64_t Offset, Type *Ty) const {
  assert(Ty->isSized() && "Expected to be sized");

  auto &DL = NewInstruction->getModule()->getDataLayout();
  unsigned SizeInBits = DL.getTypeSizeInBits(Ty);
  if (SizeInBits % 8 != 0) {
    LLVM_DEBUG(dbgs() << "Unsupported size in bits: kid=" << KlassID
                      << ", offset: +" << Offset << "\n");
    return std::nullopt;
  }
  unsigned SizeInBytes = SizeInBits / 8;
  return getFieldInfo(Offset, SizeInBytes);
}

std::optional<fsea::FieldInfo>
Allocation::getFieldInfo(int64_t Offset, unsigned SizeInBytes) const {
  auto MaybeKID = TypeUtils::runTimeToCompileTimeKlassID(KlassID);
  if (!MaybeKID)
    // We don't know object layout if the type is not known statically.
    return std::nullopt;

  auto FieldInfo = fsea::VMInterface::getFieldInfoAtOffset(
      NewInstruction->getContext(),
      fsea::TypeUtils::JavaType(*MaybeKID, true /*Exact*/),
      /*NewAllocation=*/true, Offset);
  if (!FieldInfo || FieldInfo->getSizeInBytes() != SizeInBytes) {
    LLVM_DEBUG(dbgs() << "Field doesn't match object layout: kid=" << *MaybeKID
                      << ", offset: +" << Offset << "\n");
    return std::nullopt;
  }
  return FieldInfo;
}

std::optional<ExtendedValue> Allocation::getInitialFieldValue(int64_t Offset,
                                                              Type *Ty) const {
  auto FieldInfo = getFieldInfo(Offset, Ty);
  if (!FieldInfo) {
    LLVM_DEBUG(dbgs() << "No field info for : kid=" << KlassID
                      << ", offset: +" << Offset << "\n");
    return std::nullopt;
  }
  // For now we can only handle zero initialized fields
  if (!FieldInfo->getKnownZero().isAllOnes()) {
    LLVM_DEBUG(dbgs() << "Non zero-initialized field : kid=" << KlassID
                      << ", offset: +" << Offset << "\n");
    return std::nullopt;
  }

  if (!isArray())
    return ExtendedValue(Constant::getNullValue(Ty));

  if (!ZeroInitializeFrom)
    return std::nullopt;

  // Must be const and ZeroInitializeFrom + HeaderSize <= Offset.
  auto ZeroFromV = ZeroInitializeFrom->asValue();
  if (!ZeroFromV)
    return std::nullopt;

  NewArrayDesc ArrayDesc(this);
  auto HeaderSize = ArrayDesc.getArrayHeaderSize();
  if (!HeaderSize)
    return std::nullopt;

  if (auto *ZeroFromC = dyn_cast<ConstantInt>(*ZeroFromV))
    if (ZeroFromC->getSExtValue() + *HeaderSize <= Offset)
      return ExtendedValue(Constant::getNullValue(Ty));

  return std::nullopt;
}

bool DeoptStateInstVisitor::invalidatesDeoptState(
    std::optional<State::DeoptState> DeoptState, Instruction &I) {
  if (!DeoptState)
    return false;

  if (!I.mayWriteToMemory())
    return false; // no-side effects, can re-execute

  if (fsea::isNewAllocation(I))
    return false; // allocations are safe to re-execute

  if (fsea::isFinalPublicationBarrier(I))
    return false;

  // These are just compiler markers despite reported memory effects
  if (auto *II = dyn_cast<IntrinsicInst>(&I))
    if (II->getIntrinsicID() == Intrinsic::invariant_start ||
        II->getIntrinsicID() == Intrinsic::invariant_end ||
        II->getIntrinsicID() == Intrinsic::assume)
      return false;

  if (fsea::isGCSafepointPoll(I))
    return false;

  auto ModificationInvalidatesDeoptState = [&] (Value *Ptr) {
    // If modified object is not a tracked (unescaped) allocation,
    // the modification invalidates the state.
    auto TP = S.getTrackedPointer(Ptr);
    if (!TP)
      return true;

    // The modified object is unescaped! Check that the object is not a part of
    // the deopt state we are about to reuse. For example:
    //
    //   a = new A() [ deopt bci=1, ... ]
    //   ...side effect...
    //   b = new A() [ deopt bci=3, ... a, ... ]
    //   a.f = 5
    //   c = new A() [ deopt bci=5, ... ]
    //
    // Even though the store to a.f changes unescaped memory it invalidates
    // the deopt state at bci=3 as it modifies the object 'a' which is a part
    // of this deopt state.
    if (DeoptState->refersToAllocation(*TP))
      return true;

    return false;
  };

  // Recognize some of the side-effecting operations.
  if (auto *SI = dyn_cast<StoreInst>(&I))
    return ModificationInvalidatesDeoptState(SI->getPointerOperand());

  if (auto *AMI = dyn_cast<AtomicMemCpyInst>(&I)) {
    if (!CanReexecuteExpensiveInstuctionsOnDeopt) {
      // If this is a memcpy into unescaped memory we can reuse the deopt state
      // across it. But if we hit a deopt with a replaced state we will need to
      // reexecute the memcpy in the interpreter. If this is a very long memcpy
      // this might have an undesirable impact on deoptimization latency. To
      // avoid this problem, check that the memcpy is short and invalidate the
      // state otherwise.
      //
      // NOTE: this is a purely theoretical concern. We don't have evidence that
      // this is problematic in practice.
      if (!AMI->hasFnAttr("gc-leaf-function"))
        return true;
    }
    return ModificationInvalidatesDeoptState(AMI->getRawDest());
  }

  return true;
}

bool DeoptStateInstVisitor::canUseDeoptState(CallBase *Call) {
  if (!Call->getOperandBundle(LLVMContext::OB_deopt))
    return false;
  return fsea::isNewAllocation(*Call);
}

bool DeoptStateInstVisitor::visitInstruction(Instruction &I) {
  bool ChangeMade = false;
  if (S.LastDeoptState) {
    // If we have some state, check if the current instruction invalidates it.
    if (invalidatesDeoptState(S.LastDeoptState, I)) {
      LLVM_DEBUG(dbgs() << "Instruction invalidates deopt state: ";
                 I.dump(););
      S.LastDeoptState = std::nullopt;
      ChangeMade = true;
    }

    return ChangeMade;
  }

  if (auto Call = dyn_cast<CallBase>(&I))
    if (canUseDeoptState(Call)) {
      // If we don't have a state available, check if we can use the state of this
      // instruction.
      S.LastDeoptState.emplace(cast<CallBase>(&I), S);
      ChangeMade = true;
    }

  return ChangeMade;
}

bool ExactStateInstVisitor::applyAlias(const Instruction &I,
                                       Value *Op) {
  auto TP = S.getTrackedPointer(Op);
  if (!TP)
    return false;

  S.addTrackedPointer(&I, *TP);
  return true;
}

bool ExactStateInstVisitor::applyAtomicRMW(const AtomicRMWInst &ARMW,
                                           const TrackedPointer PtrTP) {
  if (!S.getAllocation(PtrTP)->ExactState)
    return false;

  if (auto FieldValue =
          S.getFieldValue(ARMW.getType(), PtrTP, 0, false /* IsTypeStrict */)) {
    AtomicRMWStoredValue *VARMW;
    if (auto *VV = VContext.getVirtualValue(ARMW)) {
      VARMW = VV->asAtomicRMWStoredValue();
      assert(VARMW);
    } else {
      VARMW = new AtomicRMWStoredValue(ARMW);
      VContext.setInstructionModel(ARMW,
                                   std::unique_ptr<InstructionModel>(
                                       new SingleValueInstructionModel(VARMW)));
    }

    if (FixedPointReached)
      assert(VARMW->CurrentFieldValue == *FieldValue);
    else
      VARMW->setCurrentFieldValue(*FieldValue);
    return applyStorePointerUse(ARMW, VARMW, PtrTP);
  }
  LLVM_DEBUG(dbgs() << "Escaping. AtomicRMW to uncertain field: "; ARMW.dump();
             dbgs() << "  of alloc: ";
             S.getAllocation(PtrTP)->dumpInstruction(););
  return S.markContentUntrackable(PtrTP.AllocID);
}

bool ExactStateInstVisitor::applyAtomicCmpXchg(const AtomicCmpXchgInst &ACXI,
                                               const TrackedPointer PtrTP) {
  if (!S.getAllocation(PtrTP)->ExactState)
    return false;

  if (auto FieldValue =
          S.getFieldValue(ACXI.getCompareOperand()->getType(), PtrTP)) {
    assert(!S.getTrackedPointer(*FieldValue) &&
           "AtomicCmpXchg cannot be applied to reference fields");
    CASStoredValue *VCAS;
    if (auto *VV = VContext.getVirtualValue(ACXI)) {
      VCAS = VV->asCASStoredValue();
      assert(VCAS);
    } else {
      VCAS = new CASStoredValue(ACXI);
      VContext.setInstructionModel(ACXI,
                                   std::unique_ptr<InstructionModel>(
                                       new SingleValueInstructionModel(VCAS)));
    }

    if (FixedPointReached)
      assert(VCAS->CurrentFieldValue == *FieldValue);
    else
      VCAS->setCurrentFieldValue(*FieldValue);
    return applyStorePointerUse(ACXI, VCAS, PtrTP);
  }

  LLVM_DEBUG(dbgs() << "Escaping. AtomicCmpXchg to unknown field: ";
             ACXI.dump(); dbgs() << "  of alloc: ";
             S.getAllocation(PtrTP)->dumpInstruction(););
  return S.markContentUntrackable(PtrTP.AllocID);
}

bool ExactStateInstVisitor::applyOperandUse(const Use &U,
                                            const TrackedPointer UseTP) {
  assert(isa<Instruction>(U.getUser()) &&
         "Must be called only for instruction operand uses");
  auto &I = *dyn_cast<Instruction>(U.getUser());

  assert(S.getAllocation(UseTP) && "Must be a valid allocation");

  bool Changed = false;

  // (1) Check whether the given use escapes the pointer to the tracked object.
  switch (fsea::getUseEscapeKind(&U)) {
  case fsea::UseEscapeKind::NoEscape:
    break;
  case fsea::UseEscapeKind::Alias:
    // Alias in terms of getUseEscapeKind is a may alias, not must. Currently we
    // only track must aliases, so we can not express an unknown
    // UseEscapeKind::Alias instruction as a tracked pointer.
    //
    // For example, a select between two tracked pointers will be reported as
    // UseEscapeKind::Alias for both operands:
    //   %alloc1 = new A()
    //   %alloc2 = new A()
    //   %ptr = select i1 %c, %alloc1, %alloc2
    // But we don't have a way to express %ptr as a tracked pointer which is a may
    // alias for two distinct allocations.
    //
    // Conservatively treat such uses as escapes for now.
    LLVM_DEBUG(dbgs() << "Escaping for unhandled aliasing instruction: ";
               I.dump(););
    S.escape(UseTP.AllocID);
    return true;
  case fsea::UseEscapeKind::Escape:
    LLVM_DEBUG(dbgs() << "Escaping pointer use: "; I.dump();
               dbgs() << "  of alloc: ";
               S.getAllocation(UseTP.AllocID)->dumpInstruction(););
    S.escape(UseTP.AllocID);
    // There is no need to check other effects on this tracked pointer if we
    // escaped the allocation.
    return true;
  }

  assert(S.getAllocation(UseTP) && "Must have exited if escaped");
  if (!S.getAllocation(UseTP)->ExactState)
    // No further read/write changes can happen with empty exact state.
    return Changed;

  // (2) If this instruction reads from the tracked pointer it may escape the
  // content of this allocation.
  if (I.mayReadFromMemory()) {
    auto ApplyReadFromMemory = [&]() {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        // Check whether the call site can read through this particular use of
        // the tracked pointer and escape the read content.
        if (CB->onlyWritesMemory(U.getOperandNo()))
          return false;

        if (CB->isBundleOperand(U.getOperandNo())) {
          // Deopt bundle use doesn't escape the content of the passed object.
          auto Bundle = CB->getOperandBundleForOperand(U.getOperandNo());
          if (Bundle.getTagID() == LLVMContext::OB_deopt)
            return false;
        }

        // TODO: For readonly callsites that return void we do not have to make
        // content escape but the current implementation pessimistically does
        // so.
      }

      auto *UsedAllocation = S.getAllocation(UseTP);
      LLVM_DEBUG(
          dbgs() << "Escaping content for unhandled reading instruction: ";
          I.dump(););
      return S.escapeContent(UsedAllocation);
    };
    if (ApplyReadFromMemory()) {
      if (!S.getAllocation(UseTP))
        return true; // Finish if the allocation escaped.
      Changed = true;
    }
  }

  assert(S.getAllocation(UseTP) && "Must have exited if escaped");

  // (3) If this instruction writes through the tracked pointer we need to
  // update the tracked content of the allocation.
  if (I.mayWriteToMemory()) {
    auto ApplyStoreToMemory = [&]() {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        // Check whether the call site can write through the tracked pointer and
        // modify the tracked content.
        if (CB->onlyReadsMemory(U.getOperandNo()))
          return false;

        if (CB->isBundleOperand(U.getOperandNo())) {
          // Deopt bundle use doesn't modify the content of the passed object.
          auto Bundle = CB->getOperandBundleForOperand(U.getOperandNo());
          if (Bundle.getTagID() == LLVMContext::OB_deopt)
            return false;
        }
      }

      // Content can be changed in an untrackable way.
      LLVM_DEBUG(dbgs() << "Marking content untrackable for unhandled writing "
                           "instruction: ";
                 I.dump(););
      return S.markContentUntrackable(UseTP.AllocID);
    };

    Changed |= ApplyStoreToMemory();
  }

  return Changed;
}

bool ExactStateInstVisitor::visitLoadInst(LoadInst &LI) {
  auto PtrTP = S.getTrackedPointer(LI.getPointerOperand());
  if (!PtrTP)
    return false;

  auto *PtrAllocation = S.getAllocation(*PtrTP);
  if (!PtrAllocation->ExactState)
    return false;

  if (!PtrTP->Offset) {
    LLVM_DEBUG(dbgs() << "Load from uncertain field: "; LI.dump();
               dbgs() << "  of alloc: "; PtrAllocation->dumpInstruction(););
    return S.escapeContent(PtrAllocation);
  }

  int64_t Offset = *PtrTP->Offset;
  if (!PtrAllocation->isTrackedField(Offset, LI.getType())) {
    LLVM_DEBUG(dbgs() << "Load from unknown field with offset +" << Offset
                      << " : ";
               LI.dump(); dbgs() << "  of alloc: ";
               PtrAllocation->dumpInstruction(););
    return S.escapeContent(PtrAllocation);
  }

  auto FieldValue = PtrAllocation->ExactState->getFieldValue(Offset);
  if (!FieldValue)
    return false;

  // If we are loading a value which is a tracked pointer, we should register
  // the load as a tracked pointer. This way we will track escapes and
  // modifications through the loaded pointer. For example:
  //   a = new A()
  //   a.self = a
  //   a' = load a.self    // a' is an alias for allocation new A()
  //   escape(a')
  auto FieldTP = S.getTrackedPointer(*FieldValue);
  if (!FieldTP)
    return false;

  S.addTrackedPointer(&LI, *FieldTP);
  return true;
}

bool State::isDereferenceablePointer(const TrackedPointer TP, Type *Ty) const {
  auto *Alloc = getAllocation(TP);
  assert(Alloc);
  auto *AllocInstr = Alloc->NewInstruction;

  if (!Ty->isSized())
    return false;

  auto &DL = Alloc->NewInstruction->getModule()->getDataLayout();
  int64_t Offset =
      TP.Offset.value() + DL.getTypeStoreSize(Ty).getFixedValue();

  if (auto MaybeKID = TypeUtils::runTimeToCompileTimeKlassID(Alloc->KlassID)) {
    std::optional<uint64_t> ArrayLen;
    fsea::TypeUtils::JavaType T(*MaybeKID, /*IsExact=*/true);

    if (Alloc->ArrayLength)
      if (auto MaybeArrayLengthV = Alloc->ArrayLength->asValue())
        if (auto *ArrayLengthC = dyn_cast<ConstantInt>(*MaybeArrayLengthV))
          ArrayLen = ArrayLengthC->getZExtValue();
    if (auto JTI =
            fsea::VMInterface::getJavaTypeInfo(Ty->getContext(), T, ArrayLen))
      if (Offset <= JTI->getObjectSize())
        return true;
  }

  APInt Size = APInt(8 * sizeof(Offset), Offset);
  return isDereferenceableAndAlignedPointer(AllocInstr, Align(), Size, DL);
}

bool ExactStateInstVisitor::applyStoreValueUse(const StoreInst &SI,
                                               const TrackedPointer ValueTP) {
  // We are storing a tracked pointer somewhere. Check if we are storing it into
  // a tracked location. In this case this store is not an escape.
  auto PtrTP = S.getTrackedPointer(SI.getPointerOperand());
  if (!PtrTP || !PtrTP->Offset) {
    LLVM_DEBUG(dbgs() << "Escaping. Store to memory: "; SI.dump();
               dbgs() << "  of alloc: ";
               S.getAllocation(ValueTP.AllocID)->dumpInstruction(););
    S.escape(ValueTP.AllocID);
    return true;
  }

  auto *PtrAllocation = S.getAllocation(PtrTP);
  if (!PtrAllocation->ExactState) {
    LLVM_DEBUG(dbgs() << "Escaping. Store to an allocation with "
                      << "unknown exact state: "; SI.dump();
               dbgs() << "  of alloc: ";
               S.getAllocation(ValueTP.AllocID)->dumpInstruction(););
    S.escape(ValueTP.AllocID);
    return true;
  }

  if (!S.isDereferenceablePointer(*PtrTP, SI.getValueOperand()->getType())) {
    LLVM_DEBUG(dbgs() << "Escaping. Store to non-dereferenceable pointer: ";
               SI.dump(); dbgs() << "  of alloc: ";
               S.getAllocation(ValueTP.AllocID)->dumpInstruction(););
    S.escape(ValueTP.AllocID);
    return true;
  }

  Type *ValueTy = SI.getValueOperand()->getType();
  int64_t Offset = *PtrTP->Offset;

  if (!PtrAllocation->isTrackedField(Offset, ValueTy)) {
    LLVM_DEBUG(dbgs() << "Escaping. Store to unknown field at offset +"
                      << Offset << " of type " << *ValueTy << ": ";
               SI.dump(); dbgs() << "  of alloc: ";
               S.getAllocation(ValueTP.AllocID)->dumpInstruction(););
    S.escape(ValueTP.AllocID);
    return true;
  }

  return false;
}

bool ExactStateInstVisitor::applyStorePointerUse(const Instruction &I,
                                                 ExtendedValue ValueOp,
                                                 const TrackedPointer PtrTP) {
  // We are storing something into a tracked allocation. Update the allocation
  // state accordingly.
  Type *ValueTy = ValueOp.getType();

  auto *PtrAllocation = S.getAllocation(PtrTP);
  if (!PtrAllocation->ExactState) {
    LLVM_DEBUG(dbgs() << "Exact allocation state is not tracked: "; I.dump();
               dbgs() << "  of alloc: "; PtrAllocation->dumpInstruction(););
    return false;
  }

  if (!PtrTP.Offset.has_value()) {
    LLVM_DEBUG(dbgs() << "Escaping. Store to uncertain field: "; I.dump();
               dbgs() << "  of alloc: "; PtrAllocation->dumpInstruction(););
    S.markContentUntrackable(PtrTP.AllocID);
    return true;
  }

  if (!S.isDereferenceablePointer(PtrTP, ValueTy)) {
    LLVM_DEBUG(dbgs() << "Escaping. Non-dereferenceable pointer: "; I.dump();
               dbgs() << "  of alloc: "; PtrAllocation->dumpInstruction(););
    S.markContentUntrackable(PtrTP.AllocID);
    return true;
  }

  int64_t Offset = *PtrTP.Offset;
  // Check that the accessed field is a known Java field. This guarantees that
  // there are no overlapping stores into the object.
  if (PtrAllocation->isTrackedField(Offset, ValueTy)) {
    PtrAllocation->ExactState->setFieldValue(Offset, ValueOp);
    return true;
  }

  // Try applying vector store.
  VectorType *VecType = dyn_cast<VectorType>(ValueTy);
  if (!VecType || !ValueOp.asValue()) {
    LLVM_DEBUG(
        dbgs() << "Store instruction does not recognize field at offset +"
               << Offset << " of type " << *ValueTy << ": ";
        I.dump(););
    S.markContentUntrackable(PtrTP.AllocID);
    return true;
  }

  auto ElemCount = VecType->getElementCount();
  if (ElemCount.isScalable()) {
    LLVM_DEBUG(
        dbgs() << "Vector Store instruction has scalable destination: offset +"
               << Offset << " of type " << *VecType << ": ";
        I.dump(););
    S.markContentUntrackable(PtrTP.AllocID);
    return true;
  }

  if (ElemCount.getKnownMinValue() == 0)
    return false; // Zero sized vector.

  unsigned ElemSizeInBytes =
      I.getModule()->getDataLayout().getTypeSizeInBits(VecType) /
      ElemCount.getKnownMinValue() / 8;

  // Recursively call applyStorePointerUse() for each of the vector elements.
  // TODO: Improve the vector handling to not change destination fields until
  // all vector elements are checked to be assignable.
  for (unsigned ElNo = 0; ElNo < ElemCount.getKnownMinValue(); ++ElNo) {
    TrackedPointer ElemPtr(PtrTP.AllocID,
                           *PtrTP.Offset + ElNo * ElemSizeInBytes);
    auto *V = findScalarElement(const_cast<Value *>(*ValueOp.asValue()), ElNo);
    if (!V) {
      LLVM_DEBUG(
          dbgs()
              << "Vector Store instruction does not recognize field at offset +"
              << ElemPtr.Offset << " of type " << *VecType << ": ";
          I.dump(););
      S.markContentUntrackable(PtrTP.AllocID);
      return true;
    }
    applyStorePointerUse(I, V, ElemPtr);
    if (!S.getAllocation(PtrTP))
      break; // The source has escaped.
  }
  return true;
}

std::optional<bool> Equal(std::optional<TrackedPointer> LHSTP,
                          std::optional<TrackedPointer> RHSTP) {
  if (!LHSTP && !RHSTP)
    // We are comparing two pointer we don't track.
    return std::nullopt;
  // ... at least one value is a tracked pointer

  if (!LHSTP || !RHSTP)
    // We are comparing a pointer to a non-escaped object with some other
    // pointer. They can't be equal.
    return false;
  // ... both values are tracked pointers

  if (LHSTP->AllocID != RHSTP->AllocID)
    // We are comparing tracked pointers to different allocations. Can't be
    // equal.
    return false;
  // ... both pointers are tracked pointers to the same allocation.

  if (!LHSTP->Offset || !RHSTP->Offset)
    // If any of the offsets is unknown we can't optimize.
    return std::nullopt;
  // ... both offsets are known.

  return LHSTP->Offset == RHSTP->Offset;
}

std::optional<bool> Equal(const State &S, ExtendedValue LHS,
                          ExtendedValue RHS) {
  if (LHS == RHS)
    return true;
  return Equal(S.getTrackedPointer(LHS), S.getTrackedPointer(RHS));
}


static Type *getMemIntrinsicElementType(const CallBase &CB, unsigned Idx) {
  if (auto *T = CB.getParamElementType(Idx))
    return T;

  auto *PT = cast<PointerType>(CB.getArgOperand(Idx)->getType());
  // FIXME: We need to move this API change upstream.
  assert(!PT->isOpaque() &&
          "Memory intrinsics must have elementtype set in opaque pointers mode");
  return PT->getNonOpaquePointerElementType();
}

Type *GetSrcPointerElementType(const AtomicMemCpyInst &AMI) {
  // We need to distinguish between pointer and non-pointer types. Pointer
  // values must be read as pointers so as to be reported and handled by the
  // GC correctly.
  Type *PtrElementType = getMemIntrinsicElementType(AMI, /*ARG_SOURCE=*/1);
  if (!fsea::isGCPointerType(PtrElementType))
    // Non-reference element type might come inaccurate but we need it to be of
    // the correct size.
    return Type::getIntNTy(AMI.getContext(), 8 * AMI.getElementSizeInBytes());

  assert(
      AMI.getElementSizeInBytes() * 8 ==
          AMI.getModule()->getDataLayout().getTypeSizeInBits(PtrElementType) &&
      "memcpy element size must conform with source elemen type");
  return PtrElementType;
}

/// If the atomic memcpy is analyzed then its state effect is applied and the
/// boolean returned value reflects if the state has been changed. Otherwise,
/// None is returned meaning that the memcpy instruction should be further
/// processed as an ordinary instruction.
std::optional<bool>
ExactStateInstVisitor::applyMemcpy(const AtomicMemCpyInst &AMI) {
  auto *LengthCI = dyn_cast<ConstantInt>(AMI.getLength());
  if (!LengthCI)
    return std::nullopt;

  // Check that there are not too many elements.
  uint64_t LengthInBytes = LengthCI->getZExtValue();
  uint32_t ElementSizeInBytes = AMI.getElementSizeInBytes();
  uint64_t NumElements = LengthInBytes / ElementSizeInBytes;
  if (NumElements >= ModelMemcpyMaxElements)
    return std::nullopt;

  if (NumElements == 0)
    return false; // Nothing to copy - no state effect.

  if (!AMI.getModule()->getDataLayout().isLegalInteger(ElementSizeInBytes * 8))
    return std::nullopt;

  const Use &SrcU = AMI.getRawSourceUse();
  auto SrcTP = S.getTrackedPointer(SrcU.get());
  const Use &DstU = AMI.getRawDestUse();
  auto DstTP = S.getTrackedPointer(DstU.get());

  bool Changed = false;

  if (Allocation *SrcAlloc = S.getAllocation(SrcTP))
    if (!SrcTP->Offset) {
      LLVM_DEBUG(
          dbgs() << "Content escaping use: non-constant src offset in memcpy"
                 << AMI << "\n");
      if (S.escapeContent(SrcAlloc))
        Changed = true;
    }

  Type *ElementType = GetSrcPointerElementType(AMI);

  auto IsDestContentTrackable = [&]() {
    Allocation *DstAlloc = S.getAllocation(DstTP);
    if (!DstAlloc || !DstAlloc->ExactState)
      return false;

    if (!DstTP->Offset) {
      LLVM_DEBUG(
          dbgs()
          << "Content destructing use: non-constant dst offset in memcpy: "
          << AMI << "\n");
      return false;
    }

    if (!isDereferenceableAndAlignedPointer(
            DstAlloc->NewInstruction, Align(),
            APInt(8 * sizeof(uint64_t), *DstTP->Offset + LengthInBytes),
            AMI.getModule()->getDataLayout())) {
      LLVM_DEBUG(
          dbgs()
          << "Content destructing use: non-dereferenceable destination (size="
          << (*DstTP->Offset + LengthInBytes) << "): " << AMI << "\n");
      return false;
    }

    assert(DstAlloc == S.getAllocation(DstTP));

    // Check if dst fields are all trackable.
    for (unsigned i = 0; DstAlloc && i < NumElements; i++) {
      auto DstOffset = *DstTP->Offset + i * ElementSizeInBytes;
      if (!DstAlloc->isTrackedField(DstOffset, ElementType)) {
        LLVM_DEBUG(
            dbgs()
            << "Content destructing use: untracked destination field at +"
            << DstOffset << ": " << AMI << "\n");
        return false;
      }
    }
    return true;
  };

  auto MayOverlap = [&]() {
    if (!S.getAllocation(DstTP) || !S.getAllocation(SrcTP) ||
        DstTP->AllocID != SrcTP->AllocID ||
        (SrcTP->Offset && DstTP->Offset &&
         uint64_t(std::abs(*SrcTP->Offset - *DstTP->Offset)) >=
             NumElements * ElementSizeInBytes))
      return false; // Proved no overlapping.

    LLVM_DEBUG(dbgs() << "Content destructing use: src-dest overlapping: "
                      << AMI << "\n");

    // Violation of the overlap condition leads to Undefined Behavior. A better
    // overlap handling would be to mark this basic block unreachable and ignore
    // its state effect. For now we just adhere to a pessimistic scenario.
    return true; // May overlap.
  };

  if (!IsDestContentTrackable() || MayOverlap()) {
    if (S.getAllocation(DstTP))
      Changed |= S.markContentUntrackable(DstTP->AllocID);

    // Escape the pointers which we copy into untracked dest.
    Allocation *SrcAlloc = S.getAllocation(SrcTP);
    if (!SrcAlloc || !SrcAlloc->ExactState)
      // If src escaped or has no exact state there is no content to escape
      return Changed;
    if (!SrcTP->Offset)
      // In this case we have already escaped the content of the source
      return Changed;

    for (unsigned i = 0; i < NumElements && SrcAlloc; i++) {
      assert(SrcAlloc->ExactState && "Don't expect unknown state!");
      auto SrcOffset = *SrcTP->Offset + i * ElementSizeInBytes;
      auto SrcFieldValue = SrcAlloc->ExactState->getFieldValue(SrcOffset);
      if (!SrcFieldValue)
        // Src field cannot contain a trackable value. If destination was
        // trackable this value would be put to the destination as initial value
        // (if zero-initialized) or VLoad.
        continue;
      auto SrcFieldTP = S.getTrackedPointer(*SrcFieldValue);
      if (!SrcFieldTP)
        continue; // No src tracked pointer - no loss.

      LLVM_DEBUG(dbgs() << "Escaping copy of a tracked value from src offset +"
                        << SrcOffset << " to untracked destination:\n";);
      if (!S.escape(SrcFieldTP->AllocID))
        llvm_unreachable("Must escape");
      Changed = true;
      SrcAlloc = S.getAllocation(SrcTP);
    }
    return Changed;
  }

  // Change fields of Dst.
  auto DstOffset = *DstTP->Offset;
  Allocation *DstAlloc = S.getAllocation(DstTP);
  assert(DstAlloc);

  Allocation *SrcAlloc = S.getAllocation(SrcTP);

  for (unsigned i = 0; i < NumElements; i++) {
    auto FieldOffset = i * ElementSizeInBytes;

    // Get src field value or virtual load.
    std::optional<ExtendedValue> SrcFieldValue;
    if (SrcAlloc && SrcTP->Offset && SrcAlloc->ExactState) {
      auto SrcOffset = *SrcTP->Offset + FieldOffset;
      SrcFieldValue = SrcAlloc->ExactState->getFieldValue(SrcOffset);
      if (!SrcFieldValue)
        SrcFieldValue = SrcAlloc->getInitialFieldValue(SrcOffset, ElementType);
    }
    if (!SrcFieldValue) {
      LoadedFields *MemCpyFields = nullptr;
      if (auto Fields = VContext.getInstructionModel(AMI)) {
        MemCpyFields = Fields->asLoadedFields();
        assert(MemCpyFields);
        auto Field = MemCpyFields->Entries.find(FieldOffset);
        if (Field != MemCpyFields->Entries.end()) {
          assert(*Field->second == VirtualLoad(AMI, ElementType,
                                               AMI.getRawSource(),
                                               FieldOffset));
          SrcFieldValue = ExtendedValue(Field->second.get());
        }
      } else {
        MemCpyFields = new LoadedFields();
        VContext.setInstructionModel(
            AMI, std::unique_ptr<InstructionModel>(MemCpyFields));
      }
      if (!SrcFieldValue) {
        VirtualLoad *VLoad =
            new VirtualLoad(AMI, ElementType, AMI.getRawSource(), FieldOffset);
        MemCpyFields->Entries.emplace(FieldOffset, VLoad);
        SrcFieldValue = ExtendedValue(VLoad);
      }
    }

    // Set dst field value.
    DstAlloc->ExactState->setFieldValue(DstOffset + FieldOffset,
                                        *SrcFieldValue);
  }

  assert(NumElements);
  return true;
}

bool ExactStateInstVisitor::applyNewAllocation(const Instruction &I) {
  assert((fsea::isNewObjectInstance(I) || fsea::isNewArray(I)) &&
         "expecting NewObjectInstance or NewArray here");
  assert(!S.getTrackedPointer(&I) &&
         "Allocation instruction should not be tracked yet");

  const Value *KlassID = nullptr;
  std::optional<ExtendedValue> ArrayLength;
  std::optional<ExtendedValue> ZeroInitializeFrom;
  if (fsea::isNewObjectInstance(I))
    KlassID = fsea::NewObjectInstance(I).getKlassID();
  else {
    assert(fsea::isNewArray(I) && "Don't expect anything else");
    fsea::NewArray NA = fsea::NewArray(I);
    KlassID = NA.getArrayKlassID();
    ArrayLength = NA.getLength();
    Value *ZeroInitializeFromV = NA.getZeroInitializeFrom();
    if (!ZeroInitializeFromV) {
      // TODO: remove support of fewer arguments of @fsea.new_array().
      ZeroInitializeFromV =
          ConstantInt::get(Type::getInt64Ty(I.getContext()), 0);
    }
    ZeroInitializeFrom = ZeroInitializeFromV;
  }

  if (!isa<ConstantInt>(KlassID)) {
    LLVM_DEBUG(dbgs() << "Skipped new (non-const kid): "; I.dump(););
    return false;
  }

  S.addTrackedAllocation(&I, KlassID, ArrayLength, ZeroInitializeFrom);
  return true;
}

bool ExactStateInstVisitor::applyInvariantStart(
    const IntrinsicInst &InvariantStartCall) {
  // Filter out the situation when the result of an invariant.start
  // might be passed to invariant.end. Invariant.end is not used for
  // Java code, so it's here merely for completeness. In other words
  // we support only endless invariant ranges.
  if (!InvariantStartCall.hasNUses(0))
    return false;

  auto *InvPtr = InvariantStartCall.getArgOperand(1);
  auto TP = S.getTrackedPointer(InvPtr);
  if (!TP)
    return false;
  if (!TP->Offset) {
    LLVM_DEBUG(dbgs() << "Skipped invariant start on unknown offset: ";
               InvariantStartCall.dump(););
    return false;
  }

  auto Alloc = S.getAllocation(TP);
  if (!Alloc || !Alloc->ExactState)
    return false;

  auto *InvSizeInBytes =
    dyn_cast<ConstantInt>(InvariantStartCall.getArgOperand(0));
  if (!InvSizeInBytes) {
    LLVM_DEBUG(dbgs() << "Skipped invariant start with unknown size: ";
               InvariantStartCall.dump(););
    return false;
  }
  // According to langref -1 is a special case for variable size
  if (InvSizeInBytes->isMinusOne()) {
    LLVM_DEBUG(dbgs() << "Skipped invariant start with variable size: ";
               InvariantStartCall.dump(););
    return false;
  }
  int64_t Offset = *TP->Offset;
  if (!Alloc->isTrackedField(Offset, InvSizeInBytes->getZExtValue())) {
    // TODO: theoretically we can mark multiple fields using one invariant
    // start call. This is not supported for now.
    LLVM_DEBUG(dbgs() << "Skipped invariant start on unknown field: ";
               InvariantStartCall.dump(););
    return false;
  }

  Alloc->ExactState->markInvariantField(Offset);
  return true;
}

bool ExactStateInstVisitor::visitCallBase(CallBase &I) {
#ifndef NDEBUG
  // Tracked pointers being bundle operands should escape unless the bundle has
  // known non-escaping effect on its operands (like the deopt bundle). Instead
  // of imposing any particular order of operand processing here we just assert
  // that there is no tracked pointers among unknown bundle operands.
  //
  // Here is an example where processing order matters:
  //     store %obj1 -> %src
  //     store %obj2 -> %dst
  //     call void @memcpy(%dst, %src, i32 8, i32 8) [ "op-bundle"(%dst) ]
  //
  // if op-bundle makes content of %dst escape before state effect of memcpy is
  //   applied then %obj2 escapes as a content of %dst
  // if op-bundle makes content of %dst escape after state effect of memcpy is
  //   applied then %obj1 escapes as a content of %dst
  // if op-bundle does not make content of %dst escape
  //   then neither %obj1 nor %obj2 escape.
  if (I.hasOperandBundlesOtherThan({LLVMContext::OB_deopt}))
    for (unsigned i = I.getBundleOperandsStartIndex();
         i != I.getBundleOperandsEndIndex(); i++) {
      auto Bundle = I.getOperandBundleForOperand(i);
      if (Bundle.getTagID() != LLVMContext::OB_deopt)
        assert(!S.getTrackedPointer(I.getOperand(i)));
    }
#endif

  if (fsea::isNewObjectInstance(I) || fsea::isNewArray(I))
    return applyNewAllocation(I);

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    if (II->getIntrinsicID() == Intrinsic::invariant_start)
      return applyInvariantStart(*II);
    else if (II->getIntrinsicID() == Intrinsic::invariant_end)
      return false;
  }

  if (ModelMemcpy)
    if (auto *AMI = dyn_cast<AtomicMemCpyInst>(&I))
      if (auto StateChanged = applyMemcpy(*AMI))
        return *StateChanged;

  if (fsea::isGetKlassID(I)) {
    auto *Object = fsea::GetKlassID(I).getValueArg();
    if (auto ObjectTP = S.getTrackedPointer(Object))
      if (ObjectTP->isAlias())
        return false;
  }

  if (fsea::isMonitorEnter(I) || fsea::isMonitorEnterThreadLocal(I)) {
    auto *Object = fsea::MonitorEnter(I).getObject();
    if (auto ObjectTP = S.getTrackedPointer(Object))
      if (S.getAllocation(ObjectTP)->ExactState)
        if (S.getAllocation(ObjectTP)->ExactState->monitorEnter())
          return true;
  }

  if (fsea::isMonitorExit(I) || fsea::isMonitorExitThreadLocal(I)) {
    auto *Object = fsea::MonitorExit(I).getObject();
    if (auto ObjectTP = S.getTrackedPointer(Object))
      if (S.getAllocation(ObjectTP)->ExactState)
        if (S.getAllocation(ObjectTP)->ExactState->monitorExit())
          return true;
  }

  if (ModelCAS && fsea::isCompareAndSwapObject(I)) {
    fsea::CompareAndSwapObject CAS(I);
    if (auto ObjectTP = S.getTrackedPointer(CAS.getObject())) {
      if (auto *OffsetValue = dyn_cast<ConstantInt>(CAS.getOffset())) {
        uint64_t Offset = OffsetValue->getZExtValue();
        if (auto FieldValue = S.getFieldValue(CAS.getExpectedValue()->getType(),
                                            *ObjectTP, Offset)) {
          assert(ObjectTP->Offset.value() == 0);
          if (auto FieldValueTP = S.getTrackedPointer(*FieldValue)) {
            // The field happens to refer to a tracked allocation.
            LLVM_DEBUG(dbgs() << "Escaping allocation through CAS: ";
                       I.dump(););
            S.escape(FieldValueTP->AllocID);
            if (!S.getAllocation(ObjectTP->AllocID))
              return true;
          }
          if (auto NewValueTP = S.getTrackedPointer(CAS.getNewValue())) {
            S.escape(NewValueTP->AllocID);
            if (!S.getAllocation(ObjectTP->AllocID))
              return true;
          }
          CASStoredValue *VCAS;
          if (auto *VV = VContext.getVirtualValue(I)) {
            VCAS = VV->asCASStoredValue();
            assert(VCAS);
          } else {
            VCAS = new CASStoredValue(CAS);
            VContext.setInstructionModel(
                I, std::unique_ptr<InstructionModel>(
                        new SingleValueInstructionModel(VCAS)));
          }

          if (FixedPointReached)
            assert(VCAS->CurrentFieldValue == *FieldValue);
          else
            VCAS->setCurrentFieldValue(*FieldValue);
          TrackedPointer TP(ObjectTP->AllocID, Offset);
          return applyStorePointerUse(I, VCAS, TP);
        }
      }
    }
  }

  return visitInstruction(I);
}

bool ExactStateInstVisitor::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  if (!ModelCAS)
    return visitInstruction(I);

  bool ChangeMade = false;
  if (auto NewValTP = S.getTrackedPointer(I.getNewValOperand()))
    ChangeMade |= S.escape(NewValTP->AllocID);
  if (auto PtrTP = S.getTrackedPointer(I.getPointerOperand()))
    ChangeMade |= applyAtomicCmpXchg(I, *PtrTP);
  return ChangeMade;
}

bool ExactStateInstVisitor::visitAtomicRMWInst(AtomicRMWInst &I) {
  if (!ModelCAS)
    return visitInstruction(I);
  if (auto PtrTP = S.getTrackedPointer(I.getPointerOperand()))
    return applyAtomicRMW(I, *PtrTP);
  return false;
}

bool ExactStateInstVisitor::visitStoreInst(StoreInst &SI) {
  auto *Ptr = SI.getPointerOperand();
  auto *Val = SI.getValueOperand();

  bool ChangeMade = false;
  if (auto ValTP = S.getTrackedPointer(Val))
    ChangeMade |= applyStoreValueUse(SI, *ValTP);
  if (auto PtrTP = S.getTrackedPointer(Ptr))
    ChangeMade |= applyStorePointerUse(SI, Val, *PtrTP);
  return ChangeMade;
}

bool ExactStateInstVisitor::visitBitCastInst(BitCastInst &I) {
  return applyAlias(I, I.getOperand(0));
}

bool ExactStateInstVisitor::visitAddrSpaceCastInst(AddrSpaceCastInst &I) {
  return applyAlias(I, I.getOperand(0));
}

bool ExactStateInstVisitor::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  auto TP = S.getTrackedPointer(GEP.getPointerOperand());
  if (!TP)
    return false;

  if (!TP->Offset) {
    S.addTrackedPointer(&GEP, *TP);
    return true;
  }

  std::optional<int64_t> Off;
  const DataLayout &DL = GEP.getModule()->getDataLayout();
  unsigned BitWidth = DL.getIndexSizeInBits(GEP.getPointerAddressSpace());
  APInt Offset(BitWidth, 0);
  if (GEP.accumulateConstantOffset(DL, Offset))
    Off = *TP->Offset + Offset.getSExtValue();
  S.addTrackedPointer(&GEP, TrackedPointer(TP->AllocID, Off));
  return true;
}

bool ExactStateInstVisitor::visitInstruction(Instruction &I) {
  assert(!isa<PHINode>(&I) && "PHIs are handled while merging states");

  bool Changed = false;

  for (unsigned OpI = 0, OpE = I.getNumOperands(); OpI < OpE; ++OpI)
    if (auto TP = S.getTrackedPointer(I.getOperand(OpI)))
      if (applyOperandUse(I.getOperandUse(OpI), *TP))
        Changed = true;

  return Changed;
}

bool SymbolicStateInstVisitor::applyInitializingInstruction(Instruction &I,
                                                            Value *OpV) {
  auto PtrTP = S.getTrackedPointer(OpV);
  if (!PtrTP)
    return false;
  auto *Allocation = S.getAllocation(PtrTP);
  auto &SymbolicState = Allocation->SymbolicState;
  if (!SymbolicState)
    return false;
  auto II =
      SymbolicAllocationState::InitializingInstruction::createIRInstruction(
          &I, Allocation->NewInstruction);
  SymbolicState->InitializingInstructions.push_back(II);
  return true;
}

bool SymbolicStateInstVisitor::applyPublicationBarrier(
    const CallBase *PublicationBarrierCall) {
  fsea::FinalPublicationBarrier Barrier(*PublicationBarrierCall);
  auto PtrTP = S.getTrackedPointer(Barrier.getValueArg());
  if (!PtrTP)
    return false;
  auto *Allocation = S.getAllocation(PtrTP);
  auto &SymbolicState = Allocation->SymbolicState;
  if (!SymbolicState)
    return false;
  auto II = SymbolicAllocationState::InitializingInstruction::
      createPublicationBarrier();
  SymbolicState->InitializingInstructions.push_back(II);
  return true;
}

std::optional<int64_t>
SymbolicStateMemcpy::estimateInitializingInstructionLowerBound(
    const SymbolicAllocationState::InitializingInstruction &II,
    LazyValueInfo &LVI, Instruction *CtxI) {
  assert(shouldForwardInitializingInstruction(II) &&
         "Must be checked by the caller!");
  assert(II.isIRInstruction() &&
         "We do not forward any other initializing instructions");

  auto *I = cast<Instruction>(II.getIRInstruction());
  auto &DL = I->getModule()->getDataLayout();
  if (auto *SI = dyn_cast<StoreInst>(I))
    return fsea::FlowSensitiveEAUtils::estimatePointerLowerBoundOffset(
        SI->getPointerOperand(), II.getIRInstructionBaseObject(), DL, CtxI,
        &LVI);

  if (auto *AMI = dyn_cast<AtomicMemCpyInst>(I))
    return fsea::FlowSensitiveEAUtils::estimatePointerLowerBoundOffset(
        AMI->getRawDest(), II.getIRInstructionBaseObject(), DL, CtxI,
        &LVI);

  return std::nullopt;
}

/// Checks whether it is legal to forward the given initializing instruction.
/// Doesn't check that the initializing instruction is within the range being
/// copied.
bool SymbolicStateMemcpy::canForwardInitializingInstruction(
    const SymbolicAllocationState::InitializingInstruction &II) {
  assert(shouldForwardInitializingInstruction(II) &&
         "Must be checked by the caller!");
  assert(II.isIRInstruction() &&
         "We do not forward any other initializing instructions");

  const Instruction *I = II.getIRInstruction();

  if (dyn_cast<StoreInst>(I))
    return true;

  if (auto *InitializingAMI = dyn_cast<AtomicMemCpyInst>(I)) {
    // Is it a copy within the same allocation? If so, don't forward!
    //
    // This is the situation we want to avoid.
    //   a = new array[2];
    //   a[0] = 1;
    //   memcpy(a[1], a[0], 1);
    //   b = new array[2];
    //   ; State:
    //   ; alloc a, len=2
    //   ;  a[0] = 1;
    //   ;  memcpy(a[1], a[0], 1);
    //   ; alloc b, len=2
    //   memcpy(b[0], a[0], 2);
    //
    // We forward through memcpy so as to figure out a way to initialize the
    // destination allocation without referencing the intermediate allocation.
    // In this case forwarding memcpy(a[1], a[0], 1) is not beneficial. Even
    // after forwarding we would have a reference to the intermediate array.
    //   ; State:
    //   ; alloc a, len=2
    //   ;  a[0] = 1;
    //   ;  memcpy(a[1], a[0], 1);
    //   ; alloc b, len=2
    //   ;  b[0] = 1;
    //   ;  memcpy(b[1], a[0], 1);
    //
    // TODO: we can forward such memcpy instructions if we rewrite the source
    // argument of the memcpy as well.
    auto InitializingSrcTP = S.getTrackedPointer(InitializingAMI->getRawSource());
    if (InitializingSrcTP && SrcTP.AllocID == InitializingSrcTP->AllocID)
      return false;

    return true;
  }

  return false;
}

bool SymbolicStateMemcpy::isWithinMemcpyRange(
    const SymbolicAllocationState::InitializingInstruction &II,
    LazyValueInfo &LVI) {
  if (!isWithinLowerBound(II, LVI)) {
    LLVM_DEBUG(dbgs() << "Initializing instruction is not within lower bound "
                      << II << "\n");
    return false;
  }

  if (!isWithinUpperBound(II)) {
    LLVM_DEBUG(dbgs() << "Initializing instruction is not within upper bound "
                      << II << "\n");
    return false;
  }

  return true;
}

/// Checks that the initializing instruction is above the lower bound of this
/// memcpy.
bool SymbolicStateMemcpy::isWithinLowerBound(
    const SymbolicAllocationState::InitializingInstruction &II,
    LazyValueInfo &LVI) {
  assert(shouldForwardInitializingInstruction(II) &&
         "Must be checked by the caller!");
  assert(canForwardThrough() && "Should not be called otherwise!");
  assert(DestTP.Offset && "Should be checked by canForwardThrough!");

  // Estimate the lower bound for the initializing instruction offset and
  // compare it with the constant memcpy start offset.
  std::optional<int64_t> LowerBoundOffset =
        estimateInitializingInstructionLowerBound(II, LVI, AMI);
  if (LowerBoundOffset)
    return *LowerBoundOffset >= *DestTP.Offset;

  return false;
}

/// Checks that the initializing instruction is below the upper bound of this
/// memcpy.
bool SymbolicStateMemcpy::isWithinUpperBound(
    const SymbolicAllocationState::InitializingInstruction &II) {
  assert(shouldForwardInitializingInstruction(II) &&
         "Must be checked by the caller!");
  assert(canForwardThrough() && "Should not be called otherwise!");
  assert(DestTP.Offset && "Should be checked by canForwardThrough!");

  assert(II.isIRInstruction() &&
         "We do not forward any other initializing instructions");
  const Instruction *I = II.getIRInstruction();
  const Value *Base = II.getIRInstructionBaseObject();

  // We can prove that the instruction is below the upper bound is two cases:
  //
  // 1) We know that we are copying full length of the source array. In this
  // case all initializing instruction are below the upper bound. Note that full
  // length copy means that we are copying all *elements* of the source array.
  // Header bytes are not included in full length copy. So, we still need to
  // check for the lower bound even if it is a full length copy.
  if (FullSrcCopy)
    return true;

  // 2) We know the constant length of the memcpy and the constant offset of the
  // initializing instruction.
  auto *MemcpyLengthCI = dyn_cast<ConstantInt>(AMI->getLength());
  if (!MemcpyLengthCI) {
    LLVM_DEBUG(dbgs() << "Memcpy length is not constant " << *AMI << "\n");
    return false;
  }
  int64_t MemcpyTo = *DestTP.Offset + MemcpyLengthCI->getSExtValue();

  auto &DL = I->getModule()->getDataLayout();

  if (auto *SI = dyn_cast<StoreInst>(I)) {
    // Check that the store is within the range being copied.
    int64_t Offset;
    if (Base !=
        GetPointerBaseWithConstantOffset(SI->getPointerOperand(), Offset, DL))
      return false;
    auto *Ty = SI->getValueOperand()->getType();
    if (!Ty->isSized())
      return false;
    int64_t StoreSize = DL.getTypeStoreSize(Ty).getFixedValue();
    return Offset + StoreSize <= MemcpyTo;
  }

  if (auto *AMI = dyn_cast<AtomicMemCpyInst>(I)) {
    auto *Dest = AMI->getRawDest();
    // Check that the memcpy is within the range being copied.
    int64_t Offset;
    if (Base != GetPointerBaseWithConstantOffset(Dest, Offset, DL))
      return false;
    auto *Length = dyn_cast<ConstantInt>(AMI->getLength());
    if (!Length)
      return false;
    int64_t MemcpySize = Length->getSExtValue();
    return Offset + MemcpySize <= MemcpyTo;
  }

  return false;
}

/// Returns the constant header size for the array.
/// Returns std::nullopt if:
/// - ArrayAlloc->NewInstruction is not an array, or
/// - header size is unknown, or
/// - header is is not a constant.
/// We assume that memory above the header is zero initialized for newly
/// allocated arrays.
/// TODO: we have several functions which answer this kind of question:
/// - fsea::Utils::isAccessToZeroInitializedLocation
/// - fsea::TypeUtils::isArrayElementAccess
/// - fsea::TypeUtils::isObjectFieldAccess
/// We should unify these.
std::optional<unsigned> NewArrayDesc::getArrayHeaderSize() {
  // Do we know JavaTypeInfo for the constant KlassID?
  if (auto MaybeKID =
          TypeUtils::runTimeToCompileTimeKlassID(ArrayAlloc->KlassID)) {
    LLVMContext &C = ArrayAlloc->NewInstruction->getContext();
    fsea::TypeUtils::JavaType T(*MaybeKID, /*IsExact=*/true);
    if (auto JTI = fsea::VMInterface::getJavaTypeInfo(C, T, std::nullopt))
      return JTI->isArray() ? std::optional<unsigned>(JTI->getArrayHeaderSize())
                            : std::nullopt;
  }
  // Is it a new array allocation?
  // In this case query the info from the allocation abstraction.
  if (!fsea::isNewArray(*ArrayAlloc->NewInstruction))
    return std::nullopt;
  fsea::NewArray NA(*ArrayAlloc->NewInstruction);
  auto *HeaderSizeCI = dyn_cast<ConstantInt>(NA.getHeaderSize());
  if (!HeaderSizeCI)
    return std::nullopt;
  return HeaderSizeCI->getZExtValue();
}

std::optional<unsigned> NewArrayDesc::getArrayElementShift() {
  // Do we know JavaTypeInfo for the constant KlassID?
  if (auto MaybeKID =
          TypeUtils::runTimeToCompileTimeKlassID(ArrayAlloc->KlassID)) {
    LLVMContext &C = ArrayAlloc->NewInstruction->getContext();
    fsea::TypeUtils::JavaType T(*MaybeKID, /*IsExact=*/true);
    if (auto JTI = fsea::VMInterface::getJavaTypeInfo(C, T, std::nullopt))
      return JTI->isArray()
                 ? std::optional<unsigned>(JTI->getArrayElementShift())
                 : std::nullopt;
  }
  // Is it a new array allocation?
  // In this case query the info from the allocation abstraction.
  if (!fsea::isNewArray(*ArrayAlloc->NewInstruction))
    return std::nullopt;
  fsea::NewArray NA(*ArrayAlloc->NewInstruction);
  auto *ElementShiftCI = dyn_cast<ConstantInt>(NA.getElementShift());
  if (!ElementShiftCI)
    return std::nullopt;
  return ElementShiftCI->getZExtValue();
}

/// Check if the given value matches with the array length.
bool NewArrayDesc::isArrayLength(Value *V) {
  // Because NewArrayDesc might be a PHI-merged allocation we need to match
  // the lengths across all paths.
  using ArrayLengthPair = std::pair<const Value *, Value *>;
  SmallVector<ArrayLengthPair, 16> Worklist;
  SmallSet<ArrayLengthPair, 16> Visited;

  Worklist.emplace_back(ArrayAlloc->NewInstruction, V);

  while (!Worklist.empty()) {
    auto AL = Worklist.pop_back_val();
    if (!Visited.insert(AL).second)
      continue;

    // If the array is a PHI we need match lengths across all incoming paths.
    if (auto *ArrayPHI = dyn_cast<PHINode>(AL.first)) {
      if (auto *LengthPHI = dyn_cast<PHINode>(AL.second)) {
        if (ArrayPHI->getParent() == LengthPHI->getParent()) {
          // If length is also a PHI in the same BB, look through the PHI and
          // match incoming values.
          for (auto *Pred : predecessors(ArrayPHI->getParent()))
            Worklist.emplace_back(ArrayPHI->getIncomingValueForBlock(Pred),
                                 LengthPHI->getIncomingValueForBlock(Pred));
          continue;
        }
      }

      // Ok, the length is not a PHI we can look through. Try to match the
      // same length value across all incoming arrays. For example:
      //
      //   if ()
      //     %a = new array[%len]
      //   else
      //     %b = new array[%len]
      //   %merge = phi(%a, %b)
      //
      // We can match %len as array length for %merge array.
      for (auto *Pred : predecessors(ArrayPHI->getParent()))
        Worklist.emplace_back(ArrayPHI->getIncomingValueForBlock(Pred),
                              AL.second);
      continue;
    }

    // If the array is a new array call, simply match the length argument.
    if (fsea::isNewArray(*AL.first)) {
      fsea::NewArray NA(*AL.first);
      auto *Length = NA.getLength();
      if (AL.second == Length)
        // Simple case, they are the same value
        continue;

      // We might have constants of different width, e.g. i32 5 and i64 5.
      // If both are constants, check if they have the same unsigned value.
      if (auto *CI = dyn_cast<ConstantInt>(AL.second))
        if (auto *LengthCI = dyn_cast<ConstantInt>(Length)) {
          const APInt &VAPInt = CI->getValue();
          const APInt &LengthAPInt = LengthCI->getValue();
          unsigned MaxWidth = std::max(VAPInt.getBitWidth(),
                                       LengthAPInt.getBitWidth());
          // Note: we should never trunc here because the width is the max. But
          // there is no convenient API in APInt to either zext of keep the width
          // the same.
          if (VAPInt.zextOrTrunc(MaxWidth) == LengthAPInt.zextOrTrunc(MaxWidth))
            continue;
        }

      return false;
    }

    return false;
  }

  return true;
}

/// For the given value which is length in bytes returns an existing value
/// which contains the corresponding length in elements. Strips away all ZExts
/// from the returned value.
///
/// Returns std::nullopt if the length in elements is not available in the IR.
std::optional<Value *>
NewArrayDesc::getLengthInElements(Value *LengthInBytes, uint64_t ElementShift) {
  using namespace PatternMatch;

  Value *LengthInElements = nullptr;
  if (ElementShift == 0)
    // Length in bytes is the same as length in elements
    if (match(LengthInBytes, m_ZExtOrSelf(m_Value(LengthInElements))))
      return LengthInElements;

  if (auto *LengthInBytesCI = dyn_cast<ConstantInt>(LengthInBytes)) {
    const APInt &Length = LengthInBytesCI->getValue();
    // Divide length in bytes by scale
    APInt LengthInElements = Length.ashr(ElementShift);
    // Check that the remainder of division is zero.
    // E.g. reject the case when length in bytes is 5 and element shift is 2.
    if (LengthInElements.shl(ElementShift) != Length)
      return std::nullopt;
    return ConstantInt::get(LengthInBytes->getType(), LengthInElements);
  }

  // Check for the pattern where length in bytes is computed as a
  // non-overflowing shift of length in elements:
  //   LengthInBytes = LengthInElements << ElementShift
  if (match(LengthInBytes,
            m_ZExtOrSelf(m_NUWShl(m_ZExtOrSelf(m_Value(LengthInElements)),
                                  m_SpecificInt(ElementShift)))))
    return LengthInElements;

  return std::nullopt;
}

bool NewArrayDesc::isMemcpyLengthEqualToArrayLength(Value *LengthInBytes) {
  LLVM_DEBUG(dbgs() << "isMemcpyLengthEqualToArrayLength " << *LengthInBytes << "\n");
  auto MaybeElementShift = getArrayElementShift();
  if (!MaybeElementShift) {
    LLVM_DEBUG(dbgs() << "\telement shift is unknown\n");
    return false;
  }

  auto MaybeLengthInElements =
      getLengthInElements(LengthInBytes, *MaybeElementShift);
  if (!MaybeLengthInElements) {
    LLVM_DEBUG(dbgs() << "\tlength is elements is unknown\n");
    return false;
  }

  if (!isArrayLength(*MaybeLengthInElements)) {
    LLVM_DEBUG(dbgs() << "\tgiven value is not array length\n");
    return false;
  }

  return true;
}

bool SymbolicStateMemcpy::copyPreservesOffsets() {
  // Memcpy dest should have a constant offset
  if (!DestTP.Offset) {
    LLVM_DEBUG(dbgs() << "Dest offset is not constant" << *AMI << "\n");
    return false;
  }

  // Dest and src should have the same offset
  if (DestTP.Offset != SrcTP.Offset) {
    LLVM_DEBUG(dbgs() << "Memcpy src and dest offsets are not the same " << *AMI
                      << "\n");
    return false;
  }

  return true;
}

/// Checks that the memcpy is suitable for forward through transformation.
/// It only check the properties of the memcpy itself without looking at the
/// symbolic state being copied. It is up to the caller to verify that every
/// initializing instruction from the symbolic state of the source allocation
/// can be forwarded.
bool SymbolicStateMemcpy::canForwardThrough() {
  if (!copyPreservesOffsets())
    return false;

  if (!SrcHeaderSize || !DestHeaderSize)
    return false;

  // Check that we are copying tracked zero-initialized memory.
  //
  // FlowSensitiveEA doesn't track header initialization which occurs inside of
  // the allocation abstraction. If we encounter a memcpy which copies the
  // header bits we can't forward the initializing instructions and assume that
  // it will produce the same allocation state. For example:
  //   a = new A ; Initialized header
  //   store 1, a.f
  //   b = new B ; Initialized header
  //   memcpy(a, b, sizeof(a)) ; A memcpy involving the header
  // In this case the tracked state is only initializing the field f, but the
  // memcpy touches the header as well.
  //
  // Even if the memory is tracked but not zero initialized, we may not be able
  // to forward through. For example:
  //   a = new A // Everything above <headersize> is zero initialized
  //   b = new B // B has non-zero initialized fields above <headersize>
  //   memcpy(a+<headersize>, b+<headersize>, sizeof(a) - <headersize>)
  // In this case forward through would forward empty initialization state.
  // But this is not correct, because memcpy copies initial non-zero values
  // into a.
  //
  // I don't expect to see this in practice. Check this here mostly for future
  // proofing.
  //
  // Check that both src and dest are arrays and corresponding offsets are above
  // the header. We assume that the memory above the header is zero initialized
  // for newly allocated arrays. This may not always be true for instances if
  // they have non-zero initialized paddings between zero-initialized fields.
  if (*SrcHeaderSize > *SrcTP.Offset || *DestHeaderSize > *DestTP.Offset)
    return false;

  return true;
}

/// Try to model a memcpy by forwarding the initializing instructions of the
/// source allocation.
///
/// This function looks for a realloc pattern - when all initialized elements
/// of an unescaped source array are copied into a newly allocated dest array:
///   src = new array[len]
///   ... init src ...
///   dest = new array
///   memcpy(dest, src, len)
///
/// In this case instead of modelling the memcpy in the dest state explicitly,
/// we can forward all the initializing instructions of the source array into
/// the dest array. As a result we can materialize the state of the dest array
/// without referencing the intermediate source array.
bool SymbolicStateInstVisitor::tryForwardInitializingInstructions(
    AtomicMemCpyInst *AMI, TrackedPointer DestTP) {
  auto *DestAllocation = S.getAllocation(DestTP);
  assert(DestAllocation->SymbolicState && "Should be checked by the caller!");

  // We can only forward into an empty allocation
  if (!DestAllocation->SymbolicState->isUnmodified())
    return false;

  // Src must be a tracked allocation with symbolic state
  auto SrcTP = S.getTrackedPointer(AMI->getRawSource());
  if (!SrcTP)
    return false;
  auto *SrcAllocation = S.getAllocation(SrcTP);
  if (!SrcAllocation->SymbolicState)
    return false;

  SymbolicStateMemcpy Memcpy(S, AMI, DestTP, *SrcTP, DestAllocation,
                             SrcAllocation);
  if (!Memcpy.canForwardThrough())
    return false;

  // Check that all elements of the src state can be forwarded through the
  // memcpy
  for (auto &II : SrcAllocation->SymbolicState->InitializingInstructions) {
    if (!SymbolicStateMemcpy::shouldForwardInitializingInstruction(II))
      continue;
    if (!Memcpy.canForwardInitializingInstruction(II)) {
      LLVM_DEBUG(dbgs() << "Can't forward initializing instruction "
                        << II << "\n");
      return false;
    }
    if (!Memcpy.isWithinMemcpyRange(II, LVI)) {
      LLVM_DEBUG(dbgs() << "Initializing memcpy is not within the range "
                        << II << "\n");
      return false;
    }
  }

  for (auto &II : SrcAllocation->SymbolicState->InitializingInstructions)
    if (SymbolicStateMemcpy::shouldForwardInitializingInstruction(II))
      DestAllocation->SymbolicState->InitializingInstructions.push_back(II);

  for (auto &MS : SrcAllocation->SymbolicState->MemorySources)
    DestAllocation->SymbolicState->MemorySources.push_back(MS);

  return true;
}

bool SymbolicStateInstVisitor::applyMemcpy(AtomicMemCpyInst *AMI) {
  auto DestTP = S.getTrackedPointer(AMI->getRawDest());
  if (!DestTP)
    return false;
  auto *DestAllocation = S.getAllocation(DestTP);
  auto &DestSymbolicState = DestAllocation->SymbolicState;
  if (!DestSymbolicState)
    return false;

  if (!tryForwardInitializingInstructions(AMI, *DestTP)) {
    auto II =
        SymbolicAllocationState::InitializingInstruction::createIRInstruction(
            AMI, DestAllocation->NewInstruction);
    DestSymbolicState->InitializingInstructions.push_back(II);
    auto *SrcV = AMI->getRawSource();
    DestSymbolicState->MemorySources.emplace_back(
        SrcV, S.getTrackedPointer(SrcV));
  }
  return true;
}

bool SymbolicStateInstVisitor::visitCallBase(CallBase &I) {
  if (fsea::isFinalPublicationBarrier(I))
    return applyPublicationBarrier(cast<CallBase>(&I));

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    if (II->getIntrinsicID() == Intrinsic::invariant_start)
      return applyInitializingInstruction(I, I.getArgOperand(1));
  }

  if (auto *AMI = dyn_cast<AtomicMemCpyInst>(&I))
    return applyMemcpy(AMI);

  return visitInstruction(I);
}

bool SymbolicStateInstVisitor::visitStoreInst(StoreInst &SI) {
  return applyInitializingInstruction(SI, SI.getPointerOperand());
}

bool SymbolicStateInstVisitor::visitInstruction(Instruction &I) {
  assert(!isa<PHINode>(&I) && "PHIs are handled while merging states");
  if (!I.mayWriteToMemory())
    return false;

  bool Changed = false;

  for (unsigned OpI = 0, OpE = I.getNumOperands(); OpI < OpE; ++OpI)
    if (auto TP = S.getTrackedPointer(I.getOperand(OpI)))
      if (auto &SymbolicState = S.getAllocation(TP)->SymbolicState) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Use &U = I.getOperandUse(OpI);

          // Check whether the call site can write through the tracked pointer
          // and modify the tracked content.
          if (CB->onlyReadsMemory(U.getOperandNo()))
            continue;

          if (CB->isBundleOperand(U.getOperandNo())) {
            // Deopt bundle use doesn't modify the content of the passed object.
            auto Bundle = CB->getOperandBundleForOperand(U.getOperandNo());
            if (Bundle.getTagID() == LLVMContext::OB_deopt)
              continue;
          }
        }

        SymbolicState = std::nullopt;
        Changed = true;
    }

  return Changed;
}

StateInstVisitor::StateInstVisitor(State &S, FlowSensitiveEscapeAnalysis &EA,
                                   bool FixedPointReached)
    : S(S), DeoptStateVisitor(S),
      ExactStateVisitor(S, EA.VContext, FixedPointReached),
      SymbolicStateVisitor(S, EA.BatchAA, EA.LVI) {}

#ifndef NDEBUG
static bool isPHI(ExtendedValue V) {
  if (auto OptV = V.asValue())
    return isa<PHINode>(*OptV);
  assert(V.asVirtualValue());
  return V.asVirtualValue().value()->asVirtualPHI();
}
#endif

/// For incoming blocks with known states
/// calculate set of incoming values that are inputs to the \p PHI possibly
/// transitively from the other phis of the same block. For example:
/// phiBlock:
///   %ph1 = phi [%in1, %inBlock1], [%ph2, %phiBlock]
///   %ph2 = phi [%in2, %inBlock1], [%ph2, %phiBlock]
/// getIncomingOuterClosure(%ph1) results in {%in1, %in2}.
/// getIncomingOuterClosure(%ph2) results in {%in2}.
///
/// If we encounter a PHI which is a PHI-merged allocation, don't follow its
/// incoming values, treat it like a regualar value instead.
SmallSet<ExtendedValue, 8> State::getMergedValues(ExtendedValue PHI,
                                                  const BasicBlock &BB,
                                                  GetBlockOutState GetState) {
  SmallVector<ExtendedValue, 16> Worklist;
  SmallSet<ExtendedValue, 16> Visited;
  SmallSet<ExtendedValue, 8> Result;

  assert(isPHI(PHI) && "Should be a PHI!");
  Worklist.push_back(PHI);

  auto IsPHIMergedAllocation = [&] (ExtendedValue Ptr) {
    auto PtrV = Ptr.asValue();
    if (!PtrV)
      // We don't (yet) merge allocations at virtual PHIs
      return false;
    assert(isa<PHINode>(*PtrV) && "Only makes sense for PHINodes");
    auto *Allocation = getAllocation(getTrackedPointer(Ptr));
    if (!Allocation)
      return false;
    if (!Allocation->isPHIMergedAllocation())
      return false;
    // Check that PtrV is actually a PHI-merged allocation and not a tracked
    // pointer to one.
    return Allocation->NewInstruction == *PtrV;
  };

  while (!Worklist.empty()) {
    auto In = Worklist.pop_back_val();
    if (!Visited.insert(In).second)
      continue;
    if (auto InV = In.asValue()) {
      if (auto *InPHI = dyn_cast<PHINode>(*InV))
        if (InPHI->getParent() == &BB &&
            !IsPHIMergedAllocation(InPHI)) {
          for (auto &InBB : InPHI->blocks())
            if (GetState(InBB).isKnownState())
              Worklist.push_back(InPHI->getIncomingValueForBlock(InBB));
          continue;
        }
    } else {
      auto InVV = In.asVirtualValue();
      assert(InVV);
      if (auto *InVPHI = InVV.value()->asVirtualPHI())
        if (InVPHI->getParent() == &BB &&
            !IsPHIMergedAllocation(InVPHI)) {
          for (unsigned i = 0; i < InVPHI->getNumIncomingValues(); i++)
            if (GetState(InVPHI->getIncomingBlock(i)).isKnownState())
              Worklist.push_back(InVPHI->getIncomingValue(i));
          continue;
        }
    }
    Result.insert(In);
  }

  return Result;
}

bool State::applyPhi(const ExtendedValue PHI, const BasicBlock &BB,
                     GetBlockOutState GetState) {
  auto MergedValues = getMergedValues(PHI, BB, GetState);

  SmallSet<std::optional<TrackedPointer>, 8> TPs;
  for (auto V : MergedValues)
    TPs.insert(getTrackedPointer(V));

  if (TPs.empty())
    return false;

  if (TPs.size() == 1 && *TPs.begin()) {
    addTrackedPointer(PHI, **TPs.begin());
    return true;
  }

  bool Changed = false;
  for (auto &TP : TPs)
    if (TP) {
      if (!Changed)
        LLVM_DEBUG(dbgs() << "Escaping for merged PHI inputs: "; PHI.dump(););
      Changed |= escape(TP->AllocID);
    }

  return Changed;
}

SetVector<AllocationID>
State::getAllocationClosure(SmallVectorImpl<ExtendedValue> &&Worklist) const {
  SetVector<AllocationID> Visited;
  while (!Worklist.empty()) {
    ExtendedValue V = Worklist.pop_back_val();
    if (auto TP = getTrackedPointer(V))
      if (Visited.insert(TP->AllocID)) {
        const auto *A = getAllocation(TP);
        assert(A && "Must have a valid allocation");
        if (A->ExactState)
          for (const auto &FV : A->ExactState->FieldValues)
            Worklist.emplace_back(FV.second);
      }
  }

  return Visited;
}

SetVector<AllocationID>
State::getAllocationContentClosure(const Allocation *A) const {
  if (!A->ExactState)
    // Can't hold tracked pointers if exact state is unknown
    return SetVector<AllocationID>();
  SmallVector<ExtendedValue, 16> Worklist;
  for (auto &FV : A->ExactState->FieldValues)
    Worklist.emplace_back(FV.second);
  return getAllocationClosure(std::move(Worklist));
}

FlowSensitiveEscapeAnalysis &FlowSensitiveEAUpdater::getFlowSensitiveEA() {
  return FSEA;
}

bool FlowSensitiveEAUpdater::anyIRChangeMade() const {
  return InvalidateAll;
}

void FlowSensitiveEAUpdater::invalidate() {
  InvalidateAll = true;
}

void FlowSensitiveEAUpdater::invalidateBlock(const BasicBlock *BB) {
  (void)BB;
  // For now we just invalidate the whole analysis on block updates.
  invalidate();
}

void FlowSensitiveEAUpdater::applyUpdates() {
  if (InvalidateAll) {
    FSEA.clear(/* ClearBlockPRON = */true);
    FSEA.calculateBBStates();
  }
  InvalidateAll = false;
}
} // namespace FlowSensitiveEA

namespace FlowSensitiveEAUtils {
// Shared implementation for isPointerDeadThroughInstruction and
// isPointerDeadThroughBlockEntry.
using PointerDeadContext = PointerUnion<const BasicBlock*, const Instruction*>;
bool isPointerDeadThrough(const Instruction *Ptr, PointerDeadContext Ctx,
                          function_ref<bool(User *)> SkipUser) {
  assert((Ctx.is<const BasicBlock *>() ||
          !isa<PHINode>(Ctx.get<const Instruction *>())) &&
          "Use block entry instead of PHINode context!");

  const BasicBlock *CtxBB = Ctx.is<const BasicBlock *>() ?
    Ctx.get<const BasicBlock *>() :
    Ctx.get<const Instruction *>()->getParent();

  SmallVector<const Instruction *, 16> Worklist;
  SmallSet<const Instruction *, 16> Visited;

  // Collect all users of the pointer and track the paths from the users back to
  // the Ptr def. If we encounter context on the way it means that Ptr is live
  // through the context.
  for (const Use &U : Ptr->uses()) {
    auto *I = cast<Instruction>(U.getUser());
    if (SkipUser && SkipUser(I))
      continue;
    if (auto *PHI = dyn_cast<PHINode>(I)) {
      // If the context is a block entry any PHI use in the context block makes
      // the pointer live.
      if (Ctx.is<const BasicBlock *>() && PHI->getParent() == CtxBB)
        return false;
      // Otherwise analyze the predecessor corresponding to the current use.
      Worklist.push_back(PHI->getIncomingBlock(U)->getTerminator());
      continue;
    }
    Worklist.push_back(I);
  }

  while (!Worklist.empty()) {
    const Instruction *I = Worklist.pop_back_val();
    if (!Visited.insert(I).second)
      continue;
    auto *BB = I->getParent();

    if (BB != Ptr->getParent() && BB != CtxBB)
      // If this BB is neither Ptr def block nor CtxBB block, simply skip it
      // and analyze the predecessors.
      for (auto *Pred : predecessors(BB))
        Worklist.push_back(Pred->getTerminator());

    if (BB == CtxBB) {
      // If this is a CtxBB, we need to scan instructions one by one and stop
      // once we reach either the Ptr def or the context.

      auto *PrevI = I->getPrevNode();
      if (Ctx.is<const BasicBlock *>() && !PrevI)
        // Reached block entry context, pointer is not dead.
        return false;
      if (Ctx.is<const Instruction *>() && I == Ctx.get<const Instruction *>())
        // Reached instruction context, pointer is not dead.
        return false;

      if (I == Ptr)
        // Reached the Ptr def, stop the scan.
        continue;

      // Otherwise, scan backwards through the instructions.
      if (PrevI)
        Worklist.push_back(PrevI);
    }
  }

  return true;
}

bool isPointerDeadThroughInstruction(const Instruction *Ptr,
                                     const Instruction *CtxI,
                                     function_ref<bool(User *)> SkipUser) {
  return isPointerDeadThrough(Ptr, CtxI, SkipUser);
}

bool isPointerDeadThroughBlockEntry(const Instruction *Ptr,
                                    const BasicBlock *BB,
                                    function_ref<bool(User *)> SkipUser) {
  return isPointerDeadThrough(Ptr, BB, SkipUser);
}

std::optional<int64_t> estimatePointerLowerBoundOffset(
    const Value *Ptr, const Value *Base, const DataLayout &DL,
    Instruction *CtxI, LazyValueInfo *LVI) {
  // Compute the lower bound of the pointer offset using ExternalAnalysis.
  // ExternalAnalysis is queried for all non-constant offsets and provides
  // the lower bound for the value. Use ValueTracking and LVI to compute the
  // constant range for the value.
  APInt LowerBoundOffset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
  auto *PtrBase = Ptr->stripAndAccumulateConstantOffsets(
      DL, LowerBoundOffset, /*AllowNonInbounds=*/true,
      /* AllowInvariant */ false,
      [&](Value &V, APInt &ROffset) -> bool {
        // Use ValueTracking computeKnownBits and computeConstantRange to
        // compute the range. These will handle cases like this:
        //   %off = zext i32 %off.i32 to i64
        //   %off.16 = add nuw nsw i64 %off, 16
        auto Range =
            ConstantRange::fromKnownBits(computeKnownBits(&V, DL), true);
        Range = Range.intersectWith(
            computeConstantRange(&V, /*ForSigned*/false, /*UseInstrInfo=*/true,
                                 /*AssumptionCache=*/nullptr, CtxI));
        if (LVI && CtxI)
          // Use LazyValueInfo to refine the range. This will take the
          // dominating checks into account.
          // Note that LVI also does some value tracking style analysis, so
          // there is some overlap with computeKnownBits. It weaker than
          // computeKnownBits around loops, but this is not the main reason
          // to use LVI. The main reason is to get the information from the
          // dominating checks.
          Range = Range.intersectWith(LVI->getConstantRange(&V, CtxI));

        if (Range.isFullSet() || Range.isEmptySet())
          return false;
        ROffset = Range.getSignedMin();
        return true;
      });
  if (Base != PtrBase)
    // Couldn't strip/estimate offsets all the way to Base
    return std::nullopt;

  return LowerBoundOffset.getSExtValue();
}
} // namespace FlowSensitiveEAUtils
} // namespace fsea

AnalysisKey FlowSensitiveEA::Key;

FlowSensitiveEA::Result
FlowSensitiveEA::run(Function &F, FunctionAnalysisManager &FAM) {
  using namespace fsea::FlowSensitiveEA;
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &AA = FAM.getResult<AAManager>(F);
  auto &LVI = FAM.getResult<LazyValueAnalysis>(F);
  // Note: if you add any analysis dependencies here, please update
  // FlowSensitiveEscapeAnalysis::invalidate to reflect it.
  auto EA = std::make_unique<FlowSensitiveEscapeAnalysis>(F, DT, AA, LVI);
  auto EAUpdater = std::make_unique<FlowSensitiveEAUpdater>(*EA);
  return Result(std::move(EA), std::move(EAUpdater));
}
} // end of namespace llvm

/// Return true if the ptrtoint cast is used specifically for an assumption
/// about the alignment of the pointer.
static bool isOnlyUsedByAlignmentAssume(PtrToIntInst *PTI) {
  using namespace llvm::PatternMatch;
  if (!PTI->hasOneUse())
    return false;

  // an and mark off low bits
  Instruction *I = PTI->user_back();
  if (I->getOpcode() != Instruction::And ||
      !I->hasOneUse())
    return false;

  auto *CI = dyn_cast<ConstantInt>(I->getOperand(1));
  if (!CI)
    return false;
  auto AP = CI->getValue();
  if (AP.countTrailingOnes() != AP.popcount())
    return false;

  // a comparison of those low bits against zero
  auto *ICmp = dyn_cast<ICmpInst>(I->user_back());
  if (!ICmp || ICmp->getPredicate() != ICmpInst::ICMP_EQ ||
      !ICmp->hasOneUse() ||
      !match(ICmp->getOperand(1), m_Zero()))
    return false;

  // An assume that icmp is true
  I = ICmp->user_back();
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      if (Intrinsic::assume == F->getIntrinsicID())
        return true;
  return false;
}

fsea::UseEscapeKind fsea::getUseEscapeKind(const Use *U) {
  Instruction *I = cast<Instruction>(U->getUser());
  Value *V = U->get();

  // This is similar to CaptureTracking implementation, but has some subtle
  // difference for the cases which can capture the pointer but doesn't make
  // the underlying object escape. For example, we handle a ICMP differently.
  // A comparison can capture the value of the pointer but can't make it escape
  // in Java.
  //
  // We don't care about LLVM volatile memory accesses here, as we never
  // generate them for Java code. Java volatile semantic is expressed as
  // atomic operations with proper ordering constraints.

  switch (I->getOpcode()) {
  case Instruction::Call:
  case Instruction::Invoke: {
    auto *Call = cast<CallBase>(I);

    assert(!(isa<MemIntrinsic>(Call) && cast<MemIntrinsic>(Call)->isVolatile())
           && "Don't expect LLVM volatile accesses");

    // Monitor enters and exits do not cause the pointer to be captured.
    if (isMonitorEnter(*I) || isMonitorExit(*I) ||
        isMonitorEnterThreadLocal(*I) || isMonitorExitThreadLocal(*I))
      return UseEscapeKind::NoEscape;

    // Doesn't escape if the callee is readonly and doesn't return a copy
    // through its return value.
    if (Call->onlyReadsMemory() && Call->getType()->isVoidTy())
      return UseEscapeKind::NoEscape;

    // The pointer doesn't escape if returned pointer is not captured.
    if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(Call, true))
      return UseEscapeKind::Alias;

    // // The pointer doesn't escape if returned pointer is not captured.
    // if (fsea::Utils::OptimizeForCompressedPointers() &&
    //     isCallReturningPointerAliasingArgumentWithoutCapturing(Call))
    //   return UseEscapeKind::Alias;

    // Doesn't escape if only passed via 'nocapture' argument.
    return Call->doesNotCapture(U->getOperandNo()) ? UseEscapeKind::NoEscape
                                                   : UseEscapeKind::Escape;
  }
  case Instruction::Load:
    assert(!cast<LoadInst>(I)->isVolatile() &&
           "Don't expect LLVM volatile accesses");

    // Loading from a pointer does not make it escape.
    return UseEscapeKind::NoEscape;

  case Instruction::VAArg:
    // "va-arg" from a pointer does not not make it escape.
    return UseEscapeKind::NoEscape;

  case Instruction::Store:
    assert(!cast<StoreInst>(I)->isVolatile() &&
           "Don't expect LLVM volatile accesses");

    if (V == I->getOperand(0))
      // Stored the pointer - conservatively assume it may escape.
      return UseEscapeKind::Escape;
    // Storing to the pointee does not cause the pointer to escape.
    return UseEscapeKind::NoEscape;
  case Instruction::AtomicRMW: {
    // atomicrmw conceptually includes both a load and store from
    // the same location.
    // As with a store, the location being accessed doesn't escape,
    // but the value being stored does.
    auto *ARMWI = cast<AtomicRMWInst>(I);
    assert(!ARMWI->isVolatile() && "Don't expect LLVM volatile accesses");
    if (ARMWI->getValOperand() == V)
      return UseEscapeKind::Escape;
    return UseEscapeKind::NoEscape;
  }
  case Instruction::AtomicCmpXchg: {
    // cmpxchg conceptually includes both a load and store from
    // the same location.
    // As with a store, the location being accessed doesn't escape,
    // but the value being stored does.
    auto *ACXI = cast<AtomicCmpXchgInst>(I);
    assert(!ACXI->isVolatile() && "Don't expect LLVM volatile accesses");
    if (ACXI->getCompareOperand() == V || ACXI->getNewValOperand() == V)
      return UseEscapeKind::Escape;
    return UseEscapeKind::NoEscape;
  }
  case Instruction::BitCast:
  case Instruction::GetElementPtr:
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::AddrSpaceCast:
    // The original value doesn't escape via this if the new value doesn't.
    return UseEscapeKind::Alias;
  case Instruction::ICmp:
    // Comparing a reference does not make the pointer escape.
    return UseEscapeKind::NoEscape;
  case Instruction::PtrToInt:
    // We trying to match the form of an alignment assume here to avoid
    // spuriously failing on examples where the optimizer uses an assume
    // instead of an attribute. Ideally, we shouldn't see these at all, but
    // from a practical perspective it's better to tolerate them for the
    // moment.  When we introduce a strong gc pointer type to the IR, this
    // will become dead code.
    if (isOnlyUsedByAlignmentAssume(cast<PtrToIntInst>(I))) {
      LLVM_DEBUG(dbgs() << "Skipping alignment assume use\n");
      return UseEscapeKind::NoEscape;
    }
    return UseEscapeKind::Escape;
  default:
    return UseEscapeKind::Escape;
  }

  llvm_unreachable("Should return the result from the switch");
  return UseEscapeKind::Escape;
}

std::optional<fsea::JavaTypeInfo>
fsea::VMInterface::getJavaTypeInfo(llvm::LLVMContext &C, const TypeUtils::JavaType &T, std::optional<uint64_t> ArrayLen) {
  // TODO:
  return std::nullopt;
}

std::optional<fsea::FieldInfo>
fsea::VMInterface::getFieldInfoAtOffset(llvm::LLVMContext &C, TypeUtils::JavaType T, bool IsNew, int64_t Offset) {
  // TODO:
  return std::nullopt;
}

std::optional<uint64_t> fsea::VMInterface::getVMIntegerConstant(llvm::LLVMContext &C,
                                                                llvm::StringRef ConstantName) {
  // TODO:
  return std::nullopt;
}

std::optional<fsea::TypeUtils::CompileTimeKlassID>
fsea::VMInterface::runTimeToCompileTimeKlassID(llvm::LLVMContext &C, uint64_t RTKID) {
  // TODO:
  return std::nullopt;
}

std::optional<fsea::TypeUtils::CompileTimeKlassID>
fsea::TypeUtils::runTimeToCompileTimeKlassID(const Value *V) {
  const ConstantInt *CI = dyn_cast<ConstantInt>(V);
  if (!CI)
    return std::nullopt;
  return fsea::VMInterface::runTimeToCompileTimeKlassID(V->getContext(), CI->getZExtValue());
}
