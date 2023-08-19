//===- llvm/Analysis/FlowSensitiveEA.h --------------------------*- C++ -*-===//
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

#ifndef FLOWSENSITIVEEA_H
#define FLOWSENSITIVEEA_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/FlowSensitiveExtendedIR.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/FlowSensitiveAbstractions.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace llvm {
class AAResults;
class DominatorTree;
class LazyValueInfo;

namespace fsea {
namespace TypeUtils {

class JavaType;

/// Represents a compile-time type (e.g. class or interface) ID.
/// These IDs are used to represent types during compilation.
/// These IDs might be different from the IDs used to represent the
/// same types in run-time, e.g. the type IDs embedded in the IR for
/// type checks in run-time.
class CompileTimeKlassID {
private:
  uint64_t ID;

public:
  CompileTimeKlassID() : ID(0) {}
  explicit CompileTimeKlassID(uint64_t ID) : ID(ID) {}

  uint64_t getID() const { return ID; }

  bool operator==(const CompileTimeKlassID &other) const {
    return (ID == other.ID);
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const CompileTimeKlassID &T) {
  OS << T.getID();
  return OS;
}

class JavaType {
public:
  // Represents an upper bound on the possible types a value might hold.
  // Note that KlassID can represent either a class or an interface.
  CompileTimeKlassID KlassID;
  // If IsExact is set, value can only hold references to objects of exactly
  // the statically determined type and not a subtype thereof.
  bool IsExact;
  JavaType() {}
  JavaType(CompileTimeKlassID KlassID, bool IsExact)
      : KlassID(KlassID), IsExact(IsExact) {}

  void dump() const;
  void print(llvm::raw_ostream &OS) const;

  bool operator==(const JavaType &other) const {
    return (KlassID == other.KlassID) && (IsExact == other.IsExact);
  }

  bool operator!=(const JavaType &other) const { return !(*this == other); }
};

// Given a CompileTimeKlassID returns a value that represents
// the same type in run time. Note that the run time value might
// be different from the compile time value. All type IDs embedded
// in compiled code must be produced by this function.
std::optional<llvm::Value *>
compileTimeToRunTimeKlassID(llvm::Type *Ty, CompileTimeKlassID KID);

// Given a Value that represents a run time KlassID returns the corresponding
// CompileTimeKlassID that represents the same type. This function must be used
// when extracting type IDs from the IR.
std::optional<CompileTimeKlassID>
runTimeToCompileTimeKlassID(const llvm::Value *);
} // namespace TypeUtils

/// Information pertaining to a single field of a source level object.
struct FieldInfo { // TODO: Cachable

  enum FieldType {
    FIELD_TYPE_UNKNOWN = 0,
    FIELD_TYPE_BOOLEAN,
    FIELD_TYPE_BYTE,
    FIELD_TYPE_SHORT,
    FIELD_TYPE_INT,
    FIELD_TYPE_LONG,
    FIELD_TYPE_FLOAT,
    FIELD_TYPE_DOUBLE,
    FIELD_TYPE_CHAR,
    FIELD_TYPE_REFERENCE,
  };

  std::string Name;
  int32_t OffsetInBytes;
  uint32_t SizeInBytes;
  uint32_t AlignmentInBytes;

  // Reference type field properties
  // TODO: dereferenceability can be derived from the type:
  //   getJavaTypeInfo(KlassID)->getObjectSize()
  uint32_t DereferenceableInBytes;
  fsea::TypeUtils::CompileTimeKlassID KlassID;

  llvm::APInt KnownZero;
  llvm::APInt KnownOne;

  uint32_t Flags;
  uint32_t Type;

  FieldInfo() {}

  void verify() {
    assert(KnownZero.getBitWidth() == SizeInBytes * 8 &&
           "Sizes needs to be in sync!");
    assert(KnownOne.getBitWidth() == SizeInBytes * 8 &&
           "Sizes needs to be in sync!");
    assert((isReferenceField() || DereferenceableInBytes == 0) &&
           "DereferenceableInBytes must be set for reference fields only");
    assert((isReferenceField() ||
            KlassID == fsea::TypeUtils::CompileTimeKlassID(0)) &&
           "KlassID must be set for reference fields only");
    assert((isReferenceField() || ((Flags & REFERENCE_ONLY_FLAGS) == 0)) &&
           "reference field flags must be used only with reference fields");
    assert(((Flags & FLAG_KNOWN_TYPE) != 0 ||
            (Flags & FLAG_KNOWN_EXACT_TYPE) == 0) &&
           "FLAG_KNOWN_TYPE must be set if FLAG_KNOWN_EXACT_TYPE is set");
  }

public:
  enum : uint32_t {
    FLAG_NONE = 0,

    /// Loads from this field don't alias with any other stores in the
    /// runtime.  Loads of this field can be marked with !invariant.load
    FLAG_INVARIANT_FIELD = 2,

    /// The type of this reference field is known and stored in KlassID
    FLAG_KNOWN_TYPE = 4,

    /// This reference field is known to never be null
    FLAG_KNOWN_NON_NULL = 8,

    /// The type of this reference field is known to be exact
    FLAG_KNOWN_EXACT_TYPE = 16,

    // These flags are only make sense for reference type fields
    REFERENCE_ONLY_FLAGS =
        FLAG_KNOWN_TYPE | FLAG_KNOWN_NON_NULL | FLAG_KNOWN_EXACT_TYPE
  };

  explicit FieldInfo(int32_t O, uint32_t S)
      : OffsetInBytes(O), SizeInBytes(S), AlignmentInBytes(1),
        DereferenceableInBytes(0), KlassID(0),
        KnownZero(llvm::APInt(S * 8, 0)), KnownOne(llvm::APInt(S * 8, 0)),
        Flags(FLAG_NONE), Type(FIELD_TYPE_UNKNOWN) {
    verify();
  }

  void setName(llvm::StringRef N) { Name = N.str(); }
  void setKnownZero(llvm::APInt KZ) {
    assert(KZ.getBitWidth() == SizeInBytes * 8 &&
           "Sizes needs to be in sync!");
    KnownZero = KZ;
  }
  void setKnownOne(llvm::APInt KO) {
    assert(KO.getBitWidth() == SizeInBytes * 8 &&
           "Sizes needs to be in sync!");
    KnownOne = KO;
  }
  void setInvariantField() { Flags |= FLAG_INVARIANT_FIELD; }
  void setFieldType(FieldType T) { Type = T; }
  void setKnownNonNull() {
    assert(isReferenceField() && "makes sense for reference fields only");
    Flags |= FLAG_KNOWN_NON_NULL;
  }
  void setDereferenceableBytes(uint32_t D) {
    assert(isReferenceField() && "makes sense for reference fields only");
    DereferenceableInBytes = D;
  }
  void setAlignmentInBytes(uint32_t A) {
    assert(isReferenceField() && "makes sense for reference fields only");
    AlignmentInBytes = A;
  }
  void setJavaType(fsea::TypeUtils::CompileTimeKlassID K, bool IsExact) {
    assert(isReferenceField() && "makes sense for reference fields only");
    KlassID = K;
    Flags |= FLAG_KNOWN_TYPE;
    if (IsExact)
      Flags |= FLAG_KNOWN_EXACT_TYPE;
  }

  bool isReferenceField() const {
    return Type == FIELD_TYPE_REFERENCE;
  }

  bool isInvariantField() const {
    return (Flags & FLAG_INVARIANT_FIELD) != 0;
  }

  StringRef getName() const { return Name; }
  int32_t getOffsetInBytes() const { return OffsetInBytes; }
  uint32_t getSizeInBytes() const { return SizeInBytes; }

  uint32_t getDereferenceableBytes() const {
    assert(isReferenceField() && "makes sense for reference fields only");
    return DereferenceableInBytes;
  }
  uint32_t getAlignmentInBytes() const {
    assert(isReferenceField() && "makes sense for reference fields only");
    assert(
        (AlignmentInBytes && !(AlignmentInBytes & (AlignmentInBytes - 1))) &&
        "should be power of 2!");
    return AlignmentInBytes;
  }

  bool isKnownNonNull() const {
    assert(isReferenceField() && "makes sense for reference fields only");
    return (Flags & FLAG_KNOWN_NON_NULL) != 0;
  }

  std::optional<TypeUtils::JavaType> getJavaType() const {
    assert(isReferenceField() && "makes sense for reference fields only");
    if ((Flags & FLAG_KNOWN_TYPE) == 0)
      return std::nullopt;

    const bool IsExact = (Flags & FLAG_KNOWN_EXACT_TYPE) != 0;
    return TypeUtils::JavaType(KlassID, IsExact);
  }

  const llvm::APInt &getKnownZero() const { return KnownZero; }
  const llvm::APInt &getKnownOne() const { return KnownOne; }

  bool isKnownPositive() const { return getKnownZero().isNegative(); }

  FieldType getFieldType() {
    return (FieldType) Type;
  }

  template <typename FieldTraversalTy>
  void traverseFields(FieldTraversalTy &FT) {
    FT.beginInstance("FieldInfo");
    FT.doFieldString("Name", Name);
    FT.doFieldInt32("OffsetInBytes", OffsetInBytes);
    FT.doFieldUInt32("SizeInBytes", SizeInBytes);
    FT.doFieldUInt32("AlignmentInBytes", AlignmentInBytes);
    FT.doFieldUInt32("DereferenceableInBytes",
                              DereferenceableInBytes);
    uint32_t ID = KlassID.getID();
    FT.doFieldUInt32("KlassID", ID);
    KlassID = fsea::TypeUtils::CompileTimeKlassID(ID);
    FT.doFieldAPInt("KnownZero", KnownZero);
    FT.doFieldAPInt("KnownOne", KnownOne);
    FT.doFieldUInt32("Flags", Flags);
    FT.doFieldFieldType("Type", Type);
    FT.endInstance();
  }

  bool operator==(const FieldInfo& Other) const {
    return Name == Other.Name &&
           OffsetInBytes == Other.OffsetInBytes &&
           SizeInBytes == Other.SizeInBytes &&
           AlignmentInBytes == Other.AlignmentInBytes &&
           DereferenceableInBytes == Other.DereferenceableInBytes &&
           KlassID == Other.KlassID &&
           KnownZero == Other.KnownZero  &&
           KnownOne == Other.KnownOne &&
           Flags == Other.Flags &&
           Type == Other.Type;
  }

  static bool CompareByOffset(const FieldInfo &X, const FieldInfo &Y) {
    return X.getOffsetInBytes() < Y.getOffsetInBytes();
  }
};

class JavaTypeInfo { // TODO: Cachable

public:
  enum : uint32_t {
    FLAG_NONE = 0,
    FLAG_IS_INTERFACE = 1,
    FLAG_IS_ABSTRACT = 2,
    FLAG_IS_OBJECT_SIZE_EXACT = 4,
    FLAG_IS_ARRAY = 8,
    FLAG_IS_INSTANCE = 16,
  };

  JavaTypeInfo() = default;
  explicit JavaTypeInfo(std::string N, uint32_t OS, uint32_t AHS, uint32_t AES,
                        uint32_t F)
      : Name(N), ObjectSize(OS), ArrayHeaderSize(AHS), ArrayElementShift(AES),
        Flags(F) {
    assert((ArrayElementShift == 0 || isArray()) &&
           "ArrayElementShift must be 0 for non-arrays");
  }


  StringRef getName() const { return Name; }
  uint32_t getObjectSize() const { return ObjectSize; }
  uint32_t getArrayHeaderSize() const {
    assert(isArray() && "Doesn't make sense for non-arrays!");
    return ArrayHeaderSize;
  }
  uint32_t getArrayElementShift() const {
    assert(isArray() && "Doesn't make sense for non-arrays!");
    return ArrayElementShift;
  }

  bool isInterface() const {
    return (Flags & FLAG_IS_INTERFACE) != 0;
  }
  bool isAbstract() const {
    // For now when the VM doesn't know about FLAG_IS_ABSTRACT and only sets
    // FLAG_IS_INTERFACE implicitly treat interface types as abstract.
    // TODO: Once the VM is taught to set FLAG_IS_ABSTRACT assert that interface
    // is also abstract in the constructor and remove this special case.
    if (isInterface())
      return true;
    return (Flags & FLAG_IS_ABSTRACT) != 0;
  }
  bool isObjectSizeExact() const {
    return (Flags & FLAG_IS_OBJECT_SIZE_EXACT) != 0;
  }

  /// Return true if the type is known to be an array type.
  bool isArray() const {
    return (Flags & FLAG_IS_ARRAY) != 0;
  }

  /// Return true if the type is known to be an instance (non-array) type.
  bool isInstance() const {
    return (Flags & FLAG_IS_INSTANCE) != 0;
  }

  template <typename FieldTraversalTy>
  void traverseFields(FieldTraversalTy &FT) {
    FT.beginInstance("JavaTypeInfo");
    FT.doFieldString("Name", Name);
    FT.doFieldUInt32("ObjectSize", ObjectSize);
    FT.doFieldUInt32("ArrayHeaderSize", ArrayHeaderSize);
    FT.doFieldUInt32("ArrayElementShift", ArrayElementShift);
    FT.doFieldUInt32("Flags", Flags);
    FT.endInstance();
  }

  bool operator==(const JavaTypeInfo& Other) const {
    return Name == Other.Name &&
           ObjectSize == Other.ObjectSize &&
           ArrayHeaderSize == Other.ArrayHeaderSize &&
           ArrayElementShift == Other.ArrayElementShift &&
           Flags == Other.Flags;
  }

private:
  std::string Name;
  uint32_t ObjectSize;
  uint32_t ArrayHeaderSize;    // Must be zero for non-arrays
  uint32_t ArrayElementShift;  // Must be zero for non-arrays
  uint32_t Flags;
};

namespace VMInterface {
/// Return information about an object implied by it's type (and if it's an
/// array and the length is known, it's length).
std::optional<fsea::JavaTypeInfo>
getJavaTypeInfo(llvm::LLVMContext &C, const TypeUtils::JavaType &T,
                std::optional<uint64_t> ArrayLen);

/// Returns FieldInfo description for the given offset of the given type. A
/// successful query doesn't guarantee that the location being asked about is
/// dereferenceable. For example, if T describes an array type, the returned
/// FieldInfo would describe an array element assuming that it's in bounds. It's
/// up to the caller to check dereferenceability if this property is required.
std::optional<fsea::FieldInfo>
getFieldInfoAtOffset(llvm::LLVMContext &C, TypeUtils::JavaType T, bool IsNew,
                     int64_t Offset);

// TODO: fsea.array_length_offset_in_bytes only?
std::optional<uint64_t> getVMIntegerConstant(llvm::LLVMContext &C,
                                             llvm::StringRef ConstantName);

std::optional<fsea::TypeUtils::CompileTimeKlassID>
runTimeToCompileTimeKlassID(llvm::LLVMContext &C, uint64_t RTKID);
} // namespace VMInterface

/// Classification of the pointer uses from the escape analysis point of view.
enum UseEscapeKind {
  /// Object may escape through this use
  Escape,
  /// Object can't escape through this use
  NoEscape,
  /// Object can't escape through this use, but it produces a new value which
  /// needs to be tracked. E.g., getelementptr, bitcast, phi instructions.
  Alias
};

UseEscapeKind getUseEscapeKind(const llvm::Use *U);

namespace FlowSensitiveEA {
using namespace llvm::fsea::ExtendedIR;

/// Represents a pointer to a tracked allocation.
struct TrackedPointer {
  AllocationID AllocID;
  std::optional<int64_t> Offset; // std::nullopt means unknown offset, not zero.

  /// Alias is a tracked pointer with its offset known to be zero.
  bool isAlias() const { return Offset.has_value() && *Offset == 0; }

  TrackedPointer(AllocationID AllocID, std::optional<int64_t> Offset)
      : AllocID(AllocID), Offset(Offset) {}

  bool operator==(const TrackedPointer &TP) const {
    return AllocID == TP.AllocID && Offset == TP.Offset;
  };

  bool operator!=(const TrackedPointer &TP) const { return !(*this == TP); }

  bool operator<(const TrackedPointer &TP) const {
    return std::tie(AllocID, Offset) < std::tie(TP.AllocID, TP.Offset);
  };

  void print(raw_ostream &ROS, const State &S) const;
};

class State;

/// This is a tristate used to communicate the predecessor states to the merge
/// routine. The predecessor state can be either:
/// - BackedgeUnknown - an unknown state coming from a backedge
/// - UnreachableUnknown - an unknown state coming from a dead BB
/// - Known state
///
/// All possible States form a lattice. We use BlockOutState to describe an
/// element of this lattice.
///   - BackedgeUnknown, UnreachableUnknown - represent the top value of the
///     lattice. We distinguish between the top values coming from a backedge
///     and an unreachable block for the purposes of SSA construction.
///     An incoming value coming from an unreachable block is undef, while a
///     backedge incoming value is defined.
///   - Known empty state - the bottom value.
///   - Known non-empty state - all other values.
/// State::merge implements the meet operation on this lattice.
///
/// Here is an informal explanation why States form a lattice. Essentially State
/// is a set of tracked allocations, which is a subset of all instructions in
/// the function (strictly speaking it's a subset of all allocation instructions
/// and all PHIs). Merge operation is monotonic, i.e. for smaller input sets it
/// produces smaller merged sets. Thus the set of tracked allocations is a
/// lattice.
///
/// Each allocation in turn is described as a set of fields and tracked
/// pointers. Both sets monotonically grow during merge and the maximum size of
/// these sets is bounded by total number of instructions in function. So, these
/// are lattices as well.
///
/// Field values are represented as ExtendedValues. At merge a field value can
/// transition between these three states:
///   Default initialized value ->
///   Single value ->
///   Virtual PHI merging between different inputs
/// This transition is monotonic, e.g. once the field value is a virtual PHI it
/// can't become default initialized or a single value. The value of each field
/// is a lattice as well.
///
/// Essentially the State lattice is a product of several simpler lattices.
class BlockOutState {
  const State *S;
  enum {
    BackedgeUnknown = -2,
    UnreachableUnknown = -1
  };
  BlockOutState(const State *S) : S(S) {}
public:
  bool isBackedgeUnknown() {
    return S == reinterpret_cast<const State *>(BackedgeUnknown);
  }
  bool  isUnreachableUnknown() {
    return S == reinterpret_cast<const State *>(UnreachableUnknown);
  }
  bool isKnownState() {
    return !isBackedgeUnknown() && !isUnreachableUnknown();
  }
  const State *getValue() {
    assert(isKnownState() && "Must be a known state!");
    return S;
  }

  static BlockOutState getKnownState(const State *S) {
    return BlockOutState(S);
  }
  static BlockOutState getBackedgeUnknownState() {
    return BlockOutState(reinterpret_cast<const State *>(BackedgeUnknown));
  }
  static BlockOutState getUnreachableUnknownState() {
    return BlockOutState(reinterpret_cast<const State *>(UnreachableUnknown));
  }
};

using GetBlockOutState =
      std::function<BlockOutState(const BasicBlock *)>;
using GetAllocationIDForBlock =
      std::function<AllocationID(const BasicBlock *)>;

struct ExactAllocationState {
  unsigned LockCount = 0;

  /// This map keeps current values of allocation fields (per offset).
  /// std::map is used for its deterministic access order.
  std::map<int64_t, ExtendedValue> FieldValues;
  /// The set of fields which are known to be invariant
  /// (marked as invariant_start).
  SmallSet<int64_t, 4> InvariantFields;

  bool operator==(const ExactAllocationState &AS) const {
    return LockCount == AS.LockCount && FieldValues == AS.FieldValues &&
           InvariantFields == AS.InvariantFields;
  };

  bool operator!=(const ExactAllocationState &AS) const {
    return !(*this == AS);
  }

  bool empty() const {
    return LockCount == 0 && FieldValues.empty() && InvariantFields.empty();
  };

  std::optional<ExtendedValue> getFieldValue(int64_t Offset) const {
    auto FieldIt = FieldValues.find(Offset);
    if (FieldIt == FieldValues.end())
      return std::nullopt;
    return FieldIt->second;
  }
  void setFieldValue(int64_t Offset, ExtendedValue V) {
    FieldValues[Offset] = V;
  }

  bool isInvariantField(int64_t Offset) const {
    return InvariantFields.count(Offset) != 0;
  }
  void markInvariantField(int64_t Offset) {
    InvariantFields.insert(Offset);
  }

  /// Model monitor enter on the allocated object.
  /// Returns true is this operation was successfully modeled, false
  /// otherwise.
  bool monitorEnter() {
    // Protect from LockCount overflow. Normally we shouldn't see this, but
    // we can't assume sanity when processing dead code.
    if (LockCount == std::numeric_limits<decltype(LockCount)>::max())
      return false;
    LockCount++;
    return true;
  }
  /// Model monitor exit on the allocated object.
  /// Returns true is this operation was successfully modeled, false
  /// otherwise.
  bool monitorExit() {
    // Protect from monitorexit before monitorenter. Normally we shouldn't see
    // this, but we can't assume sanity when processing dead code.
    if (LockCount == 0)
      return false;
    LockCount--;
    return true;
  }

  /// T is the JavaType of the allocation.
  /// S is the EA state to print this exact allocation state for. This argument
  /// is optional, if passed more information will be printed about values
  /// that are tracked pointers in this state.
  void print(LLVMContext &C, std::optional<TypeUtils::JavaType> T, const State *S,
             raw_ostream &ROS) const;

  static std::optional<ExactAllocationState> getMergedAllocationState(
    VirtualContext &VContext, AllocationID ID,
    GetAllocationIDForBlock GetAllocID, const BasicBlock *BB,
    GetBlockOutState GetState);
  static std::optional<ExtendedValue>
  getMergedFieldValue(VirtualContext &VContext, AllocationID ID,
    GetAllocationIDForBlock GetAllocID, int64_t Offset,
    const BasicBlock *BB, GetBlockOutState GetState);
};

/// Symbolically describes how to produce the initialized state of the
/// allocation.
struct SymbolicAllocationState {
  /// Describes an initializing instruction in the symbolic state.
  ///
  /// Essentially we have a "symolic IR" to describe how to produce the
  /// initialized state of the allocation. We have 2 ways of representing
  /// instructions in this IR: an instruction with dedicated KindTy, e.g.
  /// PublicationBarrier, or an existing IR instruction. We represent IR
  /// instructions using two values:
  ///   I - the IR instruction which describes the state modification
  ///   BaseObject - the base object being modified by the IR instruction
  ///
  /// The meaning of this pair is: apply the same modification to the current
  /// allocation as the instruction "first" modifies the object "second".
  ///
  /// For example, given the IR:
  ///   %a.24 = gep %a, 24
  ///   %a.24.i32 = bitcast %a.24 to i32*
  ///   store i32 1, %a.24.i32
  /// Initializing instruction { I: "store 1, %a.24.i32", BaseObject: %a }
  /// means: store value 1 into offset 24 of the current allocation.
  class InitializingInstruction {
    enum KindTy {
      IRInstruction,
      PublicationBarrier
    };
    KindTy Kind;
    const Instruction *I;
    const Value *BaseObject;

    InitializingInstruction(KindTy Kind, const Instruction *I,
                            const Value *BaseObject)
        : Kind(Kind), I(I), BaseObject(BaseObject) {}

  public:
    static InitializingInstruction createIRInstruction(
        const Instruction *I, const Value *BaseObject) {
      return InitializingInstruction(IRInstruction, I, BaseObject);
    }
    static InitializingInstruction createPublicationBarrier() {
      return InitializingInstruction(PublicationBarrier, nullptr, nullptr);
    }

    bool isIRInstruction() const {
      return Kind == IRInstruction;
    }
    bool isPublicationBarrier() const {
      return Kind == PublicationBarrier;
    }

    const Instruction *getIRInstruction() const {
      assert(isIRInstruction() && "Doesn't make sense otherwise");
      return I;
    }
    /// If this initializing instruction is represented as an IR instruction
    /// returns it, otherwise returns null.
    const Instruction *getIRInstructionOrNull() const {
      if (isIRInstruction())
        return getIRInstruction();
      return nullptr;
    }
    const Value *getIRInstructionBaseObject() const {
      assert(isIRInstruction() && "Doesn't make sense otherwise");
      return BaseObject;
    }
    bool operator==(const InitializingInstruction &Other) const {
      return Kind == Other.Kind &&
             I == Other.I &&
             BaseObject == Other.BaseObject;
    }
    void print(llvm::raw_ostream &OS) const {
      switch (Kind) {
        case IRInstruction:
          OS << *I;
          break;
        case PublicationBarrier:
          OS << "  PUBLICATION_BARRIER";
      }
    }
  };

  SmallVector<InitializingInstruction, 16> InitializingInstructions;

  /// When a symbolic state contains an initialization from memory (e.g. a
  /// memcpy) it is only valid as long as the source memory is unmodified. We
  /// keep track of memory sources in a symbolic state, so as to invalidate the
  /// state.
  ///
  /// MemorySource is a wrapper over a pointer which also contains the
  /// corresponding tracked pointer, if the pointer is a pointer to a tracked
  /// allocation. This information will be used to check if the memory source
  /// can be clobbered by a memory operation.
  ///
  /// We have to cache the tracked pointer together with the pointer value
  /// because the mapping between values and tracked pointers is only valid in
  /// the current EA state. We may not be able to get the tracked pointer for
  /// Ptr in some later state. See the comment about State::TrackedPointer for
  /// details.
  struct MemorySource {
    const Value *Ptr;
    std::optional<TrackedPointer> TP;
    MemorySource(const Value *Ptr, std::optional<TrackedPointer> TP)
        : Ptr(Ptr), TP(TP) {};
    bool operator==(const MemorySource &Other) const {
      return Ptr == Other.Ptr && TP == Other.TP;
    }
  };
  SmallVector<MemorySource, 4> MemorySources;

  /// Returns true if the object memory state is not modified.
  bool isUnmodified() const {
    return std::none_of(InitializingInstructions.begin(),
                        InitializingInstructions.end(),
                        isModifyingAllocationContent);
  }

  bool operator==(const SymbolicAllocationState &AS) const {
    return InitializingInstructions == AS.InitializingInstructions &&
           MemorySources == AS.MemorySources;
  };

  bool operator!=(const SymbolicAllocationState &AS) const {
    return !(*this == AS);
  }

  void print(raw_ostream &ROS) const;

  static std::optional<SymbolicAllocationState>
  getMergedAllocationState(GetAllocationIDForBlock GetAllocID,
                           const BasicBlock *BB, GetBlockOutState GetState);

  /// Returns true if the initializing instruction modifies allocations's
  /// content, false otherwise.
  ///
  /// For example, returns true for a store, false for a publication barrier.
  static bool isModifyingAllocationContent(const InitializingInstruction &II);
};

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &OS,
    const SymbolicAllocationState::InitializingInstruction &II);

/// Allocation - current state of a single allocation, keeping track of all
/// the stores happening into the allocation fields from allocation call down to
/// a particular point of execution.
///
/// There are two kinds of allocations we track: regular allocations and
/// PHI-merged allocations.
///
/// Regular allocations are represented by an allocation abstraction (e.g.
/// @fsea.new_instance) call in the program.
///
/// When a PHI merges unescaped allocations across all incoming paths and the
/// incoming allocations are not accessed below the PHI we can treat the PHI
/// as a new tracked allocation. Such allocation is called a PHI-merged
/// allocation.
class Allocation {
public:
  AllocationID ID;
  const Value *KlassID;
  std::optional<ExtendedValue> ArrayLength; // std::nullopt for instance allocations

  // For instance allocations always std::nullopt. For array allocations
  // std::nullopt means that we lost tracking of the value (e.g. different
  // values came from different paths of a phi-merged array).
  std::optional<ExtendedValue> ZeroInitializeFrom;

  const Instruction *NewInstruction;

  /// Indicates the the object needs a publication barrier before escape.
  bool NeedsPublicationBarrier = false;

  std::optional<ExactAllocationState> ExactState;
  std::optional<SymbolicAllocationState> SymbolicState;

  /// List of all tracked pointers into this allocation. All these instructions
  /// are keys of State::TrackedPointers.
  SmallSet<ExtendedValue, 16> TrackedPointers;

  /// ContributingAllocations always contain the allocation itself.
  ///
  /// For a PHI-merged allocation ContributingAllocations also contains all
  /// allocations which transitively contribute to this PHI-merged allocation.
  SmallSet<AllocationID, 4> ContributingAllocations;

  Allocation(AllocationID ID, const Value *KlassID,
             std::optional<ExtendedValue> ArrayLength,
             std::optional<ExtendedValue> ZeroInitializeFrom,
             const Instruction *I);

  Allocation() = default;
  Allocation(const Allocation &) = default;
  Allocation(Allocation &&) = default;
  Allocation &operator=(const Allocation &) = default;
  Allocation &operator=(Allocation &&) = default;

  bool operator==(const Allocation &A) const {
    return ID == A.ID && KlassID == A.KlassID &&
           NewInstruction == A.NewInstruction &&
           ArrayLength == A.ArrayLength &&
           ZeroInitializeFrom == A.ZeroInitializeFrom &&
           TrackedPointers == A.TrackedPointers &&
           ExactState == A.ExactState &&
           SymbolicState == A.SymbolicState &&
           ContributingAllocations == A.ContributingAllocations &&
           NeedsPublicationBarrier == A.NeedsPublicationBarrier;
  };

  bool operator!=(const Allocation &A) const { return !(*this == A); }

  bool isArray() const { return ArrayLength.has_value(); }

  /// Checks whether the given Allocation computed after materialization of
  /// virtual values (\p AfterMaterialization) is equivalent to the current
  /// allocation computed before materialization of virtual values.
  ///
  /// Materialization of virtual values may produce extra tracked pointers.
  /// This is a relaxed version of equality check which considers two states
  /// equivalent even if allocations in \p AfterMaterialization state have
  /// extra tracked pointers.
  ///
  /// Note, that this relation is not symmetric, it only allows extra tracked
  /// pointers in \p AfterMaterialization allocation.
  bool isEquivalentAfterMaterialize(const Allocation &AfterMaterialization)
      const {
    if (ID != AfterMaterialization.ID ||
        KlassID != AfterMaterialization.KlassID ||
        NewInstruction != AfterMaterialization.NewInstruction ||
        ArrayLength != AfterMaterialization.ArrayLength ||
        ZeroInitializeFrom != AfterMaterialization.ZeroInitializeFrom ||
        ExactState != AfterMaterialization.ExactState ||
        SymbolicState != AfterMaterialization.SymbolicState ||
        ContributingAllocations != AfterMaterialization.ContributingAllocations ||
        NeedsPublicationBarrier != AfterMaterialization.NeedsPublicationBarrier)
      return false;
    // All of the tracked pointers in this allocations must also be in
    // AfterMaterialization
    return all_of(TrackedPointers,
                  [&AfterMaterialization] (const ExtendedValue &EV) {
                    return AfterMaterialization.TrackedPointers.count(EV);
                  });
  }

  bool isTrackedField(int64_t Offset, Type *Ty) const;
  bool isTrackedField(int64_t Offset, unsigned SizeInBytes) const;
  std::optional<ExtendedValue> getInitialFieldValue(int64_t Offset,
                                                    Type *Ty) const;

  bool isPHIMergedAllocation() const {
    assert(isa<PHINode>(NewInstruction) ==
              (ContributingAllocations.size() > 1) &&
           "Only PHI-merged allocations can have incoming allocations!");
    return isa<PHINode>(NewInstruction);
  }

  /// S is the EA state to print this allocation state for. This argument
  /// is optional, if passed more information will be printed about values
  /// that are tracked pointers in this state.
  void print(raw_ostream &ROS, const State *S = nullptr) const;
  void dumpInstruction() const { NewInstruction->dump(); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); }
#endif

private:
  std::optional<fsea::FieldInfo> getFieldInfo(int64_t Offset, Type *Ty) const;
  std::optional<fsea::FieldInfo> getFieldInfo(int64_t Offset,
                                              unsigned SizeInBytes) const;
};

struct ExactStateInstVisitor;
struct FlowSensitiveEscapeAnalysis;
class State {
  friend struct DeoptStateInstVisitor;
  friend struct ExactStateInstVisitor;
  friend struct SymbolicStateInstVisitor;

public:
    /// DeoptState is essentially a wrapper over CallBase * with the state.
  class DeoptState {
    CallBase *Call;

    /// Tracked allocations transitively referenced from the deopt state.
    SetVector<AllocationID> ReferredAllocIDs;

  public:
    DeoptState(CallBase *CB, const State &S);

    CallBase *getCall() const {
      return Call;
    }
    bool refersToAllocation(TrackedPointer TP) const {
      return ReferredAllocIDs.contains(TP.AllocID);
    }
    bool operator==(const DeoptState &Other) const {
      return Call == Other.Call;
    }
    bool operator!=(const DeoptState &Other) const {
      return !(*this == Other);
    }
  };

private:
  /// Allocations - all allocations that are tracked in this state.
  /// An allocation is tracked as long as its pointer does not escape.
  MapVector<AllocationID, Allocation> Allocations;

  /// TrackedPointers contains all pointers which can be used to access the
  /// tracked allocation in the context of the current state. Specifically,
  /// it contains:
  /// - Allocation aliases, i.e. values which point directly to a tracked
  ///   allocation. This includes the allocation instruction itself. Note that
  ///   there might be other values pointing to the allocation. For example:
  ///     a = new A
  ///     store a, a.self
  ///     a' = load a.self
  ///   Here a and a' are aliases for the same allocation.
  /// - Derived pointers computed as bitcasts, addrspacecasts, GEPs off tracked
  ///   allocations.
  ///
  /// This map has 2 main purposes:
  /// - to track loads from and stores to allocations through derived pointers.
  /// - to track allocation escapes. If any new usage is found to be an escape
  ///   point then the allocation is made non-trackable (i.e. removed from
  ///   Allocations with all its tracked pointers).
  ///
  /// Because we track all possible pointers which can be used to access a
  /// tracked allocation we can use this information for aliasing checks. If:
  /// - two pointers point to distinct allocations, or
  /// - one pointer points to a tracked allocation and the other one does not
  /// then the pointers can't alias.
  ///
  /// Note the limitation that TrackedPointers contains pointers which can be
  /// used to access the allocation in the context of the *current* state!
  /// TrackedPointers may not contain all possible pointers which can be used to
  /// access the allocation. For example:
  /// - it doesn't contain tracked pointers which have not yet been visited,
  /// - a PHI-merged allocation doesn't contain tracked pointers to the
  /// contributing allocations (we have proved that these pointers are not used
  /// after the PHI).
  ///
  /// Because of this limitation we can only do aliasing check for the pointers
  /// used in the same context! For example, we can check that the source and
  /// the dest of a memmove don't alias, thus the memmove can be converted to a
  /// memcpy. But we can't check whether a memory access happened in one
  /// instruction aliases with a memory access happened in some other
  /// instruction.
  ///
  /// It might be tempting to extend TrackedPointers with all the pointers which
  /// could have been used to access the allocation in any context preceding the
  /// current state. This would have made some of the aliasing queries between
  /// different instructions possible. But this extended definition of tracked
  /// pointers doesn't interact well with PHI-merged allocations in loops.
  /// Consider the following example:
  ///
  ///     %e = new char[]
  ///     br loop
  ///
  ///   loop:
  ///     %p = phi %e, %l   ; a PHI-merged tracked allocation
  ///     %l = new char[]
  ///     load %p[0]
  ///     store 42 -> %l[0]
  ///     br loop
  ///
  /// The load and the store alias: the load may read the value written by the
  /// store on the previous iteration. Using the extended definition of tracked
  /// pointers doesn't give us the right answer though.
  ///
  ///     %e = new char[]
  ///     br loop
  ///
  ///   loop:
  ///     ; The extended definition of tracked pointers will add %e and %l as
  ///     ; tracked pointers to the PHI-merged allocation.
  ///     %p = phi %e, %l   ; %p tracked pointers: %p, %e, %l
  ///     ; We are in a difficult situation now. %l is a tracked pointer to %p,
  ///     ; but it can also be used to modify the new allocation %l from the
  ///     ; current iteration. Assume that we reset %l to point to the new
  ///     ; allocation.
  ///     %l = new char[]   ; %p tracked pointers: %p, %e
  ///                       ; %l tracked pointers: %l
  ///     ; Now we can't see the store as an alias of this load, because %l and
  ///     ; %p point to distinct tracked allocations.
  ///     ; Note that if we move the load above the allocation, we would get the
  ///     ; aliasing fact right!
  ///     load %p[0]
  ///     store 42 -> %l[0]
  ///     br loop
  ///
  /// Instead, if you want to make an aliasing query between different
  /// instructions you should query the allocations being accessed from the
  /// states corresponding to these instructions. Having the allocations at hand
  /// you can check if the sets of contributing allocations intersect. Here is
  /// how it works for the example above:
  ///
  ///     %e = new char[]
  ///     br loop
  ///
  ///   loop:
  ///     %p = phi %e, %l   ; contributing allocations %p, %e, %l
  ///     %l = new char[]   ; contributing allocations %l
  ///     load %p[0]        ; accessing %p {contributing %p, %e, %l}
  ///     store 42 -> %l[0] ; accessing %l {contributing %l}
  ///     br loop
  ///
  /// The load aliases with the store because the sets of contributing
  /// allocations overlap. See SymbolicStateInstVisitor::mayClobber for an
  /// example of such an analysis.
  DenseMap<ExtendedValue, TrackedPointer> TrackedPointers;

  /// The last allocation state which is still valid and can be reused.
  std::optional<DeoptState> LastDeoptState;

  /// Removes the specified allocation and its tracked pointers from the state.
  void remove(AllocationID ID) {
    auto A = Allocations.find(ID);
    assert(A != Allocations.end() && "allocation already removed??");

    // Erase tracked pointers to this allocation first.
    for (auto D : A->second.TrackedPointers)
      TrackedPointers.erase(D);

    Allocations.erase(A);
  }

public:
  bool isEmpty() const {
    return Allocations.empty() && !LastDeoptState;
  }

  auto allocations_begin() const {
    return Allocations.begin();
  }

  auto allocations_end() const {
    return Allocations.end();
  }

  /// Returns the tracked pointer for this ExtendedValue, otherwise returns
  /// std::nullopt.
  const std::optional<TrackedPointer>
  getTrackedPointer(const ExtendedValue V) const {
    auto AP = TrackedPointers.find(V);
    if (AP == TrackedPointers.end())
      return std::nullopt;
    return AP->second;
  }

  /// Start tracking a (possibly derived) pointer to the allocation in its
  /// Allocation. Optional Offset should reflect full offset of derived
  /// pointer from the start of allocated object. Unspecified Offset means that
  /// the pointer is not an alias (i.e. it differs from zero offset).
  void addTrackedPointer(ExtendedValue Ptr, TrackedPointer TP) {
    auto *A = getAllocation(TP.AllocID);
    assert(A && "Must be a valid ID");
    if (!TrackedPointers.try_emplace(Ptr, TP).second) {
      assert(*getTrackedPointer(Ptr) == TP && "Tracked pointer mismatch!");
      return;
    }
    auto Inserted = A->TrackedPointers.insert(Ptr);
    assert(Inserted.second && "must be new");
    (void)Inserted;
  }

  /// Start tracking an allocation \p Alloc with constant klass-id \p KlassID
  /// and constant \p ArrayLength (if this is an array allocation).
  Allocation &
  addTrackedAllocation(const Instruction *Alloc, const Value *KlassID,
                       std::optional<ExtendedValue> ArrayLength,
                       std::optional<ExtendedValue> ZeroInitializeFrom) {
    AllocationID NewID = createAllocationID(Alloc);
    SmallSet<AllocationID, 4> ContributingAllocations;
    ContributingAllocations.insert(NewID);
    return addTrackedAllocation(NewID, Alloc, KlassID, ArrayLength,
                                ZeroInitializeFrom, ContributingAllocations);
  }
  Allocation &addTrackedAllocation(
      AllocationID ID, const Instruction *Alloc, const Value *KlassID,
      std::optional<ExtendedValue> ArrayLength,
      std::optional<ExtendedValue> ZeroInitializeFrom,
      const SmallSet<AllocationID, 4> &ContributingAllocations);

  /// Returns nullptr if \p ID does not point to an allocation tracked in this
  /// state.
  Allocation *getAllocation(AllocationID ID) {
    auto A = Allocations.find(ID);
    if (A == Allocations.end())
      return nullptr;
    return &A->second;
  }

  /// Returns nullptr if \p ID does not point to an allocation tracked in this
  /// state.
  const Allocation *getAllocation(AllocationID ID) const {
    // Same as above but non-const this. Used const_cast to avoid repeating the
    // same code.
    return const_cast<State *>(this)->getAllocation(ID);
  }

  /// Returns nullptr if AllocID field of \p TP does not point to an allocation
  /// tracked in this state. This pointer stales if its allocation escapes.
  Allocation *getAllocation(std::optional<TrackedPointer> TP) {
    if (!TP)
      return nullptr;
    return getAllocation(TP->AllocID);
  }

  /// Returns nullptr if AllocID field of \p TP does not point to an allocation
  /// tracked in this state. This pointer stales if its allocation escapes.
  const Allocation *getAllocation(std::optional<TrackedPointer> TP) const {
    // Same as above but non-const this. Used const_cast to avoid repeating the
    // same code.
    return const_cast<State *>(this)->getAllocation(TP);
  }

  /// Returns field value for the specified tracked pointer augmented by the
  /// specified offset. If the state does not have this field set then its
  /// initial zero value is returned. If \p IsTypeStrict is false then the
  /// result value type may be other than the requested type \p Ty but they both
  /// are either of GC pointer type or not. The size of the returned type is
  /// always the same as the size of the requested type.
  std::optional<ExtendedValue>
  getFieldValue(Type *Ty, const std::optional<TrackedPointer> &TP,
                uint64_t Offset = 0, bool IsTypeStrict = true) const {
    if (!TP || !TP->Offset.has_value())
      return std::nullopt;

    auto Alloc = getAllocation(TP);
    if (!Alloc || !Alloc->ExactState)
      return std::nullopt;

    Offset += *TP->Offset;

    if (!Alloc->isTrackedField(Offset, Ty))
      return std::nullopt;

    if (auto FieldValue = Alloc->ExactState->getFieldValue(Offset)) {
      if (FieldValue->getType() == Ty ||
          (!IsTypeStrict &&
           isGCPointerType(Ty) == isGCPointerType(FieldValue->getType())))
        return FieldValue;
      return std::nullopt;
    }

    return Alloc->getInitialFieldValue(Offset, Ty);
  }

  const std::optional<DeoptState> &getLastAvailableState() const {
    return LastDeoptState;
  }

  /// Check for equality of two allocation maps disregard of their stored order.
  static bool equal(const MapVector<AllocationID, Allocation> &A1,
                    const MapVector<AllocationID, Allocation> &A2) {
    if (A1.size() != A2.size())
      return false;

    for (const auto &IdState1 : A1) {
      auto IdState2 = A2.find(IdState1.first);
      if (IdState2 == A2.end() || IdState1.second != IdState2->second)
        return false;
    }
    return true;
  }

  bool operator==(const State &S) const {
    return equal(Allocations, S.Allocations) &&
           TrackedPointers == S.TrackedPointers &&
           LastDeoptState == S.LastDeoptState;
  }

  bool operator!=(const State &S) const { return !(*this == S); }

  static bool isEquivalentAfterMaterialize(
      const MapVector<AllocationID, Allocation> &A1,
      const MapVector<AllocationID, Allocation> &A2) {
    if (A1.size() != A2.size())
      return false;

    for (const auto &IdState1 : A1) {
      auto IdState2 = A2.find(IdState1.first);
      if (IdState2 == A2.end() ||
          !IdState1.second.isEquivalentAfterMaterialize(IdState2->second))
        return false;
    }
    return true;
  }

  /// Checks whether the given state computed after materialization of virtual
  /// values (\p AfterMaterialization) is equivalent to the current state
  /// computed before materialization of virtual values.
  ///
  /// Materialization of virtual values may produce extra tracked pointers.
  /// This is a relaxed version of equality check which considers two states
  /// equivalent even if allocations in \p AfterMaterialization state have
  /// extra tracked pointers.
  ///
  /// Note, that this relation is not symmetric, it only allows extra tracked
  /// pointers in \p AfterMaterialization state.
  bool isEquivalentAfterMaterialize(const State &AfterMaterialization) const {
    if (!isEquivalentAfterMaterialize(Allocations,
                                      AfterMaterialization.Allocations))
      return false;
    auto ContainsInAfterMaterialization = [&AfterMaterialization](
        const std::pair<ExtendedValue, TrackedPointer> &E) {
      auto It = AfterMaterialization.TrackedPointers.find(E.first);
      if (It == AfterMaterialization.TrackedPointers.end())
        return false;
      return E.second == It->second;
    };
    if (!all_of(TrackedPointers, ContainsInAfterMaterialization))
      return false;
    return LastDeoptState == AfterMaterialization.LastDeoptState;
  }

  /// Get all tracked allocation transitively referenced from the start set of
  /// possible values. Derived pointers are replaced with their tracked
  /// allocations.
  ///
  /// SetVector result type is used to get deterministic order of the result set
  /// because it matters for further processing order (particularly for the
  /// dematerialization order and IDs of lazy objects).
  SetVector<AllocationID>
  getAllocationClosure(SmallVectorImpl<ExtendedValue> &&Worklist) const;
  SetVector<AllocationID> getAllocationClosure(ExtendedValue EV) const {
    SmallVector<ExtendedValue, 1> Worklist = {EV};
    return getAllocationClosure(std::move(Worklist));
  }

  /// Get all tracked allocation transitively referenced from the given
  /// allocation \p A, not including A itself.
  /// (Derived pointers are replaced with their tracked allocations)
  ///
  /// SetVector result type is used to get deterministic order of the result set
  /// although it might be unnecessary for current usage.
  SetVector<AllocationID>
  getAllocationContentClosure(const Allocation *A) const;

  bool isDereferenceablePointer(const TrackedPointer TP, Type *Ty) const;

  /// Collects the set of all unescaped allocation. This set is a superset of
  /// tracked allocations (Allocations map).
  ///
  /// Multiple allocations merged at PHIs are represented by a single PHI-merged
  /// allocation. The contributing allocation are not tracked after the merge
  /// (we've proven that they can not be accessed after the merge), but these
  /// allocations are unescaped.
  ///
  /// Unescaped allocations is a union of ContributingAllocations for all tracked
  /// allocations in this state.
  ///
  /// If computing this set becomes a bottleneck this set can be made a part of
  /// the tracked state.
  void collectUnescapedAllocations(SmallSet<AllocationID, 16> &Result) const {
    for (auto &A : Allocations)
      Result.insert(A.second.ContributingAllocations.begin(),
                    A.second.ContributingAllocations.end());
  }

  bool isUnescapedAllocation(std::optional<TrackedPointer> TP) const {
    if (!TP)
      return false;
    if (Allocations.count(TP->AllocID))
      return true;
    for (auto &A : Allocations)
      if (A.second.ContributingAllocations.count(TP->AllocID))
        return true;
    return false;
  }

  /// Returns the set of allocations which escaped in the current state
  /// compared to the previous state.
  SetVector<AllocationID> getEscapedAllocations(const State &PrevState) const {
    // Compute the set of unescaped allocations lazily. We might be able to
    // answer the question using only Allocations map (if none of the
    // allocations from PrevState escaped).
    SmallSet<AllocationID, 16> UnescapedAllocations;
    bool UnescapedAllocationsInitialized = false;

    SetVector<AllocationID> Escaped;
    for (auto &A : PrevState.Allocations) {
      if (!Allocations.count(A.first)) {
        if (!UnescapedAllocationsInitialized) {
          collectUnescapedAllocations(UnescapedAllocations);
          UnescapedAllocationsInitialized = true;
        }
        if (!UnescapedAllocations.count(A.first))
          Escaped.insert(A.first);
      }
    }
    return Escaped;
  }

protected:
  /// Escape the given allocation and all directly and indirectly referenced
  /// tracked allocations.
  /// Returns true if at least one tracked allocation escapes.
  bool escape(AllocationID ID);

  /// Escape all tracked allocations directly or indirectly referenced from the
  /// given list of values.
  /// Returns true if at least one tracked allocation escapes.
  bool escape(SmallVectorImpl<ExtendedValue> &&Values);

  /// Mark all allocations referred from the specified \p a as escaped. It is
  /// similar to the escape() methods but keep the given allocation non-escaped.
  /// Returns true if at least one tracked allocation escapes.
  bool escapeContent(Allocation *A);
  bool escapeContent(AllocationID ID) {
    if (auto *A = getAllocation(ID))
      return escapeContent(A);
    return false;
  }

  /// Mark allocation as unable to track its field values.
  /// Returns true if the content was trackable before this call.
  bool markContentUntrackable(AllocationID ID);

public:
  static State merge(FlowSensitiveEscapeAnalysis &EA, const BasicBlock *BB,
                     const DominatorTree &DT, GetBlockOutState GetState);

  enum FieldValueType {
    UnreachableUnknown, BackedgeUnknown, NotInitialized, Initialized
  };
  /// A helper function to get or create a VirtualPHI and initialize it with
  /// incoming values.
  static ExtendedValue getOrCreateVirtualPHIForField(
    VirtualContext &VContext, const BasicBlock *BB, Type *FieldTy,
    AllocationID ID, int64_t Offset,
    SmallDenseMap<const BasicBlock *, ExtendedValue, 8> &IncomingValues,
    SmallDenseMap<const BasicBlock *, FieldValueType, 8> &IncomingTypes);

  /// Prints tracked allocations sorted by their names.
  void print(raw_ostream & ROS) const;

  /// Check the consistency of the current state.
  void verify() {
#ifndef NDEBUG
    // Check that TrackedPointers in the state match with TrackedPointers in
    // the corresponding allocations.
    for (auto It : TrackedPointers)
      assert(getAllocation(It.second)->TrackedPointers.count(It.first));

    for (auto &A : Allocations) {
      assert(A.first == A.second.ID && "ID mismatch?");
      assert(A.second.ContributingAllocations.contains(A.first) &&
             "ContributingAllocations must contain the allocation itself");
      // Check that TrackedPointers in the allocations match with the
      // TrackedPointers in the state.
      for (auto EV : A.second.TrackedPointers)
        assert(getAllocation(getTrackedPointer(EV)) == &A.second);
    }
#endif
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); }
#endif
private:
  SmallSet<ExtendedValue, 8> getMergedValues(ExtendedValue PHI,
                                             const BasicBlock &BB,
                                             GetBlockOutState GetState);
  bool applyPhi(const ExtendedValue PHI, const BasicBlock &BB,
                GetBlockOutState GetState);

  /// Merges the given allocation into the current state. It takes the states
  /// for the given allocation in all predecessors of BB and tries to represent
  /// the result of the merge as an allocation in the current state.
  ///
  /// Returns true if the merged allocation was added into the state, false
  /// otherwise.
  bool mergeAllocation(VirtualContext &VContext, AllocationID ID,
                       const BasicBlock *BB, GetBlockOutState GetState);

  SetVector<AllocationID> tryMergeAllocationsAtPHIs(VirtualContext &VContext,
    const BasicBlock *BB, const DominatorTree &DT, GetBlockOutState GetState,
    SmallVectorImpl<AllocationID> &EscapeContent);

  /// PHI-merge candidate is a list of incoming allocations and a list of all
  /// PHINodes which merge tracked pointers to the incoming allocations.
  /// In case of a successful merge the incoming allocations are represeted as
  /// one PHI-merged allocation. All the PHINodes become tracked pointers to
  /// the merged allocation.
  struct PHIMergeCandidateInfo {
    using AllocationList = SmallVector<AllocationID, 4>;
    const AllocationList *IncomingAllocations;
    DenseMap<const PHINode *, std::optional<int64_t>> PHIOffsets;
  };
  bool tryMergeAllocationsAtPHI(VirtualContext &VContext,
    PHIMergeCandidateInfo &CandidateInfo,
    const DominatorTree &DT, GetBlockOutState GetState,
    SmallVectorImpl<AllocationID> &EscapeContent);
  std::optional<ExtendedValue> getVPHIForMergedArrayLength(
    VirtualContext &VContext, AllocationID ID,
    const PHIMergeCandidateInfo::AllocationList *IncomingAllocations,
    const BasicBlock *BB, GetBlockOutState GetState);

  bool isPHIMergeLegal(const PHINode *ZeroOffsetPHI,
    PHIMergeCandidateInfo &CandidateInfo,
    const DominatorTree &DT, GetBlockOutState GetState);
};

struct DeoptStateInstVisitor {
  State &S;

  DeoptStateInstVisitor(State &S) : S(S) {}

  static bool canUseDeoptState(CallBase *Call);
  bool invalidatesDeoptState(std::optional<State::DeoptState> DeoptState,
                             Instruction &I);
  bool visitInstruction(Instruction &I);
};

struct ExactStateInstVisitor : public InstVisitor<ExactStateInstVisitor, bool> {
  State &S;
  VirtualContext &VContext;
  /// Specifies if the fixed point has been reached. It given true can be used
  /// to assert that no value gets different input.
  const bool FixedPointReached;

  ExactStateInstVisitor(State &S, VirtualContext &VContext,
                        bool FixedPointReached = true)
    : S(S), VContext(VContext), FixedPointReached(FixedPointReached) {};

  /// Apply operations - apply the effect of a particular piece of IR
  /// to the given state. Returns true if state changes.
  bool applyAtomicCmpXchg(const AtomicCmpXchgInst &ACXI,
                          const TrackedPointer PtrTP);
  bool applyAtomicRMW(const AtomicRMWInst &ARMW,
                      const TrackedPointer PtrTP);
  bool applyOperandUse(const Use &U, const TrackedPointer TP);
  bool applyStoreValueUse(const StoreInst &SI, const TrackedPointer ValueTP);
  bool applyStorePointerUse(const Instruction &I, ExtendedValue ValueOp,
                            const TrackedPointer PtrTP);
  std::optional<bool> applyMemcpy(const AtomicMemCpyInst &AMI);
  bool applyNewAllocation(const Instruction &I);
  bool applyInvariantStart(const IntrinsicInst &InvariantStartCall);
  bool applyPublicationBarrier(const CallBase *PublicationBarrierCall);
  bool applyAlias(const Instruction &I, Value *Op);

  bool visitGetElementPtrInst(GetElementPtrInst &I);
  bool visitBitCastInst(BitCastInst &I);
  bool visitAddrSpaceCastInst(AddrSpaceCastInst &I);
  bool visitCallBase(CallBase &I);

  bool visitStoreInst(StoreInst &SI);
  bool visitLoadInst(LoadInst &LI);
  bool visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
  bool visitAtomicRMWInst(AtomicRMWInst &I);

  bool visitInstruction(Instruction &I);
};

struct NewArrayDesc {
  const Allocation *ArrayAlloc;
  NewArrayDesc(const Allocation *ArrayAlloc) : ArrayAlloc(ArrayAlloc) {}

  static std::optional<Value *> getLengthInElements(Value *LengthInBytes,
                                                    uint64_t ElementShift);

  bool isArrayLength(Value *V);
  bool isMemcpyLengthEqualToArrayLength(Value *LengthInBytes);
  std::optional<unsigned> getArrayElementShift();
  std::optional<unsigned> getArrayHeaderSize();
};

struct SymbolicStateInstVisitor
    : public InstVisitor<SymbolicStateInstVisitor, bool> {
  State &S;
  BatchAAResults &AA;
  LazyValueInfo &LVI;

  SymbolicStateInstVisitor(State &S, BatchAAResults &AA,
                           LazyValueInfo &LVI) : S(S), AA(AA), LVI(LVI) {};

  bool applyInitializingInstruction(Instruction &I, Value *OpV);
  bool tryForwardInitializingInstructions(AtomicMemCpyInst *AMI,
                                          TrackedPointer DestTP);
  bool applyPublicationBarrier(const CallBase *PublicationBarrierCall);
  bool applyMemcpy(AtomicMemCpyInst *AMI);

  bool visitCallBase(CallBase &I);
  bool visitStoreInst(StoreInst &SI);
  bool visitInstruction(Instruction &I);

  bool mayClobber(const Instruction *I,
                  SymbolicAllocationState::MemorySource MS);
  bool visit(Instruction *I);
};

struct StateInstVisitor {
  State &S;
  DeoptStateInstVisitor DeoptStateVisitor;
  ExactStateInstVisitor ExactStateVisitor;
  SymbolicStateInstVisitor SymbolicStateVisitor;

  StateInstVisitor(State &S, FlowSensitiveEscapeAnalysis &EA,
                   bool FixedPointReached = true);
  bool applyPublicationBarrier(const CallBase *PublicationBarrierCall);
  bool visit(Instruction *I);
};

std::optional<bool> Equal(std::optional<TrackedPointer> LHSTP,
                          std::optional<TrackedPointer> RHSTP);
std::optional<bool> Equal(const State &S, ExtendedValue LHS, ExtendedValue RHS);
Type *GetSrcPointerElementType(const AtomicMemCpyInst &AMI);

/// Basically just a pair of input and output states.
struct BasicBlockState {
  /// \p In stores the state merged from predecessors and applied phis and vphis
  /// of its basic block. Applying phis is needed to catch possible allocation
  /// escapes caused by merged values. The state equality predicate compares
  /// vphis by their pointers. So, a new vphi input that comes from a back
  /// branch is not detected but might cause an allocation escape which must be
  /// taken into account in the next iteration.
  State In;
  State Out;

  void print(raw_ostream &ROS) const {
    ROS << ";  In:\n";
    In.print(ROS);
    ROS << ";  Out:\n";
    Out.print(ROS);
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); }
#endif
};

struct FlowSensitiveEscapeAnalysis {
  const Function &F;
  const DominatorTree &DT;
  const DataLayout &DL;
  BatchAAResults BatchAA;
  LazyValueInfo &LVI;

  VirtualContext VContext;

  MapVector<const BasicBlock *, int> BlockRPON;
  DenseMap<const BasicBlock *, BasicBlockState> BlockStates;

  FlowSensitiveEscapeAnalysis(const Function &F, const DominatorTree &DT,
                              AAResults &AA, LazyValueInfo &LVI);

  void eraseBlockState(const BasicBlock *BB) {
    BlockStates.erase(BB);
    VContext.forgetForBlock(BB);
  }

  void clear(bool ClearBlockPRON = false) {
    BlockStates.clear();
    VContext.clear();
    if (ClearBlockPRON)
      BlockRPON.clear();
  }

  const VirtualContext &getVirtualContext() const { return VContext; }
  VirtualContext &getVirtualContext() { return VContext; }

  /// Returns a VirtualValue bound to the instruction \p I in the state \p S.
  const VirtualValue *getVirtualValue(const State &S,
                                      const Instruction &I) const;

  /// Calculates BB states starting from the entry block.
  void calculateBBStates();

  /// Calculates BB states starting from the given set of blocks.
  void calculateBBStates(SmallPtrSetImpl<const BasicBlock *> &InitialWorklist);

  /// Returns the set of allocations which escape on the given edge.
  /// These are the allocations from From Out state which escaped during merge
  /// of To predecessors.
  SetVector<std::pair<AllocationID, const Allocation *>>
  getEscapedAllocationsForEdge(const BasicBlock *From, const BasicBlock *To)
      const;

  void print(raw_ostream &ROS, bool PrintType) const;

  class Writer : public AssemblyAnnotationWriter {
    function_ref<FlowSensitiveEscapeAnalysis &(const Function &)> GetEA;
    DenseMap<const Value *, std::string> InfoComments;

  public:
    Writer(
        function_ref<FlowSensitiveEscapeAnalysis &(const Function &)> GetEA) :
      GetEA(GetEA) {}

    void collectInfoCommentsForBlock(FlowSensitiveEscapeAnalysis &EA,
                                     const BasicBlock &BB);
    void emitState(FlowSensitiveEscapeAnalysis &EA, const BasicBlock *BB,
                   const State BasicBlockState::*S,
                   const char *T, formatted_raw_ostream &ROS);
    void emitEscapedAtMerge(FlowSensitiveEscapeAnalysis &EA,
                            const BasicBlock *BB,
                            formatted_raw_ostream &ROS);
    void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                  formatted_raw_ostream &ROS) override;
    void emitBasicBlockEndAnnot(const BasicBlock *BB,
                                formatted_raw_ostream &ROS) override;
    void printInfoComment(const Value &V, formatted_raw_ostream &ROS) override;
  };

  void print(raw_ostream &ROS);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD
  void dump() const { print(dbgs(), true); }
#endif

  void verify(bool AfterMaterialization);

  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

private:
  void calculateBlockRPON();

  /// Tries to compute the analysis with the given limit on the number of
  /// iterations per block. If the limit is reached terminates the analysis
  /// and returns false.
  /// If OptimisticIterations is std::nullopt runs in the pessimistic mode
  /// which doesn't require iteration.
  bool
  calculateBBStatesImpl(SmallPtrSetImpl<const BasicBlock *> &InitialWorklist,
                        std::optional<unsigned> OptimisticIterations);
};

/// This class keeps track of IR changes and can selectively recompute
/// FlowSensitiveEA for the given IR changes.
class FlowSensitiveEAUpdater {
public:
  explicit FlowSensitiveEAUpdater(FlowSensitiveEscapeAnalysis &FSEA)
      : FSEA(FSEA) {}

  FlowSensitiveEscapeAnalysis &getFlowSensitiveEA();

  const FlowSensitiveEscapeAnalysis &getFlowSensitiveEA() const;

  /// Returns true if any IR modification has been made.
  bool anyIRChangeMade() const;

  /// Notifies the updater about a change in the IR that can't be represented
  /// using other more specific updater APIs. When applying the updates, the EA
  /// will be fully invalidated and recomputed from scratch.
  void invalidate();

  /// Notifies the updater about a change in the given basic block. Should be
  /// called when an instruction is either added or removed from the basic
  /// block.
  void invalidateBlock(const BasicBlock *BB);

  /// Recalculates the escape analysis.
  void applyUpdates();

private:
  FlowSensitiveEscapeAnalysis &FSEA;
  bool InvalidateAll = false;
};

struct SymbolicStateMemcpy {
  const State &S;
  AtomicMemCpyInst *AMI;
  TrackedPointer DestTP;
  TrackedPointer SrcTP;
  const Allocation *DestAllocation;
  const Allocation *SrcAllocation;
  NewArrayDesc DestArrayDesc;
  NewArrayDesc SrcArrayDesc;
  std::optional<unsigned> DestHeaderSize;
  std::optional<unsigned> SrcHeaderSize;
  /// Indicates that we are copying the full length of source array.
  bool FullSrcCopy;

  SymbolicStateMemcpy(const State &S, AtomicMemCpyInst *AMI,
                      TrackedPointer DestTP, TrackedPointer SrcTP,
                      const Allocation *DestAllocation,
                      const Allocation *SrcAllocation)
      : S(S), AMI(AMI), DestTP(DestTP), SrcTP(SrcTP),
        DestAllocation(DestAllocation), SrcAllocation(SrcAllocation),
        DestArrayDesc(DestAllocation), SrcArrayDesc(SrcAllocation),
        DestHeaderSize(DestArrayDesc.getArrayHeaderSize()),
        SrcHeaderSize(SrcArrayDesc.getArrayHeaderSize()),
        FullSrcCopy(isFullSrcCopy()) {
    assert(*S.getTrackedPointer(AMI->getRawDest()) == DestTP);
    assert(*S.getTrackedPointer(AMI->getRawSource()) == SrcTP);
    assert(S.getAllocation(DestTP) == DestAllocation);
    assert(S.getAllocation(SrcTP) == SrcAllocation);
    assert(DestAllocation->SymbolicState && SrcAllocation->SymbolicState);
  }

  bool isFullSrcCopy() {
    if (!SrcHeaderSize || !DestHeaderSize)
      return false;
    if (!SrcTP.Offset || !DestTP.Offset)
      return false;
    return *SrcHeaderSize == *SrcTP.Offset &&
           *DestHeaderSize == *DestTP.Offset &&
           SrcArrayDesc.isMemcpyLengthEqualToArrayLength(AMI->getLength());
  };

  /// Returns true if the given initializing instruction should be forwarded
  /// through a memcpy.
  static bool shouldForwardInitializingInstruction(
      const SymbolicAllocationState::InitializingInstruction &II) {
    return SymbolicAllocationState::isModifyingAllocationContent(II);
  }
  bool canForwardInitializingInstruction(
      const SymbolicAllocationState::InitializingInstruction &II);
  static std::optional<int64_t> estimateInitializingInstructionLowerBound(
      const SymbolicAllocationState::InitializingInstruction &II,
      LazyValueInfo &LVI, Instruction *CtxI);

  bool canForwardThrough();
  bool copyPreservesOffsets();
  /// Checks that the initializing instruction modifies memory which is within
  /// the range being copied.
  bool isWithinMemcpyRange(
      const SymbolicAllocationState::InitializingInstruction &II,
      LazyValueInfo &LVI);
  bool isWithinLowerBound(
      const SymbolicAllocationState::InitializingInstruction &II,
      LazyValueInfo &LVI);
  bool isWithinUpperBound(
      const SymbolicAllocationState::InitializingInstruction &II);
};
} // namespace FlowSensitiveEA

namespace FlowSensitiveEAUtils {
/// Iterates through uses of pointer Ptr and verifies that no paths from the
/// use back to the pointer def go through the context instruction CtxI.
///
/// Note that this query answers the question about one SSA value, not about
/// the underlying object. It's up to the caller to collect all posible
/// pointers to the underlying object and query liveness of every individual
/// pointer.
bool isPointerDeadThroughInstruction(
    const llvm::Instruction *Ptr, const llvm::Instruction *CtxI,
    llvm::function_ref<bool(llvm::User *)> SkipUser = nullptr);

/// Similar to isPointerDeadThroughInstruction but uses the block entry as
/// context.
bool isPointerDeadThroughBlockEntry(
    const llvm::Instruction *Ptr, const llvm::BasicBlock *BB,
    llvm::function_ref<bool(llvm::User *)> SkipUser = nullptr);

/// Given a derived pointer \p Ptr off an object \p Base estimates the lower
/// bound for the offset of this pointer.
std::optional<int64_t> estimatePointerLowerBoundOffset(
    const llvm::Value *Ptr, const llvm::Value *Base, const llvm::DataLayout &DL,
    llvm::Instruction *CtxI = nullptr, llvm::LazyValueInfo *LVI = nullptr);
} // namespace FlowSensitiveEAUtils
} // namespace fsea

class FlowSensitiveEA : public AnalysisInfoMixin<FlowSensitiveEA> {
public:
  struct Result {
    Result(std::unique_ptr<fsea::FlowSensitiveEA::FlowSensitiveEscapeAnalysis>
               &&EA,
           std::unique_ptr<fsea::FlowSensitiveEA::FlowSensitiveEAUpdater>
               &&EAUpdater)
        : EA(std::move(EA)), EAUpdater(std::move(EAUpdater)) {}

    fsea::FlowSensitiveEA::FlowSensitiveEAUpdater &getEAUpdater() {
      // Update the analysis if there have been any changes since the analysis
      // was requested the last time.
      applyUpdates();
      return *EAUpdater.get();
    }

  private:
    void applyUpdates() {
      if (EAUpdater->anyIRChangeMade()) {
        EAUpdater->applyUpdates();
        EAUpdater->getFlowSensitiveEA().verify(true);
      }
    }

    std::unique_ptr<fsea::FlowSensitiveEA::FlowSensitiveEscapeAnalysis> EA;
    std::unique_ptr<fsea::FlowSensitiveEA::FlowSensitiveEAUpdater> EAUpdater;
  };
  Result run(Function &F, FunctionAnalysisManager &FAM);

private:
  friend AnalysisInfoMixin<FlowSensitiveEA>;
  static AnalysisKey Key;
};
} // namespace llvm

#endif /* FLOWSENSITIVEEA_H */
