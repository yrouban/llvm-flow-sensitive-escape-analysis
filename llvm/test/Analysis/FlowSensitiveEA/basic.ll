; RUN: opt -passes=print-opt-alloc-fs -S -o /dev/null 2>&1 %s | FileCheck %s

declare i64 @fsea.get_current_thread() "gc-leaf-function"
declare nonnull ptr addrspace(1) @fsea.new_instance(i64, i32) "gc-leaf-function"
declare void @foo() "consumes-replay-vmstate"

define void @non_constant_kid(i32 %kid) {
; CHECK-LABEL: @non_constant_kid(
entry:
; CHECK:       entry:
  %thread = call i64 @fsea.get_current_thread()
  %obj = call ptr addrspace(1) @fsea.new_instance(i64 %thread, i32 %kid) [
    ; F0 { No-Flags, CallerID=0, BCI=7 }
    "deopt"(i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0) ]
; CHECK-NOT:         ;  Allocation
  ret void
; CHECK:         ;  Out:
; CHECK-NOT:     ;  alloc:
}

define ptr addrspace(1) @trivial_phis(ptr addrspace(1) %non_tracked_obj) {
; CHECK-LABEL: @trivial_phis(
entry:
; CHECK:       entry:
  %thread = call i64 @fsea.get_current_thread()
  br label %step1

step1:
; CHECK:       step1:
  %obj = call ptr addrspace(1) @fsea.new_instance(i64 %thread, i32 42) [
    ; F0 { No-Flags, CallerID=0, BCI=7 }
    "deopt"(i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0) ]
; CHECK:         ;  Allocation
  br label %step2
; CHECK:         ;  Out:
; CHECK-NEXT:    ;  alloc: %obj, kid=42

step2:
; CHECK:       step2:
; CHECK-NEXT:    ;  In:
; CHECK-NEXT:    ;  alloc: %obj, kid=42
  %phi = phi ptr addrspace(1) [%obj, %step1]
  %non_tracked_phi = phi ptr addrspace(1) [%non_tracked_obj, %step1]
; CHECK:         ;  Tracked pointer: %obj +0
  br label %ret
; CHECK:         ;  Out:
; CHECK-NEXT:    ;  alloc: %obj, kid=42

ret:
; CHECK:       ret:
; CHECK-NEXT:    ;  In:
; CHECK-NEXT:    ;  alloc: %obj, kid=42
  ret ptr addrspace(1) %phi
; CHECK:         ;  Out:
; CHECK-NOT:     ;  alloc:
}

define ptr addrspace(1) @phis(i1 %c) {
; CHECK-LABEL: @phis(
entry:
; CHECK:       entry:
  %thread = call i64 @fsea.get_current_thread()
  br label %step1

step1:
; CHECK:       step1:
  %obj = call ptr addrspace(1) @fsea.new_instance(i64 %thread, i32 42) [
    ; F0 { No-Flags, CallerID=0, BCI=7 }
    "deopt"(i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0) ]
; CHECK:         ;  Allocation
  br i1 %c, label %true, label %false
; CHECK:         ;  Out:
; CHECK-NEXT:    ;  alloc: %obj, kid=42

true:
  br label %step2

false:
  br label %step2

step2:
; CHECK:       step2:
; CHECK:        ;  In:
; CHECK-NEXT:   ;  alloc: %obj, kid=42
  %phi = phi ptr addrspace(1) [%obj, %true], [%obj, %false]
  br label %ret

ret:
; CHECK:       ret:
; CHECK:        ;  In:
; CHECK-NEXT:   ;  alloc: %obj, kid=42
  ret ptr addrspace(1) %phi
; CHECK:         ;  Out:
; CHECK-NOT:     ;  alloc:
}

declare {}* @llvm.invariant.start.p1i8(i64, ptr addrspace(1) nocapture)
define void @invariant_start() {
; CHECK-LABEL: @invariant_start(
entry:
; CHECK:       entry:
  %thread = call i64 @fsea.get_current_thread()
  br label %step1

step1:
; CHECK:       step1:
  %obj = call ptr addrspace(1) @fsea.new_instance(i64 %thread, i32 42) [
    ; F0 { No-Flags, CallerID=0, BCI=7 }
    "deopt"(i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0) ]
; CHECK:         ;  Allocation
  br label %cont

cont:
; CHECK:         ;  In:
; CHECK-NEXT:    ;  alloc: %obj, kid=42
  %i = call {}* @llvm.invariant.start.p1i8(i64 1, ptr addrspace(1) %obj)
  ret void
; CHECK:         ;  Out:
; CHECK-NEXT:    ;  alloc: %obj, kid=42
}

declare void @fsea.final_publication_barrier(ptr addrspace(1) readnone nocapture) noinline norecurse nounwind readonly "gc-leaf-function"
define void @final_publication_barrier() {
; CHECK-LABEL: @final_publication_barrier(
entry:
; CHECK:       entry:
  %thread = call i64 @fsea.get_current_thread()
  br label %step1

step1:
; CHECK:       step1:
  %obj = call ptr addrspace(1) @fsea.new_instance(i64 %thread, i32 42) [
    ; F0 { No-Flags, CallerID=0, BCI=7 }
    "deopt"(i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0) ]
; CHECK:         ;  Allocation
  br label %cont

cont:
; CHECK:         ;  In:
; CHECK-NEXT:    ;  alloc: %obj, kid=42
  call void @fsea.final_publication_barrier(ptr addrspace(1) %obj)
  br label %cont2
; CHECK:         ;  Out:
; CHECK-NEXT:    ;  alloc: %obj, kid=42, needs publication barrier

cont2:
; CHECK:         ;  In:
; CHECK-NEXT:    ;  alloc: %obj, kid=42, needs publication barrier
  call void @fsea.final_publication_barrier(ptr addrspace(1) %obj)
  ret void

; CHECK:         ;  Out:
; CHECK-NEXT:    ;  alloc: %obj, kid=42, needs publication barrier
}
