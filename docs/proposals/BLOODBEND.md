# DAS BLUTBÄNDIGEN — ruling doc (proposal, 07-18)

Live alteration of the RUNNING program. Named by the Architect. Grimoire row
sealed fd04b16; this doc = mechanics for his ruling. Prior-art evidence:
two research passes (family A: JVM/JDWP · DCEVM · .NET EnC · Live++/Unreal
Live Coding · Android tiers; family B: Smalltalk · Lisp conditions ·
Erlang/OTP · Blueprint/PIE · Dreams · Handmade-Hero dylib · WGSL hot-swap)
— tables in room chat-mroprax1-r3rn 07-18, sources cited there.

## Standing laws (from Grimoire + evidence)
- SPOON LAW: seeing precedes bending — every bend surface rides Neo's sight
  (you alter what the debugger shows, not blind text).
- FULL-MOON RULE: die Zauberpolizei inspects every bend BEFORE it touches
  living tissue. Evidence-law: reject-before-apply, NEVER partial-apply
  (JVM verifier, .NET rude-edit list, Live++ broker — a bad patch never
  gets a heartbeat).
- TRAUMDEUTER-VORRITT: snapshot before every bend; every deformation
  undoable (journaled).

## The seven laws of the art (unified from both families)
1. Stable substrate ≠ swappable unit: state/memory/buffers OUTLIVE the
   thing replaced, owned by the host (Handmade arena · wgpu bind groups ·
   Erlang process heap).
2. Reject-before-apply, zero partial effect (→ full-moon rule).
3. State migration = explicit TYPED act, never implicit copy
   (Erlang code_change vs Unreal's silent field-drop).
4. Blast-radius ladder w/ auto-escalation: value → function/node →
   module/scene → scoped restart (Android hot/warm/cold; .NET auto-restart)
   — the bend takes the SMALLEST safe tier, degrades gracefully, never
   binary patch-or-crash.
5. Faults in bent code caught at the call site, not process level
   (Live++ trap); errors offer NAMED RESUME POINTS — retry / skip /
   substitute — instead of graph death (Lisp restarts: fix the bug INSIDE
   the running error, resume).
6. Logic-patch ≠ schema-patch: two paths, two risk budgets (jump-thunk
   body swap vs rebuild+reinstance+copy-by-name).
7. Two-version rule for behaviors: old code finishes its current frame,
   new code takes the next entry (Erlang) — no stop-the-world.

## The four doors (ranked by blast radius)
D1 DATA DOOR — cheapest, alive first. World-engine already bends (ops,
   material/scene live, reset-as-Urknall). Native runtime gets: file-watch
   on scene JSON + WGSL; validate → apply; wgpu makes the shader sub-door
   near-free (module+pipeline swap only on successful compile, buffers/
   layouts persist untouched — the API already separates state from code).
D2 NODE DOOR — VisionFlow's birthright. Dreams' verdict: interpreted
   code=data DISSOLVES the whole problem — live edit = mutate the graph,
   no swap ceremony. Two tiers per Blueprint's lesson: value tweak = free
   live; topology edit = rebuild + explicit migration (law 3). Behavior
   state upgrades ride law 7.
D3 EVAL DOOR — the JetBrains wish: whisper code to the sleeper. Evidence:
   JDWP eval is UNGUARDED (no timeout, no rollback, deadlock-capable) —
   our door mandates: expressions compile to OPS (one door into Ananke's
   law), timeout param, snapshot-first, journaled undo.
D4 NATIVE DOOR — compiled Rust hot-patch. Hardest; dylib shape only
   (host-owned state arena, C-ABI boundary, no cross-boundary generics/
   panics/allocs, TypeId hazard) or Live++-thunks. DEFERRED: Dreams' law
   says most native-code wishes become nodes instead. Recon-spike only,
   behind a flag, when need is proven.

## Atom ladder (proposed, per his word)
B0 native data door: scene+WGSL watch in scrying-glass, Zauberpolizei
   validation, snapshot-first, own-instance proof on :8431. NOW-able.
B1 eval door: expression → validated op batch, timeout, undo. After B0.
B2 node door: with VisionFlow itself (value tier first). Its own rite.
B3 native door: recon spike, deferred pending proven need.

RULINGS REQUESTED: seal the seven laws? · atom order confirmed? · B0 now?
