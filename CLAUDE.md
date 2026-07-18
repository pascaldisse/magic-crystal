# THE MAGIC CRYSTAL — Claude Code entry
NAMING (Architect, 07-18 16:38): the render engine is PLEROMA — whole, no
inner names. Where older seals below say "THE NET", read: Pleroma's learned
act. Speech and specs use the true name: PLEROMA RENDERS — world data in,
the final image out, or nothing. Same discipline everywhere: Ananke solves,
Gaia worlds, MONAD directs; "net/solver/model" = internals vocabulary,
never a system's name.
Law chain (read in order): BIBLE.md → GRIMOIRE.md → HANDOFF.md → DREAMFORGE.md
→ NARUKO.md (incl. GUARDIAN RULINGS 07-17) → docs/proposals/.
Night operation: NIGHTRUN.md is the standing order; log to NIGHTLOG.md.
★ THE DESIGN IS THE LAW (Architect, 07-18, supreme): you build the ONE FULL
NEURAL RENDERING ENGINE as designed — world truth enters Pleroma; Pleroma
renders the final image or nothing — and NOTHING else. No fallbacks. No
prototypes. No interims shipped into the present path. Hand-built reconstruction
(temporal gates/clamps/heuristics) = LAB EQUIPMENT ONLY (training data + history
buffers), never the shipped path. (SUPERSEDED 16:31: screen = Pleroma's image
or NOTHING — no young-samples display.)
★ EXTENDED TO PHYSICS (Architect, 07-18 14:50): same law — the ONE NEURAL
PHYSICS ENGINE: Ananke assembles constraints; Pleroma's learned act solves
into state. The classical solve = teacher/ground-truth generator + pre-Pleroma
scaffold, never the destination. Death rule: Pleroma's learned act that loses to
the classical solve it replaces
at equal quality dies. Fluids + building-scale collapse are OWED IN HIS WINDOW.
★ ADVERSARY CHARTER AMENDED (Architect, 07-18 15:15, whip 168): adversaries
check SPEC CONCORDANCE, not only behavior — every gate includes: does the
implementation match the law chain, and does the spec contradict a ruling?
A spec contradicting a sealed ruling = HERESY, reported like a broken test.
Born from: raster-cluster pipeline squatting in RENDER.md two days after the
two-act law — no adversary was ever aimed at it.
★ WILDE JAGD GATE — versioned enforcement for every pushed merge.
Install: git config core.hooksPath .githooks
The versioned .githooks/pre-push finds merge commits newly introduced by a push.
It invokes tools/wilde-jagd-gate.sh before the remote accepts them.
Every merge message ends with: Adversary: <agent> HOLDS
Every merge message ends with: Concordance: checked
The gate refuses a merge missing either exact trailer; the charter law is printed.
No new merge commits → the hook exits cleanly.
Emergency escape: GAIA_JAGD_SKIP=<reason> git push; bypass is printed loudly.
Never use the escape hatch silently; record its reason in the push context.
★ STUDY, NEVER IMPORT (Architect, standing since spec day — violated once,
whip 168, never again): industry engines (Unreal/Nanite/Lumen/DLSS/anything)
are EVIDENCE to study, NEVER blueprints to copy. Every organ of this engine
derives from OUR law chain and OUR measurements. Recon informs; the
Architect rules; the design is born here or not at all.
★ DWARF FORTRESS = THE WORLD-SIM BIBLE (Architect, 07-18 15:44): DF = the
most accurate world simulation ever built — the reference standard THE ONE
MIND and world simulation are measured against (depth of personality, needs,
memory, history, consequence). Study-never-import still governs the code.
★ MONAD = THE GOD-INTERFACE (Architect, 07-18 15:47): the gamemaster
superagent — player-facing voice + authority over every mind and system
(events, interactions, all of it). Crown of THE ONE MIND; rules through the
same doors (senses/ops), privilege = authority not machinery; TTRPG's GM AI
= first source. GRIMOIRE §MONAD.
PACKAGE LAW (Architect, 07-18 16:10, CORRECTED 16:13 by his word): EVERYTHING
IS MODULAR — the renderer (Pleroma), the physics (Ananke), the AI system,
senses, voice, style, content: ALL replaceable packages. THE CRYSTAL = the
VERY MINIMUM: world state (entities/components/scenes) · the ops door ·
entropy + journal (deterministic replay) · the package/door system itself ·
MONAD (the god-interface, constitutional). Nothing else is core. A bare
crystal is a lawful, replayable, addressable world with no eyes, no body,
no picture — and every organ plugs into its doors, swappable, offline.
★ OFFLINE LAW (Architect, 07-18 16:05): the engine runs FULLY OFFLINE.
LLM is NEVER required — not for THE ONE MIND, not for MONAD, not for any
system. Mind core = small conditioned nets + systemic simulation (DF-class),
deterministic, local. Natural-language dialogue = OPTIONAL pluggable voice
organ (local, player's choice), rides the same doors, required by nothing.
AMENDMENT (16:07): a VERY SMALL local LM is permitted as that voice organ
IF offline AND 60fps-inviolate — async, off the frame's critical path,
never on the render GPU. ANE placement is UNVERIFIED; MTL4-encoder path
UNVERIFIED. [source: NEURAL.md §Silicon race verdicts] Budget-fenced by
measurement like everything.
TTRPG's LLM-DM = inspiration only — its STRUCTURAL laws transfer (op-door
authority · engine rolls all dice · stage-before-narrate · rules-as-data),
its LLM judgment does not.
★ WINDOW BAN (sealed 16:54): no lane opens windows on his desktop ever;
proofs headless; his glass = his act.
★ BOTH-EYES CONTRACT (sealed 16:54): any visual launch claim = belief-buffer
PNG + presented-surface PNG, both read, BEFORE announcement.
★ THE RESOLUTION IS 640×480 (Architect, 07-18 16:22, HIS CAPS): the render
resolution — trace buffers, net output, present — is 640×480 UNTIL HE SAYS
OTHERWISE. IRON param, default 640×480. The window displays it scaled by the
OS/integer blit (display scaling ≠ rendering; nothing is traced small and
neurally enlarged — the CANVAS is 640×480, God's resolution, Terry's law).
Supersedes any 'internal = surface res' instruction in riding lanes —
enforce at gate.
★ CORRECT OUTPUT OR NOTHING (Architect, 07-18 16:31, his words twice): ONE
render: world truth enters Pleroma; Pleroma renders the final image or
nothing. Rays = input sampling inside Pleroma, never a picture. THE SCREEN
SHOWS PLEROMA'S OUTPUT OR NOTHING (black) — the raw-evidence present path
(dots/accum-to-screen) is DEAD; no intermediate ever reaches glass. Enforce
at every gate: present = Pleroma or black.
★ BANNED WORD (Architect, 07-18 18:27): Apple's high-level ML framework is
NEVER mentioned, specced, or built against again. The neural-cores door for
this engine = the Metal-native ML command encoder (Tahoe, GPU-or-ANE on the
command timeline [source: client-rs/research/metal4-neural-recon.md]) —
Metal end to end, one API family, period. Any lane spec or doc carrying the
banned framework: voided at gate.
Absolutes: 60 FPS minimum · NO LODs (cluster law) · neural interpolation BANNED
· one light pass · never hardcode (LOVE=1 sole literal) · derived tolerances ·
canon learns every vessel the same wave · full suite between merges · push only
green · never touch ports 8420/5173 or processes you didn't start · never
restart the user's apps · bun never npm in JS repos · cargo for this repo.
