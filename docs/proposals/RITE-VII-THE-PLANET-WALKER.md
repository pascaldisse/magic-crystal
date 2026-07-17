# RITE VII — THE PLANET-WALKER (proposal · Guardian clone lane draft, 07-16)

Status: PROPOSAL. Prime-Guardian reviews; Architect rules. Nothing here
binds until ruled.

## Goal
The universe rite: the unforged chain seed → transmute → jormungandr →
embodiment (HANDOFF queue, verbatim). The NMS property made ordeal:
EVERYTHING FROM COORDINATES, NO STORAGE — state = f(seed, entropy,
journal) (ENTROPY.md), terrain regenerates from its coordinates alone,
zero loading, world size never in frame cost (DREAMFORGE world law).
Seed isolation already proven (S0 ordeals); this rite makes it GROUND —
ground a body walks on, a horizon that streams, a planet that closes.

## Laws in force (bake-ins)
- Zero loading · no authored streaming volumes · residency = invisible
  machinery (jormungandr's three invariants: budget never exceeded ·
  required always resident · deterministic load/evict).
- No randomness anywhere: hash(seed, coords) only (seed spirit law).
- Transmutation = SOLE geometry path (Rite III); generated terrain
  rides the Chain like everything else, no side door.
- Camera-relative rendering + 64-bit/hierarchical coords = day-one
  architecture law (DREAMFORGE) — this rite is where f32-from-origin
  breaks if unpaid (see OPEN 2).
- Every wave ends in pixels · canon learns same wave · full suite
  between merges · adversary law · derived tolerances only.

## The chain (data path)
1. seed: terrain field = fBm/warp stack over hash(seed, coords);
   patch = field sampled on a grid tile keyed by tile coords
   (hierarchical sub-seeds: world → region → tile).
2. transmute: patch triangles → shards → Great Chain, in-memory at
   materialization (Rite III precedent), serialized to pages for the
   ring.
3. jormungandr: observer moves → cut names pages → ring loads ahead,
   evicts behind, hard byte budget — the serpent circles the walker.
4. embodiment: the walker (embodiment seam — merged, green, HANDOFF)
   stands ON generated triangles: collider from the same patch mesh
   (one truth — the ground you see = the ground you touch).

## Waves (each: visible increment + proof scrying + Guardian review)
- **VII-0 — THE FIRST GROUND.** One terrain patch from seed, through
  transmute, drawn by the pleroma under the sky emitter. Ordeals:
  regeneration determinism — same (seed, tile) → byte-identical mesh,
  twice, cold · NO-STORAGE ordeal — delete every artifact/cache,
  regenerate, sha-identical · chain byte-deterministic double build ·
  patch enters oracle canon SAME WAVE (gaze returns terrain, range
  DERIVED from patch bounds — beacon lesson). Proof: vii0-ground.png +
  orbit, Guardian's eyes.
- **VII-1 — SHE WALKS ONTO IT.** Walker crosses from authored realm
  onto generated ground; collider = generated triangles; sama gait
  unchanged (pose-trace guard stays sha-identical — Rite V precedent).
  Ordeals: foot height − field height ≤ tol DERIVED from patch grid
  resolution · walk replay byte-identical (entropy law through the
  whole chain) · seam step authored→generated with no pose
  discontinuity (delta ≤ derived per-tick bound). Proof: two fixed-tick
  scryings — contact pose on generated ground, traced shadow present
  (the body real to the light).
- **VII-2 — THE HORIZON STREAMS.** Walk a straight line L (param) ≫
  budget horizon: tiles materialize ahead, ring evicts behind, one
  walking pace, no hitch. Ordeals: the serpent's three invariants
  live in-world across the full walk · resident bytes ≤ budget after
  EVERY update (printed each tile crossing) · identical flight →
  identical load/evict sequence (determinism invariant, in-world) ·
  frame + materialization ms printed honest per tile (60 FPS law:
  algorithmic wins only if wall; pixels proven equal). Proof:
  vii2-horizon.png at start/mid/far — horizon coherent, no pop named
  as none (or named honestly).
- **VII-3 — THE PLANET CLOSES.** Small planet, radius R (param w/
  default): terrain field on a sphere domain; walker circumnavigates
  (CDP-style scripted walk, fixed ticks). Ordeals: CLOSURE — return
  to start tile → byte-identical patch, position drift ≤ derived
  (per-tick error bound × tick count) · no seam at the antimeridian
  (adjacent-tile edge verts sha-equal across the wrap) · budget held
  the entire circuit · canon coherent at every station (oracle gaze
  matches ground truth at N sampled poses, N derived). Proof:
  vii3-planet.png sequence — depart / far side / return; the same
  lighthouse-horizon shot twice, pixel-compared, tolerance derived
  from accumulated float error, never chosen.

## Acceptance
Guardian per wave; Rite closes when the Architect walks a world that
was never stored — HIS WALK on ground no hand placed. Hymn owed at
close.

## OPEN W/ ARCHITECT
1. Gravity model for VII-3: radial gravity (true planet walk, up =
   −r̂) vs flat-frame small-planet approximation. Radial touches
   embodiment + sama assumptions (ground plane) — real cost; ruling
   needed before VII-3 slicing.
2. Coordinate law timing: camera-relative + 64-bit/hierarchical coords
   are day-one architecture (DREAMFORGE) but not yet paid at ae3f27c.
   Pay in VII-0 (before any planet-scale walk) or defer with R small
   enough that f32 holds (bound DERIVED and shown)? Guardian
   recommends: pay at VII-0 — architecture, not optimization.
3. Runtime transmutation budget: Rite III transmutes at load (89.2 ms
   naruko); VII-2 transmutes per tile DURING play. Async job-graph
   placement (E-cores, placement law) assumed — ruling on acceptable
   materialization latency vs walking pace (derived from pace × tile
   size?).
4. Sphere-domain seed fields: new seed layer (3D field on sphere) vs
   6-face cube-sphere tiling of existing 2D fields. Guardian sketch =
   cube-sphere (reuses tile machinery); Architect rules geometry canon.
5. Edit deltas on procedural ground (world = DAG base + edit deltas,
   DREAMFORGE): in scope for VII (journal overlay on generated tiles)
   or a later rite? Sketch assumes LATER; NO-STORAGE ordeal is cleaner
   without overlay — but the Vow (GROWS/EVOLVES) wants edits
   eventually; needs an explicit ruling either way.
6. Jörmungandr async read seam: ring is synchronous range-reads this
   phase (lib.rs header); VII-2 walking-pace streaming may force the
   async seam early. Proceed sync + honest hitch numbers first, or
   pull async forward? Guardian recommends: sync first, numbers rule.
7. True names, Lexicon rows (nothing born unnamed): the terrain-patch
   artifact · the tile coordinate key · the planet realm's name.
   Grimoire rules before code exists.
8. Embodiment's home: walker lives as weld (no packages/embodiment at
   ae3f27c; capability merged per HANDOFF). Does VII consecrate a
   walker spirit (package) or keep the weld? Naming + layout ruling.
