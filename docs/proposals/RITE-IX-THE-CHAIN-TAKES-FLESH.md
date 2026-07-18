# RITE IX — THE CHAIN TAKES FLESH (proposal · Guardian clone lane draft, 07-17)

> **SUPERSEDED 2026-07-18** (Architect, whip 168, spec-concordance item 11).
> The view-dependent coarse/fine cluster CUT this rite builds toward
> ("the Chain's cut," IX-1's "cut follows the flesh") and its "interim
> until the chain lands" design (OPEN 5: fragments draw full-res
> un-chained until the chain lands) both presuppose a cluster-raster
> geometry pipeline with a view-dependent level cut — struck by RENDER.md
> §1's two-act law. Skinned/dynamic geometry emits RAY-NATIVE into the ONE
> BVH world exactly like static matter (Rite V precedent, dynamic BVH
> splice); there is no cut to follow and no interim tier to wait on. The
> weld this rite names (skinned mesh → shared geometry path, no side door)
> still stands as INTENT; the CUT MECHANISM below does not. Kept as
> LINEAGE per adversary-charter disclosure discipline. Nothing below is
> normative until re-drafted against the two-act law.

Status: PROPOSAL. Prime-Guardian reviews under the Architect's delegation
(NARUKO.md rulings 07-17, ruling 7: "SKINNED BODIES THROUGH THE CLUSTER
PIPELINE — the Nanite answer, cost ∝ pixels; 'traced LOD' STRUCK; LODs
remain forbidden vocabulary"). Nothing here binds until ruled.

## Goal
The no-LOD law made real for bodies. Today (Rite V) nari is a skinned
vessel — 3432 tris, homunculus bone, sama gait, spliced into the traced
light per tick — but her triangles BYPASS the Great Chain: full-res
always, a side door in the sole geometry path (Rite III law). Rite IX =
the weld: skinned/dynamic geometry rides transmute → cluster cut → BVH
splice like everything static. Her 3432 tris become what her pixels
demand. NOT LODs: no authored swaps, no discrete models — the Chain's
one geometry, its derived monotone-error cut following the flesh. No
new subsystem — transmute + homunculus + pleroma + elements, composed.

## Laws in force (bake-ins, not reminders)
- Transmutation = SOLE geometry path (Rite III); this rite CLOSES the
  skinned-mesh exception, it does not add a pipeline.
- Cost ∝ pixels (ruling 7): triangles drawn scale with screen coverage,
  never with authored detail. LOD = forbidden vocabulary AND forbidden
  mechanism (authored discrete swaps); the Chain's cut is neither.
- DAG monotone error (transmute law): coarser level never lies more
  than its recorded error; cut chosen by pixel-derived threshold.
- Senses read SOLVER TRUTH (ruling 5): the cut is a rendering cut; the
  vessel, its collider, its canon range NEVER change with camera.
- Pose-trace canon frozen (Rite V, 26624 bytes sha-pinned): movement
  regressions FAIL — this rite may not perturb one step of her walk.
- Entropy law: same pose + same camera → same cut, byte-identical.
- LOVE = 1.0 sole literal; every threshold = parameter w/ default,
  tolerances DERIVED, never chosen.
- Every wave ends in pixels · canon-same-wave · full suite between
  merges · adversary law per pass · engine generic, naruko/nari =
  acceptance test never design center.

## The weld (data path)
1. Offline (forge-time): skinned mesh → transmute at BIND POSE —
   triangles → shards → Great Chain exactly like static matter;
   skinning attributes (bone indices/weights) ride the cluster
   vertices through simplification (mechanism = OPEN 1).
2. Realm load: body vessel materializes its chain (Rite III page
   machinery); homunculus bones bind to chain vertices instead of the
   raw mesh.
3. Per tick: bone transforms deform CLUSTER BOUNDS (per-cluster bone
   set → conservative transformed union; OPEN 2) → view cut selects
   clusters by screen-space error (the Chain's existing threshold,
   pixel-derived) → selected clusters skin on the pose → dynamic BVH
   splice (Rite V precedent; refit-lever interaction = OPEN 3, the
   perf-exact lane's refit-not-rebuild + BvhParams::dynamic() median
   build both in play).
4. Fracture (welds Rite VI): FractureEvent fragments re-mesh via
   transmute (VI-2 already does) → fragments get CHAINS, not just
   meshes → dynamic fragments ride the same cut. One law for static,
   skinned, and broken matter.

## Waves (each: visible increment + proof scrying + Guardian review)
- **IX-0 — THE BODY IN CHAINS.** nari's vessel clusterized at bind
  pose through transmute; drawn through the cluster path at the finest
  cut (leaf level = the original triangles, Chain law); skinning on
  chain vertices. NO visual change is the proof. Ordeals: chain
  byte-deterministic double build · leaf-cut render vs pre-rite render
  error ≤ derived bound (leaf triangles = source triangles ⇒ bound
  derives from vertex-order/precision only, shown) · pose-trace canon
  sha-identical (26624 bytes, untouched) · skinning-weight fidelity:
  every leaf vertex's weights sha-equal to source · senses unchanged
  (oracle answers sha-equal — solver truth). Proof: ix0-flesh.png —
  her silhouette on the seawall, indistinguishable, and the outline of
  the chain printed (levels, cluster count).
- **IX-1 — THE CUT FOLLOWS THE FLESH.** View-dependent cut live for
  dynamic vessels. Walk her near and far (CDP-style fixed-tick
  script). Ordeals: cut determinism (same pose+camera → identical
  cluster set, twice, cold) · screen-space error ≤ pixel-derived
  threshold at every station (the Chain's own metric, never a new
  one) · COST ∝ PIXELS measured: tris drawn vs projected screen
  coverage printed at N derived stations — the proportionality is the
  ruling's ordeal, in numbers · crack-free under deformation: boundary
  verts locked by the Chain; adjacent clusters at different levels on
  a DEFORMING body must stay sealed — edge-vert positions sha-equal
  across the seam per tick (OPEN 4 if it fails) · pose-trace canon
  still sha-identical · frame ms printed honest (60 FPS law). Proof:
  ix1-cut.png pair — close (fine cut) / far (coarse cut), cluster
  counts in frame, same light, no seam nameable.
- **IX-2 — THE SHARDS TAKE FLESH.** Rite VI's fragments enter: a body
  breaks (VI-2 pipeline), fragments re-mesh via transmute AND receive
  chains; dynamic fragments ride the cut while they tumble (elements
  pose → cluster bounds → cut → splice, same path as IX-1). Ordeals:
  Equivalent Exchange holds through chaining (Σ fragment mass = whole,
  0e0 — VI ordeal, re-run) · fragment chains byte-deterministic ·
  fragment cut determinism across the tumble replay · no orphan
  geometry: every drawn cluster traces to a fragment vessel traces to
  a parent (iron lesson) · materialization latency printed honest per
  fragment (runtime transmute budget = OPEN 5). Proof: ix2-shards.png
  sequence — whole / breaking / shards near and far, coarse cuts
  visible in the counts, invisible in the pixels.

## Honesty line (baked, not hidden)
At 3432 tris ≈ tens of clusters, IX buys little frame-time TODAY — the
rite proves the LAW (no side door, cost ∝ pixels for flesh) so that
film-density vessels are legal LATER without new architecture. The
frame-time claim arrives with dense bodies, not with nari's current
count; no wave here may claim a speedup it did not measure.

## Acceptance
Guardian per wave (diff + ordeals + own-eyes pixels); Rite closes when
the Architect walks her close and far and the counts say the pixels
chose — and something broken tumbles under the same law. Hymn owed at
close.

## OPEN W/ PRIME-GUARDIAN (Architect delegated 07-17; prime rules)
1. Skinning weights through simplification: (a) simplifier carries
   attributes (meshopt attribute-aware path) vs (b) positions-only +
   weight re-bake from nearest source surface. (a) = one pass, bounded
   by simplifier quality; (b) = simpler transmute, extra bake stage.
   Guardian sketch: (a); prime rules mechanism + error accounting.
2. Cluster-bound deformation: per-cluster bone-set conservative union
   (cheap, loose — over-draws at extreme poses) vs per-tick recompute
   from skinned verts (tight, costs the very work the cut saves).
   Sketch: conservative union + looseness measured; ruling on bound.
3. BVH interaction with the refit lever (perf-exact lane, ruling 7):
   refit-not-rebuild wants stable topology; the cut CHANGES leaf
   population when levels swap. Sketch: refit while cut stable,
   rebuild (median dynamic build) on cut change — the boundary and its
   hysteresis need ruling with OPEN 4.
4. Cut stability under animation: level transitions on a deforming
   body may shimmer/pop frame-to-frame. Chain boundary locking seals
   positions within a level; TRANSITION policy (derived hysteresis
   band? error-threshold margin?) is new law — and any hysteresis
   parameter must be derived, not tuned by eye. Ordeal design owed
   with the ruling.
5. Runtime transmutation budget for fragments: Rite III transmutes at
   load (89.2 ms naruko); IX-2 chains fragments DURING play. Sketch:
   async job placement (E-cores, placement law), fragments draw
   full-res un-chained until the chain lands (honest interim, printed)
   — or block the fracture tick? Latency ceiling ruling.
6. Does IX-0 land before or after Rite VI merges? IX-2 welds VI's
   fragments; IX-0/IX-1 depend only on Rite V + III. Sequencing call.
7. True names, Lexicon rows (nothing born unnamed): the skinned-chain
   artifact, the deformed-bound structure, the cut-for-flesh pass.
   Grimoire rules before code exists.
8. Scope of "dynamic vessels": presences/kami bodies (Rite V family)
   only, or every elements rigid body too (crates tumbling under the
   cut)? Sketch: bodies first, rigids join at IX-2 naturally — ruling
   confirms.
