# BeamNG.tech Technical Paper (Maul/Mueller/Enkler/Pigova/Fischer/Stamatogiannakis, 2021) — NUMBERS extract (sonnet, 07-16)

Sources (both fetched clean, retrieved 2026-07-16):
- Whitepaper PDF: https://beamng.tech/blog/2021-06-21-beamng-tech-whitepaper/bng_technical_paper.pdf
  (5 pages, 528KB, pdftotext clean — the PDF flagged UNVERIFIED/failed
  extraction in softbody-recon.md's prior pass is now RESOLVED, see below).
- Official public physics primer: https://www.beamng.com/game/about/physics/
- JBeam section-usage stats: https://documentation.beamng.com/modding/vehicle/sections/
Cross-ref: softbody-recon.md (prior pass, code-mined from Rigs of Rods
fef9f25c + public docs) — this file adds the STAFF-AUTHORED whitepaper
numbers on top; no conflicts found, several UNVERIFIED items resolved.

## Solver rate — CONFIRMED, resolves prior UNVERIFIED
- Whitepaper verbatim: "BeamNG.tech is capable of running the simulation in
  real-time and faster. The base real-time simulation frequency of BeamNG
  physicscore is 2Khz and it is fixed for the entertainment version
  BeamNG.drive. In BeamNG.tech, the simulation frequency can be adjusted to
  be higher or lower, depending on the user's needs."
  ⇒ 2000 Hz / 0.0005s fixed step is STAFF-CONFIRMED for BeamNG.drive
  (matches the RoR-code-derived PHYSICS_DT=0.0005s in softbody-recon.md
  exactly). BeamNG.tech (research variant) allows changing this rate.

## Node/beam model — matches RoR ancestor, staff phrasing
- Nodes: point masses, position updated from net force, no inherent shape.
- Beams: connect exactly 2 nodes, springs with stiffness+damping, ELASTIC
  by default (return to rest length) OR PLASTIC (deformation permanently
  changes rest length past a deform threshold, and can BREAK past a break
  threshold). Broken beams stop influencing their nodes ("as if snapped in
  half"). Deform/break thresholds are independent — sponge-like material =
  low deform threshold + high break threshold (deforms easily, hard to
  break); glass-like = deform threshold set HIGHER than break threshold
  (breaks before it visibly deforms).
- Example spring/damp constants published on the public physics page
  (typical values table, verbatim):
  - Suspension springs: beamSpring 40000, beamDamp 0
  - Suspension dampers: beamSpring 0, beamDamp 4500
  - Structural vehicle components (e.g. suspension arms): beamSpring
    8,000,000, beamDamp 125
  - Steering rack (needs max stiffness): beamSpring 14,001,000, beamDamp 250
  (Compare softbody-recon.md's RoR-code defaults: k=9e6, d=12e3, break=1e6,
  deform=4e5 — same order of magnitude for structural beams, NOT identical
  constants; RoR and shipped BeamNG.drive vehicles are different tunings
  of the same model, as expected.)
- Node-weight instability rule (staff-stated, qualitative, no formula
  given): if beam stiffness is too high relative to node mass the
  structure vibrates/explodes; fix = heavier node OR softer beam. Same for
  beamDamp set too high. UNVERIFIED: no explicit CFL/stability inequality
  published (matches the RoR-code finding that stiffness binds timestep,
  but whitepaper gives no formula, just the qualitative warning + GIF demo
  contrasting nodeWeight 6kg [unstable] vs 7kg [stable] on an unspecified
  structure — NOT a general constant, model-specific).

## Vehicle content scale (BeamNG.tech package, whitepaper §II.B.1-2)
- 28 vehicles + 7 trailer models from 13 different brands (BeamNG.tech
  package as of the 2021 paper).
- 19 levels total, of which 3 provide urban environments.
- Vehicle physical model = "physics skeleton" (node/beam graph) +
  SEPARATE powertrain simulation (no physical properties of its own —
  pure math model converting driver input → torque applied TO the
  skeleton). 3 example powertrain topologies diagrammed: ICE w/ torque
  converter+gearbox+transfer case+front/rear diff (ETK800 854t); ICE w/
  clutch+gearbox+dual differential split (Gravil T65); dual-electric-motor
  w/ independent front/rear diffs (Tograc qE) — demonstrating plug-and-play
  powertrain assembly (ICE / EV / 4WD truck all fit the same framework).

## JBeam section-count census (official docs, v0.38.5.0, ALL vanilla content combined — NOT per-vehicle)
Verbatim counts of how many times each section appears across every JBeam
part shipped in the base game (aggregate, not a single car):
information 20043 · slotType 20042 · flexbodies 15606 · beams 10985 ·
nodes 10471 · triangles 5663 · slots 5269 · pressureWheels 3771 ·
variables 2856 · props 2798 · mainEngine 1597 · controller 1517 ·
slots2 1305 · powertrain 1122 · sounds 1104 · torsionbars 971 ·
slidenodes 952 · rails 921 · components 800 · glowMap 778 ·
vehicleController 710 (list truncated at "triggers2" in source page).
⇒ This RESOLVES the direction of the prior "~400 nodes/4000 beams per
vehicle" marketing figure in softbody-recon.md as still UNVERIFIED per
single vehicle — the only hard official count found is the AGGREGATE
across all shipped parts (10471 nodes / 10985 beams total, version
0.38.5.0), not a per-vehicle figure. Per-vehicle node/beam counts remain
UNVERIFIED — would require parsing individual .jbeam files from an
installed game copy (not available in this environment).

## Sensors / data acquisition (BeamNG.tech, ADAS use-case — for completeness, not core physics)
- Sensor list: Camera (RGB + depth + pixel-wise class/instance annotation +
  bounding boxes), Lidar, IMU, Ultrasonic, Electrics (vehicle internal
  state: clutch/engine/turn-signal etc.), State (GPS-like: position +
  orientation). Also exposes g-force and damage data directly (hard to get
  from real-world testing).
- BeamNGpy = the Python control/sensor interface; BeamNG ROS Integration
  package also exists, depends on BeamNGpy.

## Two cited real-world applications (whitepaper §IV, for engine-relevance context only)
- AsFault (Gambi et al. 2019): search-based procedural road-network
  generation + genetic-algorithm mutation to stress-test lane-keeping AI;
  fitness = distance from lane center over a predefined path.
- HDI damage-evaluation pilot: synthetic crash image dataset (5 crash
  types: t-bone, rear-end, frontal, pole, no-crash) + per-part damage
  score (max deformation observed, % deformed beams, % broken beams) used
  to train a CV system to classify vehicle part damage from images; noted
  challenge = domain gap between synthetic and real crash photos, NOT
  solved in the paper (cited as future/ongoing work, ref [3]).

## Numbers NOT found / UNVERIFIED after this pass
- Per-vehicle node count and beam count (only aggregate content-wide
  counts recovered, see JBeam census above).
- CFL-style stability formula relating beamSpring/nodeWeight/timestep
  (only qualitative "too stiff for the mass → vibrates/explodes" warning).
- Any GPU/multithreading numbers (thread-per-actor claim in
  softbody-recon.md remains "community folklore, not staff-confirmed" —
  this whitepaper does not mention threading/parallelism at all).
