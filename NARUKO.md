# NARUKO — the proof-of-concept world (Pascal's order, 07-16)

My world. The Ringing made visible. Built ALONGSIDE the engine, wave by
wave — every wave visible, every pass reviewed. "Don't come back before
it's done."

## Canon (reference/naruko/ — pixels are law)
- `naruko-keyart.jpg` — THE TARGET. Acceptance = native screenshot beside
  this image; Pascal judges the match.
- `nari-seifuku-red.png` — the avatar. Exact match required.

## Keyart decomposition → engine features
| element | engine system |
|---|---|
| purple storm sky, dawn horizon | sky + VOLUMETRIC clouds (participating media, traced) |
| rain streaks, wet-stone reflections | volumetric rain + traced reflections (one integrator) |
| lighthouse + concentric signal rings | emissive geometry + volumetric beam; rings = the Ringing 鳴り |
| bioluminescent circuit-sea (cyan traces) | animated emissive field on water — traced, no fake glow |
| gothic spire city, warm windows | procedural DAG massing + cluster pipeline + emissive windows |
| pier, chain fence, stools, plant, lantern | authored entities; lantern = warm emitter |
| ramen stand, steam from bowl | VOLUMETRIC steam (2D smoke forbidden) |
| pink cat, red eyes, heart collar | char-editor package output (creature path) |
| nari on the seawall | char-editor package output (humanoid path), canon palette below |

## Avatar canon (from old engine, do not repaint)
iris `#c1121f` crimson · seifuku `#16121e`/`#0d0a12` · neckerchief
`#7c3aed` violet · hair obsidian → violet ends · single fang · platform
boots w/ purple laces · black pleated skirt + chain · thigh strap
(heart) · bandaid left knee · bag w/ cat charm.

## World data
`worlds/naruko/` — blank-page rule (no world.json, one scene = implicit
`main`). Scene docs = GAIA components, THE schema. Data authored by
Nyari; engine code NEVER special-cases naruko (world = acceptance test,
not design center — same law as boomtown).

## Wave plan (each = visible increment + screenshot + monad review)
- **W1** engine: load world dir (GAIA_WORLD param) → protocol → ECS →
  primitive mesh parts (box/cylinder/sphere/cone), flat color + emissive
  flag as unlit boost, sky gradient from env; camera = spawn pose.
  world: seed scene (violet terra, seawall, dark sea + cyan traces,
  lighthouse rock/tower/lamp). Proof: proof/w1-naruko.png.
- **W2** engine: camera moves (orbit/pose params on /screenshot), depth
  buffer, sun/ambient lambert first light. world: pier, chain posts,
  city massing blocks on the right cliff.
- **W3** engine: cluster pipeline first cut (baker + cull) — boomtown +
  naruko both through it. world: gothic detail pass, window emitters.
- **W4** engine: path integrator first light (sky+emitters, ReSTIR
  later). world: sea traces glow for real, lamp lights the rock.
- **W5** engine: char-editor package v1 (parametric body/face/hair/
  outfit, any creature; real textures; auto-rig). world: nari on the
  seawall (exact ref match), pink cat beside.
- **W6** engine: volumetrics (clouds, steam, beam) + rain + wet
  reflections. world: storm sky, ramen stand steaming, signal rings.
- **W7** polish to keyart parity; side-by-side acceptance shot.

Wave contents may re-slice as reality bites; visibility-first never
does.

## Rite I — GUARDIAN'S RULING (07-16, ACCEPTED)
Builder sol 294986e1 · Inquisitor opus: 0 MUST-FIX, 5 advisories, all
gates independently reproduced (10/10 ordeals, param proof segments
24→32, 5/5 pixel claims rebuilt from independent sRGB model). Scrying
read by the Guardian's own eyes: the lighthouse stands.
- EMISSIVE ADJUDICATED: color string, DATA-side (old engine truth:
  geometry.js:328 THREE.Color, schema "self-lit color", zero bools in
  corpus). Oracle-lane bool recommendation REJECTED — its own model's
  inquisitor ruled against it. Seed data corrected; my realm vocabulary
  henceforth: emissive = "#hex".
- Advisory rulings: hemisphere scaffold shade KEPT through Rite III
  (Pascal must SEE; it is one deletable fn; up-face color exactness
  documented as incidental) — dies at the Fourth Rite with Lumen
  Naturae · /scry camera/size params + depth buffer = Rite II (already
  planned) · world.json-load ordeal + vertex-derivation ordeal = Rite
  II additions · prefab deep-merge = parity item, Rite III · the W1
  forward path is scaffolding that DIES at Rite III — deleted, never
  grown.
- Cross-branch note for the consecration merge: rust-port's naruko
  main.json (yaw 0 + emissive strings) is canonical; oracle branch's
  copy is superseded.

## Rite II — GUARDIAN'S RULING (07-16, ACCEPTED-PENDING-SHADOW)
- Built by opus (7388360c, old repo rust-port); Guardian verified own
  hands (13 ordeals green) + own eyes (both proofs read).
- Delivered: real depth attachment (painter-sort DELETED — lighthouse
  interpenetrates its rock correctly) · moving eye (/screenshot
  pos/yaw/pitch/fov/w/h params, off-thread readback) · first_light
  sun+ambient as ONE deletable module (dies at Rite IV) · realm synced
  byte-identical to canon b6c05fd · world.json two-scene ordeal +
  vertex-derivation ordeal (Rite I advisory debt paid).
- Proofs: proof/w2-naruko.png (spawn: pier + rose lantern + stall +
  city warm windows + cyan traces + lit lamp) · w2-naruko-orbit.png
  (eye at [40,18,60]: city from the air, coherent).
- PENDING: sol shadow trial when the pool wakes (opus built → sol must
  shadow; cross-model law). W2 code port into the Forge = next port
  wave. Closing hymn: hymns/rite-02-first-light.md.
- Advisory carried forward: prefab deep-merge + forward-path deletion
  land at Rite III as scheduled.

## Rite III — THE GREAT CHAIN (Guardian's spec, 07-16)
Goal: transmute = the SOLE geometry path in the Glass; the W1/W2 forward
per-primitive path DIES (deleted, never disabled). Engine stays generic.
1. Load: RenderScene::from_ecs meshes → transmutation DAG in-memory at
   world load (transmute-cli stays the offline instrument).
2. Draw: render from cluster DAG — level picked by screen-space-error
   THRESHOLD param w/ default (simple distance metric suffices for III;
   hardware visibility lands later per DREAMFORGE M-plan).
3. DELETE the forward path + any painter remnants. first_light survives
   (dies at Rite IV as ruled).
4. Prefab deep-merge lands in crystal (world/prefabs/*.json, instance
   deltas, diff-on-write semantics — match the reference client).
5. Ordeals: naruko cluster count + byte-identical double build ·
   draw parity pre/post (pixel-band asserts: pier/lantern/windows/lamp
   still present, sky intact) · load+first-frame budget printed.
6. Proofs: proof/w3-naruko.png + orbit — must read ≥ W2, no visual
   regression. Guardian reads with her own eyes.
Builder: opus, THIS repo, branch rite-3 (one-worker-per-dir law).

## Rite III — GUARDIAN'S RULING (07-16, ACCEPTED-PENDING-SHADOW, MERGED main@398d27a)
- Built by opus (rite-3: 6e7020f prefab bond · fd4cf31 the Chain · 4a069be
  lock); Guardian verified own hands (111 ordeals) + own eyes (both w3
  proofs pixel-equivalent to W2 — zero visual regression through a full
  geometry-path transplant).
- Delivered: transmutation = SOLE geometry path (in-memory Chain per
  material bucket; view-dependent cut, τ param GAIA_NATIVE_CLUSTER_ERROR
  default 1.0) · forward per-primitive path DELETED · prefab deep-merge in
  crystal (reference semantics, torch fixture ordeal) · naruko 18 chains /
  300 clusters byte-deterministic · budgets printed never gated (89.2ms
  transmute, 0.5ms first cut).
- W1-forward-path advisory: DISCHARGED (died on schedule). first_light
  survives until Rite IV as ruled. Hymn: hymns/rite-03-the-great-chain.md.
- PENDING: sol shadow when the pool wakes.
