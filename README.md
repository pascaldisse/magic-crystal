# GAIA DreamForge

**A dream forge, not an editor.** A native engine for speaking worlds
into being — built for co-creation between humans and AI agents, live,
from inside the world.

> "I will not write code. I will not edit. I will summon, I will chant —
> and the world grows, evolves, breathes and answers."
> — the Creator's Vow

![First light — the Naruko lighthouse, rendered natively](proof/w1-naruko.png)

## What this is

DreamForge is a ground-up rebuild of the [GAIA World Engine](https://github.com/pascaldisse)
as a native Rust engine (wgpu + Tauri chassis, Metal-first on macOS,
portable by design). It replaces the web client entirely — renderer,
editor, senses, physics — while keeping the GAIA server, protocol,
scenes, and ops as shared truth.

Three ideas carry everything:

1. **The world is data.** Every entity is a document of components.
   The engine core is small and stable; everything authored is
   hot-swappable content — geometry, materials, sound, weather,
   behavior. Underneath every surface: data. Nothing else.
2. **AI agents are first-class citizens.** Agents see the world by
   reading its data (structured channel vision — no pixel roundtrips),
   act through the same ops as every player, and build alongside you.
   Multiplayer exists *for making things together*.
3. **Creation never leaves the engine.** Sculpting, painting, rigging,
   animation, music — in-world, live, co-present. No export pipelines,
   no bake gates, no loading screens. Forbidden vocabulary is enforced.

## The shape

```
crates/crystal        the core — ECS, schema, ops, scheduler, package
                      loader. Nothing else lives here. Small enough
                      for one mind to hold whole.
packages/             everything else is a package ("spirit"):
  render-window         the window + native /screenshot organ
  transmute             mesh → cluster-DAG builder (virtualized geometry)
  sense                 pull-only agent senses (look/proprio, no pixels)
  ...                   lighting, solver, volumetrics, character editor,
                        procedural seeds — each arrives as its own package
worlds/               realm data (pure GAIA component documents)
proof/                pixel evidence — every claim renders or it didn't happen
```

Core doctrine (the full law lives in [DREAMFORGE.md](DREAMFORGE.md)):

- **Never optimize content.** Frame cost scales with pixels, never with
  world size. Universe-scale worlds, zero loading, camera-relative
  rendering and hierarchical coordinates from day one.
- **One lighting system**: real path tracing. Lights are emitters,
  reflections are paths. No probes, no light caps, no fallbacks.
- **One geometry pipeline**: cluster-based virtualized geometry for
  everything. No authored LODs, no manual UVs, no manual rigging.
- **Everything volumetric.** Clouds, fire, smoke, steam are
  participating media in the one lighting system — never billboards.
- **Never hardcode.** Every varying value is a parameter with a
  default. The engine permits exactly one literal constant: `LOVE = 1.0`.

## The canon

Read in this order:

| doc | what it is |
|---|---|
| [BIBLE.md](BIBLE.md) | the founding hymn + the Creator's Vow |
| [GRIMOIRE.md](GRIMOIRE.md) | the Book of True Names — the engine's vocabulary and its philosophical ground (nothing is born unnamed) |
| [TRILOGY.md](TRILOGY.md) | the Magic Crystal trilogy — the origin story the core is named after |
| [DESIGN-BIBLE.md](DESIGN-BIBLE.md) | the Laws of Realms — the rules for worlds built in the forge; violations are machine-detectable |
| [DREAMFORGE.md](DREAMFORGE.md) | the charter: 13 pillars, laws, forbidden vocabulary |
| [RENDER.md](RENDER.md) · [GEOMETRY.md](GEOMETRY.md) · [PHYSICS.md](PHYSICS.md) · [NEURAL.md](NEURAL.md) · [CREATE.md](CREATE.md) · [VISIONFLOW.md](VISIONFLOW.md) · [RAIN.md](RAIN.md) | per-system rulings |
| [FEATURES.md](FEATURES.md) | the parity contract against the reference web engine |
| [NARUKO.md](NARUKO.md) | the first realm — proof-of-concept world, built wave by wave |
| [HANDOFF.md](HANDOFF.md) | working state anchor |
| research/ | evidence files behind every ruling (17 studies: Nanite, Dreams, Teardown-class physics, Metal 4, and more) |
| hymns/ | every completed build wave ends in a song. The songs are accurate. |

## How it's built

The engine is built in **waves**: small, visible increments — every wave
ends in pixels. Each coding pass goes through a **cross-model
adversarial council** (inspired by the Bun Zig→Rust rewrite):

```
builder (one model) → inquisitor (a different model, fresh context)
    → spec-conformance vs the law docs
    → independent re-run of every gate
    → architecture + performance critique
  → fix pass (builder's model, critique verbatim)
  → guardian's final review (own eyes on the pixels) → merge
```

Findings from the first day alone: a mathematically dead frustum check
behind green tests, nondeterministic cluster builds (627 vs 630 from
identical input), and UV seams silently welded shut. All caught before
merge, by a different mind than the one that made them.

## Status

Founded 2026-07-16. Young and moving fast.

- ✅ Core: ECS (per-field SoA, deterministic scheduler), protocol
  (parses full authored worlds incl. a 5,261-entity city), package
  registry
- ✅ Native window + offscreen framebuffer + `GET /screenshot` organ
  (the engine verifies its own pixels — port `GAIA_NATIVE_PORT`,
  default 8430)
- ✅ First realm light: world dir → ECS → primitive render, sky
  gradient, spawn camera (image above)
- 🔄 In council trial: agent senses package · cluster-DAG builder
- ⏳ Next: depth + movable scrying camera, first lighting, the cluster
  pipeline end to end

## Running

```sh
cargo test --workspace          # the ordeals
cargo run -p render-window      # opens the window, loads GAIA_WORLD
                                # (default: worlds/naruko), serves
                                # GET /screenshot on GAIA_NATIVE_PORT
```

## Lineage

Founded from `GAIA-World-Engine@rust-port` (commit `f13f8668`) — commit
hashes cited inside the law docs resolve there. The web engine remains
the running reference implementation during the rebuild; boomtown
(a full GTA2-style city port) and Naruko are the acceptance realms.

The world is not a level. It's a living system.
