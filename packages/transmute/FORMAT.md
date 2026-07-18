# The Great Chain — DAG binary format (.cbdg)

§ scope → on-disk serialization of a transmuted cluster Great Chain. Producer:
`serialize`, consumers: `deserialize` (full) / `read_root` / `read_page` /
`read_directory` (residency-scale). See `src/serialize.rs`. Versioned; header
checked on read.

## SUPERSEDED — level/LOD runtime serialization
`levels`, group error/bounds, page `level`, and `roots` preserve bake lineage
only. The live BVH consumes level-0 loss-free leaves in full; no reader may use
this metadata for camera/projection detail selection. Retained page machinery is
teacher/lineage material for RENDER.md §1's future ray-footprint residency slot.

## Why CHUNKED (finding 4)
A whole-file `bincode(Dag)` blob forces the ENTIRE chain resident to read
anything — a non-starter at universe scale. v2 is header + independently
range-readable pages + a bounded directory, so bake lineage and future
ray-footprint residency research can range-read it without decoding all geometry.

## Byte layout
```
[0..4)   magic          = b"CBDG"                (ASCII, fixed)
[4..6)   format_version  : u16 LE                 (current = 3)
[6..8)   flags           : u16 LE                 (0)
[8..16)  dir_offset      : u64 LE                 (byte offset of the directory)
[16..20) dir_len         : u32 LE                 (directory length in bytes)
[20..24) root_page       : u32 LE                 (first root page id; u32::MAX = none)
[24..dir_offset)         : pages region           (concatenated Page chunks)
[dir_offset..+dir_len)   : directory              (bincode(Directory))
```
- header is fixed `HEADER_LEN = 24` and stable forever; a version mismatch is
  rejected loudly (`UnsupportedVersion`), never mis-decoded.
- each **page** is a standalone `bincode(Page)` — decodable from its byte range
  alone, no other page or the directory required.
- v3 supersedes v2: mandatory METIS with entropy-derived seeds replaces the
  former fixed-seed/fallback bake semantics; v2 readers reject it loudly.
- the **directory** is bounded (index-sized, NOT geometry-sized): levels, group
  records, and a `PageRef` table (offset/len/level/cluster-ids/deps).

## Read paths
| fn | reads | use |
|---|---|---|
| `read_header` | 24 B | validate + locate directory |
| `read_directory` | header + directory range | plan residency, NO geometry |
| `read_root` | directory + root page(s) | inspect superseded bake lineage |
| `read_page(&PageRef)` | one page's range | stream a page on GPU request |
| `Directory::subtree_pages(id)` | directory only | transitive page deps of a subtree |
| `deserialize` | everything | full in-memory `Dag` (tests / tools) |

## Pages (group- / leaf-granular)
Every cluster lives in exactly ONE page:
- a **group page** holds the parent clusters that group produced.
- a **leaf page** holds level-0 clusters, bucketed by consuming (parent) group;
  ungrouped leaves ride a single orphan page.
Pages are laid out in a deterministic order (group id, then leaf buckets in
key order), so identical DAGs serialize byte-identically (finding 8).

## `PageRef`
| field | type | meaning |
|---|---|---|
| `id` | `u32` | page id = index into `Directory.pages` |
| `offset` | `u64` | absolute byte offset of the page chunk |
| `len` | `u32` | page chunk length (range read = `bytes[offset..offset+len]`) |
| `level` | `u32` | superseded bake-lineage level; never a live selector |
| `clusters` | `Vec<u32>` | cluster ids stored in this page |
| `deps` | `Vec<u32>` | page ids this page's clusters depend on (children live there) |

## `Directory`
| field | type | meaning |
|---|---|---|
| `input_tri_count` | `u32` | input mesh tri count (leaf-sum invariant) |
| `partitioner` | `String` | sole mandatory backend: `"metis"` |
| `levels` | `Vec<Vec<u32>>` | superseded bake lineage; `[0]` = live loss-free leaves |
| `groups` | `Vec<Group>` | superseded bake-lineage child/parent sets + bounds/error |
| `pages` | `Vec<PageRef>` | page index table (order = page id) |
| `roots` | `Vec<u32>` | coarsest-level page ids (load first) |
| `cluster_page` | `Vec<u32>` | cluster id → owning page id |
| `cluster_count` | `u32` | total clusters (reassembly bound) |

## `Group` (finding 2 — the crack-free unit)
| field | type | meaning |
|---|---|---|
| `id` | `u32` | index into `Dag.groups` |
| `level` | `u32` | level of the CHILDREN this group consumes |
| `children` | `Vec<u32>` | child cluster ids sharing this cut |
| `parents` | `Vec<u32>` | parent cluster ids produced by simplifying the group |
| `bounds` | `Bounds` | superseded bake-lineage sphere (merged child geometry) |
| `error` | `f32` | superseded bake-lineage monotone simplify error |

## `Cluster`
| field | type | meaning |
|---|---|---|
| `id` | `u32` | global id = index into `Dag.clusters` |
| `level` | `u32` | level (0 = leaf) |
| `vertices` | `Vec<Vertex>` | self-contained compact vertex list (≤ `max_vertices`) |
| `indices` | `Vec<u32>` | local tri indices into `vertices` (≤ `max_triangles` tris) |
| `error` | `f32` | absolute world-unit error of the simplify that produced this (0 leaf); == `group(group).error` |
| `parent_error` | `f32` | switch-UP threshold; `∞` when terminal; == consuming group error |
| `children` | `Vec<u32>` | child cluster ids (empty for leaves) |
| `bounds` | `Bounds` | self bounding sphere (frustum CULLING only) |
| `group` | `Option<u32>` | producing bake-lineage group; `None` for leaves |
| `parent_group` | `Option<u32>` | consuming bake-lineage group; `None` for terminal nodes |

### `Vertex`  (`repr(C)`, 32 B, position at offset 0)
`position:[f32;3]` · `normal:[f32;3]` · `uv:[f32;2]`

> TRACKED (advisory, not built this pass): this fixed 3-attribute vertex must
> grow additional attribute streams (tangents, vertex color, second UV set,
> skinning weights/indices) BEFORE the Great Chain becomes the sole geometry
> pipeline (RENDER.md §1). Adding streams changes the `Vertex` layout → a
> FORMAT_VERSION bump per the policy below.

## SUPERSEDED runtime cut rule
The former `parent_error > τ ≥ error` projection rule is deleted from the
renderer. Group bounds/error remain serialized only as bake lineage; runtime
geometry is always the complete level-0 leaf set in the BVH.

## Invariants (enforced by tests — `src/lib.rs`, `tests/inquisition.rs`)
- `sum(tri_count over levels[0]) == input_tri_count` (shardize = loss-free partition)
- leaf triangle MULTISET (by sorted positions) == input triangle multiset
- every cluster: `vertices.len() ≤ max_vertices` AND `tri_count ≤ max_triangles`
- monotone error up the chain: `child.error ≤ group.error == parent.error ≤ parent_error`
- shared borders resolve consistently across neighboring groups (boundary locking)
- serialize→deserialize = identity; each page range-reads independently; root
  loads without the rest; the root dependency closure covers every page
- two INDEPENDENT builds of one input → byte-identical file

## Versioning policy (finding 4 — the old "additive ⇒ no bump" claim was WRONG)
This is a RANGE-INDEXED format: page/directory offsets are absolute. Adding a
field to `Dag`/`Cluster`/`Group` shifts every downstream offset, so a reader on
the old layout mis-decodes. Therefore:
- ANY layout or semantic change (new field, reordered chunk, changed header) →
  bump `FORMAT_VERSION` and update this document. There is NO "additive fields
  need no bump" exemption.
- old readers reject a newer/older `format_version` loudly; they never guess.
