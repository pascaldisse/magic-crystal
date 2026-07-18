//! Re-inquisition regression suite — one test (or cluster of tests) per
//! inquisitor MUST-FIX finding. Named by finding number so a failure points
//! straight at the ruling it defends.

use std::collections::{BTreeMap, BTreeSet};
use transmutation::{
    default_partitioner, deserialize, read_directory, read_page, read_root, serialize, transmute,
    transmute_default, uv_sphere, Cluster, Dag, Mesh, MeshletParams, TransmuteError,
    TransmuteParams, Vertex, WeldParams,
};

// ---------------------------------------------------------------------------
// mesh fixtures (some deliberately pathological: holes / non-manifold / open)
// ---------------------------------------------------------------------------

/// A subdivided plane in the XZ plane — OPEN BORDERS everywhere (finding 1/5).
fn plane_grid(size: f32, n: usize) -> Mesh {
    let mut m = Mesh::default();
    let row = n + 1;
    for iz in 0..=n {
        for ix in 0..=n {
            let x = (ix as f32 / n as f32 - 0.5) * size;
            let z = (iz as f32 / n as f32 - 0.5) * size;
            m.vertices.push(Vertex::new(
                [x, 0.0, z],
                [0.0, 1.0, 0.0],
                [ix as f32 / n as f32, iz as f32 / n as f32],
            ));
        }
    }
    for iz in 0..n {
        for ix in 0..n {
            let a = (iz * row + ix) as u32;
            let b = a + 1;
            let c = ((iz + 1) * row + ix) as u32;
            let d = c + 1;
            m.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    m
}

/// Two plane grids meeting at a shared seam (x=0), the SECOND grid nudged by a
/// sub-quantum `offset` in x. Seam positions quantize to ONE key but carry two
/// distinct physical coordinates — the finding-1 canonical-coordinate probe
/// (adjacent grids 1e-6 apart inside pos_quant). Enough tris to force a chain.
fn two_offset_grids(size: f32, n: usize, offset: f32) -> Mesh {
    let mut m = Mesh::default();
    let row = n + 1;
    let emit = |m: &mut Mesh, x0: f32, dx: f32| {
        let start = m.vertices.len() as u32;
        for iz in 0..=n {
            for ix in 0..=n {
                let x = x0 + (ix as f32 / n as f32) * size + dx;
                let z = (iz as f32 / n as f32 - 0.5) * size;
                m.vertices.push(Vertex::new(
                    [x, 0.0, z],
                    [0.0, 1.0, 0.0],
                    [ix as f32 / n as f32, iz as f32 / n as f32],
                ));
            }
        }
        for iz in 0..n {
            for ix in 0..n {
                let a = start + (iz * row + ix) as u32;
                let b = a + 1;
                let c = start + ((iz + 1) * row + ix) as u32;
                let d = c + 1;
                m.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    };
    emit(&mut m, -size, 0.0); // grid A: x in [-size, 0]
    emit(&mut m, 0.0, offset); // grid B: x in [offset, size+offset]
    m
}

/// Plane grid with a rectangular block of quads removed → a HOLE (finding 1/5).
fn plane_with_hole(size: f32, n: usize) -> Mesh {
    let full = plane_grid(size, n);
    let row = n + 1;
    let lo = n / 3;
    let hi = 2 * n / 3;
    let mut m = Mesh {
        vertices: full.vertices,
        indices: Vec::new(),
    };
    for iz in 0..n {
        for ix in 0..n {
            if ix >= lo && ix < hi && iz >= lo && iz < hi {
                continue; // punch the hole
            }
            let a = (iz * row + ix) as u32;
            let b = a + 1;
            let c = ((iz + 1) * row + ix) as u32;
            let d = c + 1;
            m.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    m
}

/// Three triangles sharing ONE edge → NON-MANIFOLD (finding 1/5).
fn non_manifold_fan() -> Mesh {
    let n = [0.0, 0.0, 1.0];
    let vs = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.5, 1.0],
    ];
    let mut m = Mesh::default();
    for p in vs {
        m.vertices.push(Vertex::new(p, n, [p[0], p[1]]));
    }
    // edge 0-1 shared by three faces (2, 3, 4)
    m.indices.extend_from_slice(&[0, 1, 2, 0, 1, 3, 0, 1, 4]);
    m
}

/// Two coincident sphere shells at the same center → COINCIDENT SHELL (finding 5).
fn coincident_shells(radius: f32, u: usize, v: usize) -> Mesh {
    let a = uv_sphere(radius, u, v);
    let b = uv_sphere(radius, u, v);
    let mut m = a.clone();
    let base = m.vertices.len() as u32;
    m.vertices.extend(b.vertices);
    m.indices.extend(b.indices.iter().map(|&i| i + base));
    m
}

/// A quad with a duplicated seam edge carrying DIFFERENT UVs on each side →
/// UV SEAM (finding 5): the two seam vertices coincide in position but must NOT
/// weld.
fn uv_seam_strip() -> Mesh {
    let n = [0.0, 0.0, 1.0];
    let mut m = Mesh::default();
    // left quad: verts 0..4, seam edge at x=0 with uv u=1.0
    m.vertices
        .push(Vertex::new([-1.0, 0.0, 0.0], n, [0.0, 0.0]));
    m.vertices.push(Vertex::new([0.0, 0.0, 0.0], n, [1.0, 0.0])); // seam, uv=1
    m.vertices
        .push(Vertex::new([-1.0, 1.0, 0.0], n, [0.0, 1.0]));
    m.vertices.push(Vertex::new([0.0, 1.0, 0.0], n, [1.0, 1.0])); // seam, uv=1
    m.indices.extend_from_slice(&[0, 1, 2, 2, 1, 3]);
    // right quad: seam verts COINCIDE in position but uv u=0.0 (a real seam)
    m.vertices.push(Vertex::new([0.0, 0.0, 0.0], n, [0.0, 0.0])); // seam, uv=0
    m.vertices.push(Vertex::new([1.0, 0.0, 0.0], n, [1.0, 0.0]));
    m.vertices.push(Vertex::new([0.0, 1.0, 0.0], n, [0.0, 1.0])); // seam, uv=0
    m.vertices.push(Vertex::new([1.0, 1.0, 0.0], n, [1.0, 1.0]));
    m.indices.extend_from_slice(&[4, 5, 6, 6, 5, 7]);
    m
}

// bit-exact position key (locked verts keep byte-identical coords)
fn pkey(p: [f32; 3]) -> [u32; 3] {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

// ---------------------------------------------------------------------------
// Finding 1 — BOUNDARY LOCKING: shared borders survive identically in both
// neighboring groups' parents (no mixed-LOD cracks).
// ---------------------------------------------------------------------------

fn assert_boundaries_locked(dag: &Dag, label: &str) {
    // group borders per child-level: canonical position → groups touching it.
    let mut by_level: BTreeMap<u32, Vec<&transmutation::Group>> = BTreeMap::new();
    for g in &dag.groups {
        by_level.entry(g.level).or_default().push(g);
    }
    let mut checked = 0usize;
    let mut survived = 0usize; // shared borders that stayed present in ALL producers
    for (_lvl, groups) in by_level {
        // position → set of group ids whose CHILDREN contain it
        let mut pos_groups: BTreeMap<[u32; 3], BTreeSet<u32>> = BTreeMap::new();
        for g in &groups {
            for &c in &g.children {
                for v in &dag.cluster(c).vertices {
                    pos_groups.entry(pkey(v.position)).or_default().insert(g.id);
                }
            }
        }
        // parent position sets per group (what survived simplify)
        let mut parent_pos: BTreeMap<u32, BTreeSet<[u32; 3]>> = BTreeMap::new();
        for g in &groups {
            let set = parent_pos.entry(g.id).or_default();
            for &p in &g.parents {
                for v in &dag.cluster(p).vertices {
                    set.insert(pkey(v.position));
                }
            }
        }
        for (pos, gset) in &pos_groups {
            if gset.len() < 2 {
                continue; // interior, not a border
            }
            // A locked border position must resolve CONSISTENTLY across every
            // touching group that produced parents: present in ALL of them or
            // absent from ALL (consistent removal of a degenerate sliver is not
            // a crack; present-in-one/absent-in-another IS).
            let mut present = 0usize;
            let mut producing = 0usize;
            for gid in gset {
                let pp = &parent_pos[gid];
                if pp.is_empty() {
                    continue; // group produced no parents (degenerate) — skip
                }
                producing += 1;
                if pp.contains(pos) {
                    present += 1;
                }
            }
            if producing >= 2 {
                assert!(
                    present == 0 || present == producing,
                    "{label}: border position {pos:?} present in {present}/{producing} \
                     neighboring groups → mixed-LOD crack"
                );
                if present == producing {
                    survived += 1; // advisory: a real, PRESENT-in-≥1-parent border
                }
                checked += 1;
            }
        }
    }
    // Advisory: the regression must actually WITNESS surviving shared borders
    // (present in every producer), not pass vacuously on an all-absent world
    // (present==0 everywhere). present==0-for-everything now fails here.
    assert!(
        checked > 0,
        "{label}: no shared borders exercised — weak test"
    );
    assert!(
        survived > 0,
        "{label}: no shared border survived in ≥1 parent (present==0 everywhere) — \
         boundary locking not exercised"
    );
}

// Finding 1 (canonical COORDINATE): two grids offset a sub-quantum apart share
// one canonical KEY per seam vertex, so every group that touches that key MUST
// write the SAME physical coordinate. Pre-fix each group kept its own first-seen
// coordinate → shared keys with mismatched parent boundary coordinate sets.
fn assert_canonical_coords(dag: &Dag, quant: f32, label: &str) {
    // quantize a test position to the probe key (coarser-or-equal to pos_quant)
    let qk = |p: [f32; 3]| -> [i64; 3] {
        let q = |x: f32| (x / quant).round() as i64;
        [q(p[0]), q(p[1]), q(p[2])]
    };
    let mut by_level: BTreeMap<u32, Vec<&transmutation::Group>> = BTreeMap::new();
    for g in &dag.groups {
        by_level.entry(g.level).or_default().push(g);
    }
    let mut shared = 0usize;
    let mut mismatched = 0usize;
    for (_lvl, groups) in by_level {
        // canonical key -> (set of groups producing it, set of bit-exact coords)
        let mut key_groups: BTreeMap<[i64; 3], BTreeSet<u32>> = BTreeMap::new();
        let mut key_coords: BTreeMap<[i64; 3], BTreeSet<[u32; 3]>> = BTreeMap::new();
        for g in &groups {
            for &p in &g.parents {
                for v in &dag.cluster(p).vertices {
                    let k = qk(v.position);
                    key_groups.entry(k).or_default().insert(g.id);
                    key_coords.entry(k).or_default().insert(pkey(v.position));
                }
            }
        }
        for (k, gset) in &key_groups {
            if gset.len() < 2 {
                continue; // not a mixed cut — only one group produced it
            }
            shared += 1;
            if key_coords[k].len() > 1 {
                mismatched += 1; // same key, DIVERGENT physical coordinates
            }
        }
    }
    assert!(
        shared > 0,
        "{label}: no shared canonical positions across mixed cuts — weak probe"
    );
    assert_eq!(
        mismatched, 0,
        "{label}: {mismatched}/{shared} shared canonical positions have MISMATCHED \
         physical coordinates across neighboring groups (finding 1)"
    );
}

#[test]
fn finding1_canonical_coordinate_offset_grids() {
    // Two grids offset 1e-6 in x — inside the pipeline's position quantum, so
    // seam vertices share ONE canonical key. After the canonical-coordinate fix
    // every group that touches a shared key writes the SAME physical coordinate:
    // 0 mismatches across mixed cuts. (Probe quantum 1e-3 groups the 1e-6 twins
    // but not distinct grid points ~0.25 apart.)
    let mesh = two_offset_grids(6.0, 24, 1e-6);
    let dag = transmute(&mesh, &TransmuteParams::default(), &default_partitioner()).unwrap();
    assert!(dag.level_count() > 1, "need a chain to have parent borders");
    assert_canonical_coords(&dag, 1e-3, dag.partitioner.as_str());
}

#[test]
fn finding1_boundary_locking_sphere_and_plane() {
    let params = TransmuteParams::default();
    // Clean-topology meshes (no degenerate poles): plane grid + subdivided cube
    // (shared edge positions across faces = real cross-group borders).
    for (mesh, label) in [
        (plane_grid(10.0, 48), "plane"),
        (transmutation::subdivided_cube(2.0, 40), "cube"),
    ] {
        let dag = transmute_default(&mesh, &params).unwrap();
        assert!(
            dag.level_count() > 1,
            "{label}: need a chain to have borders"
        );
        assert_boundaries_locked(&dag, label);
    }
}

#[test]
fn finding1_boundary_locking_pathological() {
    let params = TransmuteParams::default();
    for (mesh, label) in [
        (plane_with_hole(10.0, 48), "hole"),
        (non_manifold_fan(), "non-manifold"),
        (plane_grid(10.0, 40), "open-border"),
    ] {
        // must not panic, must stay loss-free
        let dag = transmute_default(&mesh, &params).unwrap();
        assert_eq!(
            dag.leaf_tri_sum(),
            mesh.tri_count(),
            "{label}: lossy leaves"
        );
        if dag.groups.iter().any(|g| g.children.len() > 1) {
            assert_boundaries_locked(&dag, label);
        }
    }
}

// ---------------------------------------------------------------------------
// Finding 2 — GROUP RECORDS: shared LOD sphere + error; members transition on
// the same metric.
// ---------------------------------------------------------------------------

#[test]
fn finding2_group_records_share_metric() {
    let dag = transmute_default(&uv_sphere(1.0, 96, 64), &TransmuteParams::default()).unwrap();
    assert!(!dag.groups.is_empty());
    for g in &dag.groups {
        assert!(
            g.bounds.radius > 0.0,
            "group {} has empty shared sphere",
            g.id
        );
        // every produced parent carries the group's shared error + back-ref
        for &p in &g.parents {
            let c = dag.cluster(p);
            assert_eq!(c.error, g.error, "parent {p} error != group error");
            assert_eq!(
                c.group,
                Some(g.id),
                "parent {p} missing producing-group ref"
            );
        }
        // every consumed child transitions UP on the group's shared error
        for &ch in &g.children {
            let c = dag.cluster(ch);
            assert_eq!(
                c.parent_error, g.error,
                "child {ch} parent_error != group error"
            );
            assert_eq!(
                c.parent_group,
                Some(g.id),
                "child {ch} missing consuming-group ref"
            );
            // monotone: a child is never coarser than its parent group
            assert!(
                c.error <= g.error,
                "child {ch} error exceeds group error (non-monotone)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Finding 3 — STAGED LEVEL COMMIT: rejection leaves no orphans / no ∞ corruption.
// ---------------------------------------------------------------------------

#[test]
fn finding3_unsimplifiable_stays_terminal_no_orphans() {
    // simplify_ratio = 1.0 → zero reduction → the first build level is rejected.
    let params = TransmuteParams {
        simplify_ratio: 1.0,
        ..TransmuteParams::default()
    };
    let dag = transmute_default(&uv_sphere(1.0, 64, 48), &params).unwrap();
    assert_eq!(dag.level_count(), 1, "no level should commit");
    assert!(dag.groups.is_empty(), "rejected level must leave no groups");
    // every cluster reachable from levels; every leaf terminal with ∞ threshold.
    let reachable: BTreeSet<u32> = dag.levels.iter().flatten().copied().collect();
    assert_eq!(reachable.len(), dag.clusters.len(), "orphan clusters exist");
    for c in &dag.clusters {
        assert!(
            c.parent_error.is_infinite(),
            "terminal child lost ∞ threshold"
        );
        assert_eq!(c.parent_group, None);
        assert_eq!(c.group, None);
    }
}

#[test]
fn finding3_single_triangle_terminal() {
    let mut m = Mesh::default();
    let n = [0.0, 0.0, 1.0];
    m.vertices.push(Vertex::new([0.0, 0.0, 0.0], n, [0.0, 0.0]));
    m.vertices.push(Vertex::new([1.0, 0.0, 0.0], n, [1.0, 0.0]));
    m.vertices.push(Vertex::new([0.0, 1.0, 0.0], n, [0.0, 1.0]));
    m.indices.extend_from_slice(&[0, 1, 2]);
    let dag = transmute_default(&m, &TransmuteParams::default()).unwrap();
    assert_eq!(dag.leaf_tri_sum(), 1);
    let reachable: BTreeSet<u32> = dag.levels.iter().flatten().copied().collect();
    assert_eq!(reachable.len(), dag.clusters.len());
}

// ---------------------------------------------------------------------------
// Finding 4 — CHUNKED FORMAT: root loadable alone; pages range-readable;
// dependency graph well-formed; version bumped.
// ---------------------------------------------------------------------------

#[test]
fn finding4_root_loads_without_the_rest() {
    let dag = transmute_default(&uv_sphere(1.0, 96, 64), &TransmuteParams::default()).unwrap();
    let bytes = serialize(&dag).unwrap();

    // read_root touches ONLY the directory + root pages.
    let (dir, root_clusters) = read_root(&bytes).unwrap();
    assert!(!dir.roots.is_empty(), "no root pages recorded");
    assert!(
        dir.roots.len() < dir.pages.len(),
        "root must not be the whole file"
    );
    // the returned clusters are exactly the coarsest level.
    let top: BTreeSet<u32> = dag.levels.last().unwrap().iter().copied().collect();
    let got: BTreeSet<u32> = root_clusters.iter().map(|c| c.id).collect();
    assert_eq!(got, top, "root page != coarsest level");
}

#[test]
fn finding4_pages_range_read_independently() {
    let dag = transmute_default(&uv_sphere(1.0, 96, 64), &TransmuteParams::default()).unwrap();
    let bytes = serialize(&dag).unwrap();
    let dir = read_directory(&bytes).unwrap();
    // every page decodes from its own byte range alone, and its clusters match.
    for pr in &dir.pages {
        let page = read_page(&bytes, pr).unwrap();
        let want: BTreeSet<u32> = pr.clusters.iter().copied().collect();
        let got: BTreeSet<u32> = page.clusters.iter().map(|c| c.id).collect();
        assert_eq!(want, got, "page {} range read mismatch", pr.id);
        for c in &page.clusters {
            assert_eq!(
                *c,
                *dag.cluster(c.id),
                "page {} cluster {} corrupted",
                pr.id,
                c.id
            );
        }
    }
}

#[test]
fn finding4_dependency_closure_covers_all_pages() {
    let dag = transmute_default(&uv_sphere(1.0, 96, 64), &TransmuteParams::default()).unwrap();
    let bytes = serialize(&dag).unwrap();
    let dir = read_directory(&bytes).unwrap();
    // Following root dependencies downward must reach every page (no unreachable
    // geometry, no dangling dep offsets).
    let mut reached: BTreeSet<u32> = BTreeSet::new();
    for &r in &dir.roots {
        for p in dir.subtree_pages(r) {
            reached.insert(p);
        }
    }
    let all: BTreeSet<u32> = dir.pages.iter().map(|p| p.id).collect();
    assert_eq!(reached, all, "some pages unreachable via dependency graph");
    // deps offsets are valid range reads.
    for pr in &dir.pages {
        for &d in &pr.deps {
            let dep = dir.page_ref(d).expect("dangling dep id");
            assert!(read_page(&bytes, dep).is_ok(), "dep page {d} unreadable");
        }
    }
}

#[test]
fn finding4_version_bumped_and_v1_rejected() {
    assert_eq!(
        transmutation::FORMAT_VERSION,
        3,
        "entropy/partition semantics must bump the format version"
    );
    // a v1-style blob (magic + version=1) must be rejected loudly.
    let mut fake = Vec::new();
    fake.extend_from_slice(&transmutation::MAGIC);
    fake.extend_from_slice(&1u16.to_le_bytes());
    fake.extend_from_slice(&[0u8; 32]);
    assert!(deserialize(&fake).is_err(), "v1 must not mis-decode as v2");
}

// ---------------------------------------------------------------------------
// Finding 5 — UV-SEAM-SAFE WELD: seams survive; pathological inputs stay
// loss-free (never welded into corruption).
// ---------------------------------------------------------------------------

#[test]
fn finding5_uv_seam_not_welded() {
    // The seam position (0,0,0)/(0,1,0) carries uv u=1.0 on the left and u=0.0
    // on the right. After transmute the leaves must still expose BOTH uvs at
    // that position — a position-only weld would have fused them.
    let dag = transmute_default(&uv_seam_strip(), &TransmuteParams::default()).unwrap();
    let mut uvs_at_seam: BTreeSet<u32> = BTreeSet::new();
    for c in &dag.clusters {
        for v in &c.vertices {
            if v.position[0] == 0.0 && (v.position[1] == 0.0 || v.position[1] == 1.0) {
                uvs_at_seam.insert(v.uv[0].to_bits());
            }
        }
    }
    assert!(
        uvs_at_seam.contains(&0.0f32.to_bits()) && uvs_at_seam.contains(&1.0f32.to_bits()),
        "UV seam collapsed — both u=0 and u=1 must survive, got {uvs_at_seam:?}"
    );
}

/// A large seam strip: two grids meeting at x=0 whose seam vertices coincide in
/// position but carry u=1 (left) vs u=0 (right). Big enough to build a MULTI-
/// level chain, so the seam must survive THROUGH simplification, not just at the
/// leaves (advisory).
fn uv_seam_grid(size: f32, n: usize) -> Mesh {
    let mut m = Mesh::default();
    let row = n + 1;
    // left half: x in [-size, 0], seam column (x=0) uv u=1.0
    let half = |m: &mut Mesh, x0: f32, u_at_seam: f32, seam_on_right: bool| {
        let start = m.vertices.len() as u32;
        for iz in 0..=n {
            for ix in 0..=n {
                let t = ix as f32 / n as f32;
                let x = x0 + t * size;
                let z = (iz as f32 / n as f32 - 0.5) * size;
                // u ramps 0..1 across the half; force the seam edge's u so the
                // two halves DISAGREE at x=0 (a real texture seam).
                let at_seam = if seam_on_right { ix == n } else { ix == 0 };
                let u = if at_seam { u_at_seam } else { t };
                m.vertices.push(Vertex::new(
                    [x, 0.0, z],
                    [0.0, 1.0, 0.0],
                    [u, iz as f32 / n as f32],
                ));
            }
        }
        for iz in 0..n {
            for ix in 0..n {
                let a = start + (iz * row + ix) as u32;
                let b = a + 1;
                let c = start + ((iz + 1) * row + ix) as u32;
                let d = c + 1;
                m.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    };
    half(&mut m, -size, 1.0, true); // left half, seam at its right edge, u=1
    half(&mut m, 0.0, 0.0, false); // right half, seam at its left edge, u=0
    m
}

#[test]
fn finding5_uv_seam_survives_multi_level() {
    // Advisory: exercise the seam on an ACTUAL hierarchy (multi-level), not just
    // the leaf layer. Both u=0 and u=1 must persist at the seam position at
    // EVERY non-leaf level too.
    let dag = transmute_default(&uv_seam_grid(6.0, 24), &TransmuteParams::default()).unwrap();
    assert!(
        dag.level_count() > 1,
        "need a multi-level chain to test seam survival"
    );
    let u0 = 0.0f32.to_bits();
    let u1 = 1.0f32.to_bits();
    let mut checked_levels = 0usize;
    for (lvl, ids) in dag.levels.iter().enumerate() {
        let mut uvs: BTreeSet<u32> = BTreeSet::new();
        for &id in ids {
            for v in &dag.cluster(id).vertices {
                if v.position[0] == 0.0 {
                    uvs.insert(v.uv[0].to_bits());
                }
            }
        }
        // A-2: the old test SKIPPED any level where BOTH sentinels vanished,
        // so a level that silently dropped the whole seam passed vacuously.
        // Now assert per NON-LEAF level (lvl > 0): at least one sentinel must be
        // present (FAIL if both lost), AND if the seam is present it must keep
        // BOTH u=0 and u=1 (no half-seam crack). The leaf level (lvl 0) is
        // covered by finding5_uv_seam_not_welded.
        if lvl > 0 {
            let has0 = uvs.contains(&u0);
            let has1 = uvs.contains(&u1);
            assert!(
                has0 || has1,
                "non-leaf level {lvl}: UV seam wholly vanished — both sentinels lost \
                 (A-2: no level may drop the seam entirely)"
            );
            assert!(
                has0 && has1,
                "non-leaf level {lvl}: UV seam collapsed — both u=0 and u=1 must survive, got {uvs:?}"
            );
            checked_levels += 1;
        }
    }
    assert!(
        checked_levels > 1,
        "seam not exercised across multiple non-leaf levels ({checked_levels})"
    );
}

#[test]
fn finding5_pathological_lossless() {
    let params = TransmuteParams::default();
    let cases: [(Mesh, &str); 4] = [
        (coincident_shells(1.0, 48, 32), "coincident-shell"),
        (uv_sphere(1e-4, 48, 32), "tiny-scale"),
        (non_manifold_fan(), "non-manifold"),
        (plane_with_hole(10.0, 40), "hole/internal"),
    ];
    for (mesh, label) in cases {
        let dag = transmute(&mesh, &params, &default_partitioner()).unwrap();
        assert_eq!(
            dag.leaf_tri_sum(),
            mesh.tri_count(),
            "{label}: shardize was lossy"
        );
        for c in &dag.clusters {
            assert!(c.vertices.len() <= params.meshlet.max_vertices);
            assert!(c.tri_count() <= params.meshlet.max_triangles);
        }
    }
}

// ---------------------------------------------------------------------------
// Finding 6 — CAP ENFORCEMENT: illegal params rejected with typed errors,
// BEFORE any FFI.
// ---------------------------------------------------------------------------

#[test]
fn finding6_illegal_params_rejected() {
    let mesh = uv_sphere(1.0, 16, 12);
    let base = TransmuteParams::default();
    let run = |p: TransmuteParams| transmute_default(&mesh, &p).unwrap_err();

    // zero budgets
    let mut p = base;
    p.meshlet.max_vertices = 0;
    assert_eq!(run(p), TransmuteError::ZeroBudget("max_vertices"));
    let mut p = base;
    p.meshlet.max_triangles = 0;
    assert_eq!(run(p), TransmuteError::ZeroBudget("max_triangles"));
    let mut p = base;
    p.group_size = 0;
    assert_eq!(run(p), TransmuteError::ZeroBudget("group_size"));

    // engine ceilings
    let mut p = base;
    p.meshlet.max_vertices = 256;
    assert_eq!(run(p), TransmuteError::VerticesTooLarge(256));
    let mut p = base;
    p.meshlet.max_triangles = 200; // multiple of 4 but > 124 ceiling
    assert_eq!(run(p), TransmuteError::TrianglesTooLarge(200));

    // meshopt multiple-of-4 contract
    let mut p = base;
    p.meshlet.max_triangles = 122; // <=124 but not a multiple of 4
    assert_eq!(run(p), TransmuteError::TrianglesNotMultiple(122));

    // out-of-range ratios/errors
    let mut p = base;
    p.simplify_ratio = 2.0;
    assert_eq!(run(p), TransmuteError::OutOfRange("simplify_ratio"));
    let mut p = base;
    p.simplify_ratio = f32::NAN;
    assert_eq!(run(p), TransmuteError::OutOfRange("simplify_ratio"));
    let mut p = base;
    p.target_error = -1.0;
    assert_eq!(run(p), TransmuteError::OutOfRange("target_error"));

    // bad weld tolerances
    let mut p = base;
    p.weld = WeldParams {
        pos_quant_frac: 0.0,
        ..WeldParams::default()
    };
    assert_eq!(run(p), TransmuteError::OutOfRange("pos_quant_frac"));
}

// ---------------------------------------------------------------------------
// Finding 7 — SOLE METIS: no fallback implementation exists.
// ---------------------------------------------------------------------------

// Finding 8 — DETERMINISM: two independent builds → byte-identical output.
// ---------------------------------------------------------------------------

/// The sole METIS backend is deterministic in-process: its seed derives from
/// identical `(world_seed, entropy, canonical geometry identity)` state, and
/// the seed→partition interval is process-serialized.
#[test]
fn finding8_metis_default_in_process_double_build_identical() {
    let mesh = uv_sphere(1.0, 96, 64);
    let params = TransmuteParams::default();
    let a = transmute_default(&mesh, &params).unwrap();
    let b = transmute_default(&mesh, &params).unwrap();
    // With METIS compiled in, this proves the per-call InitRandom(seed) claim.
    assert_eq!(
        a.partitioner, b.partitioner,
        "backend varied between builds"
    );
    assert_eq!(
        serialize(&a).unwrap(),
        serialize(&b).unwrap(),
        "two in-process default-backend builds are NOT byte-identical"
    );
}

/// A nonzero world coordinate is reproducible: identical
/// `(world_seed, entropy, canonical geometry)` produces identical pages.
#[test]
fn finding8_entropy_state_is_byte_identical() {
    let mesh = uv_sphere(1.0, 96, 64);
    let params = TransmuteParams {
        world_seed: 0x4d43_2026,
        entropy: 731,
        ..TransmuteParams::default()
    };
    let a = transmute_default(&mesh, &params).unwrap();
    let b = transmute_default(&mesh, &params).unwrap();
    assert_eq!(serialize(&a).unwrap(), serialize(&b).unwrap());
}

/// The real gate: two independent METIS process runs with the same canonical
/// input and entropy state produce byte-identical `.cbdg` pages.
#[test]
fn finding8_two_independent_builds_byte_identical() {
    let exe = env!("CARGO_BIN_EXE_transmute-cli");
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")); // under target/, never /tmp
    let a = dir.join("det_a.cbdg");
    let b = dir.join("det_b.cbdg");
    let run = |out: &std::path::Path| {
        let status = std::process::Command::new(exe)
            .args(["sphere", "--res", "128", "--out"])
            .arg(out)
            .status()
            .expect("spawn transmute-cli");
        assert!(status.success());
    };
    run(&a);
    run(&b);
    let ba = std::fs::read(&a).unwrap();
    let bb = std::fs::read(&b).unwrap();
    assert_eq!(ba, bb, "two independent builds are NOT byte-identical");
    assert!(!ba.is_empty());
}

// ---------------------------------------------------------------------------
// ADVISORY — leaf triangle MULTISET equality (not just count).
// ---------------------------------------------------------------------------

#[test]
fn advisory_leaf_triangle_multiset_equals_input() {
    let mesh = uv_sphere(1.0, 48, 32);
    let dag = transmute_default(&mesh, &TransmuteParams::default()).unwrap();

    let tri_key = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| -> [[u32; 3]; 3] {
        let mut t = [pkey(a), pkey(b), pkey(c)];
        t.sort();
        t
    };
    let mut input: BTreeMap<[[u32; 3]; 3], i32> = BTreeMap::new();
    for tri in mesh.indices.chunks(3) {
        let k = tri_key(
            mesh.vertices[tri[0] as usize].position,
            mesh.vertices[tri[1] as usize].position,
            mesh.vertices[tri[2] as usize].position,
        );
        *input.entry(k).or_insert(0) += 1;
    }
    let mut leaves: BTreeMap<[[u32; 3]; 3], i32> = BTreeMap::new();
    for &id in &dag.levels[0] {
        let c: &Cluster = dag.cluster(id);
        for tri in c.indices.chunks(3) {
            let k = tri_key(
                c.vertices[tri[0] as usize].position,
                c.vertices[tri[1] as usize].position,
                c.vertices[tri[2] as usize].position,
            );
            *leaves.entry(k).or_insert(0) += 1;
        }
    }
    assert_eq!(input, leaves, "leaf triangle MULTISET differs from input");
}

// ---------------------------------------------------------------------------
// ADVISORY — monotone error up the whole chain.
// ---------------------------------------------------------------------------

#[test]
fn advisory_error_monotone_up_chain() {
    let dag = transmute_default(&uv_sphere(1.0, 96, 64), &TransmuteParams::default()).unwrap();
    for c in &dag.clusters {
        assert!(
            c.error <= c.parent_error,
            "cluster {} error > parent_error",
            c.id
        );
        for &ch in &c.children {
            assert!(
                dag.cluster(ch).error <= c.error,
                "child {ch} coarser than parent {}",
                c.id
            );
        }
    }
}

// keep MeshletParams import meaningful
#[test]
fn budgets_default_is_engine_ceiling() {
    let p = MeshletParams::default();
    assert!(p.max_vertices <= transmutation::MAX_VERTICES_CEIL);
    assert!(p.max_triangles <= transmutation::MAX_TRIANGLES_CEIL);
    assert_eq!(p.max_triangles % transmutation::TRI_MULTIPLE, 0);
}
