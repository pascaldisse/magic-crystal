//! The ORDEALS of Jörmungandr — the serpent's law, proven not asserted.
//!
//! A large artifact is FORGED from procedural geometry (never a committed blob;
//! see `support::generate_artifact`) under cargo's `CARGO_TARGET_TMPDIR`
//! (always under `target/`, never `/tmp`). A scripted 1000-step observer flight
//! drives the ring over it.

mod support;

use jormungandr::{ArtifactId, PageKey, Ring, RingError};
use support::{generate_artifact, page_boxes, PageBox};

const FLIGHT_STEPS: usize = 1000;
const CUT_K: usize = 8;

/// Deterministic observer flight through the lattice box (~0..48 per axis).
fn observer_at(step: usize) -> [f32; 3] {
    let t = step as f32;
    [
        24.0 + 22.0 * (t * 0.017).sin(),
        24.0 + 22.0 * (t * 0.011).cos(),
        24.0 + 22.0 * (t * 0.023).sin(),
    ]
}

/// The current cut = the K spatially-nearest pages to the observer. Deterministic
/// (distance then page-id tie-break).
fn cut(boxes: &[PageBox], art: ArtifactId, obs: [f32; 3], k: usize) -> Vec<PageKey> {
    let mut idx: Vec<&PageBox> = boxes.iter().collect();
    idx.sort_by(|a, b| {
        a.distance2(obs)
            .partial_cmp(&b.distance2(obs))
            .unwrap()
            .then(a.page.cmp(&b.page))
    });
    idx.into_iter()
        .take(k)
        .map(|pb| PageKey::new(art, pb.page))
        .collect()
}

/// Size a budget that (a) fits the largest cut with headroom (so the cut can
/// always stay resident) yet (b) is far smaller than the whole artifact (so
/// residency, not "load it all", is forced).
fn sized_budget(boxes: &[PageBox], art: ArtifactId) -> u64 {
    let mut max_cut = 0u64;
    for step in 0..FLIGHT_STEPS {
        let c = cut(boxes, art, observer_at(step), CUT_K);
        let bytes: u64 = c.iter().map(|k| boxes[k.page as usize].len as u64).sum();
        max_cut = max_cut.max(bytes);
    }
    // 1.5x the largest cut: every cut fits, but non-required pages accumulate
    // past this line as the observer moves → eviction is forced.
    (max_cut * 3) / 2
}

/// One tick's (loaded, evicted) key lists.
type TickDelta = (Vec<PageKey>, Vec<PageKey>);

/// Run the whole flight, returning the per-tick (loaded, evicted) sequence plus
/// the final ring stats — the raw material for the determinism ordeal.
fn fly(
    path: &std::path::Path,
    boxes: &[PageBox],
    budget: u64,
    mut assert_each: impl FnMut(&Ring, &[PageKey]),
) -> (Vec<TickDelta>, jormungandr::RingStats) {
    let mut ring = Ring::new(budget);
    let art = ring.mount(path).expect("mount");
    let mut seq = Vec::with_capacity(FLIGHT_STEPS);
    for step in 0..FLIGHT_STEPS {
        let obs = observer_at(step);
        let required = cut(boxes, art, obs, CUT_K);
        let tick = ring.update(obs, &required).expect("update");
        assert_each(&ring, &required);
        seq.push((
            tick.loaded_this_tick.clone(),
            tick.evicted_this_tick.clone(),
        ));
    }
    (seq, ring.stats())
}

#[test]
fn ordeals_flight() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = generate_artifact(dir);
    let boxes = page_boxes(&path);
    let art = ArtifactId(0);
    let total_bytes: u64 = boxes.iter().map(|b| b.len as u64).sum();
    assert!(
        boxes.len() >= 40,
        "expected a LARGE artifact (many pages), got {}",
        boxes.len()
    );
    let budget = sized_budget(&boxes, art);
    assert!(
        budget < total_bytes,
        "budget must be below the whole artifact"
    );

    // ORDEAL 1 (budget) + ORDEAL 2 (required resident): checked after EVERY tick.
    let mut peak_seen = 0u64;
    let (seq_a, stats_a) = fly(&path, &boxes, budget, |ring, required| {
        assert!(
            ring.resident_bytes() <= budget,
            "budget exceeded: {} > {}",
            ring.resident_bytes(),
            budget
        );
        for &k in required {
            assert!(ring.is_resident(k), "required page {k:?} not resident");
        }
        peak_seen = peak_seen.max(ring.resident_bytes());
    });
    assert_eq!(
        stats_a.peak_resident_bytes, peak_seen,
        "stats peak must match observed peak"
    );
    assert!(
        stats_a.peak_resident_bytes <= budget,
        "peak resident {} must be ≤ budget {}",
        stats_a.peak_resident_bytes,
        budget
    );
    assert!(
        stats_a.evictions > 0,
        "flight must force eviction (else the budget was not binding)"
    );

    // ORDEAL 3 (determinism): a second independent ring over the same flight
    // replays an IDENTICAL load/evict sequence.
    let (seq_b, stats_b) = fly(&path, &boxes, budget, |_, _| {});
    assert_eq!(seq_a, seq_b, "load/evict sequence must be deterministic");
    assert_eq!(stats_a, stats_b, "cumulative stats must be deterministic");

    eprintln!("── Jörmungandr ordeal report ──────────────────────────────");
    eprintln!("  pages in artifact         : {}", boxes.len());
    eprintln!("  artifact page bytes total : {total_bytes}");
    eprintln!("  budget                    : {budget}");
    eprintln!(
        "  peak resident bytes       : {}",
        stats_a.peak_resident_bytes
    );
    eprintln!(
        "  headroom (budget - peak)  : {}",
        budget - stats_a.peak_resident_bytes
    );
    eprintln!("  flight steps              : {FLIGHT_STEPS}  (cut K = {CUT_K})");
    eprintln!(
        "  loads / bytes             : {} / {}",
        stats_a.loads, stats_a.loaded_bytes
    );
    eprintln!(
        "  evictions / bytes         : {} / {}",
        stats_a.evictions, stats_a.evicted_bytes
    );
    eprintln!("───────────────────────────────────────────────────────────");
}

#[test]
fn ordeal_eviction_order_honors_distance() {
    // Spot-check: with only ONE eviction slot needed, the FARTHEST non-required
    // resident page is the victim — derived from the real page boxes.
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = generate_artifact(dir);
    let boxes = page_boxes(&path);
    let art = ArtifactId(0);

    // Pick four spatially-spread pages by their box centers.
    let mut spread: Vec<&PageBox> = boxes.iter().collect();
    spread.sort_by(|a, b| a.center()[0].partial_cmp(&b.center()[0]).unwrap());
    let chosen: Vec<u32> = [
        spread[0].page,
        spread[spread.len() / 3].page,
        spread[2 * spread.len() / 3].page,
        spread[spread.len() - 1].page,
    ]
    .to_vec();

    // Budget = exactly the four chosen pages' bytes (they can ALL be resident).
    let four_bytes: u64 = chosen.iter().map(|&p| boxes[p as usize].len as u64).sum();
    let mut ring = Ring::new(four_bytes);
    let art = ring.mount(&path).map(|_| art).unwrap();

    // Observer near the first chosen page's center. Make all four resident by
    // requiring them (fills the budget exactly).
    let obs = boxes[chosen[0] as usize].center();
    let all_four: Vec<PageKey> = chosen.iter().map(|&p| PageKey::new(art, p)).collect();
    ring.update(obs, &all_four).unwrap();
    for &p in &chosen {
        assert!(ring.is_resident(PageKey::new(art, p)));
    }

    // Now require just ONE new page (not among the four). To fit it, exactly one
    // of the four must be evicted — and it MUST be the farthest from the
    // observer among the non-required three (the required one is pinned).
    // Require chosen[0] (nearest, pinned) + a NEW page → one victim from {1,2,3}.
    let new_page = boxes
        .iter()
        .find(|b| !chosen.contains(&b.page))
        .expect("a spare page")
        .page;
    let required2 = vec![PageKey::new(art, chosen[0]), PageKey::new(art, new_page)];

    // Expected victim: farthest-from-observer among the non-required resident
    // pages (chosen[1..=3]) — computed the same way the ring does.
    let expected_victim = *chosen[1..]
        .iter()
        .max_by(|&&a, &&b| {
            boxes[a as usize]
                .distance2(obs)
                .partial_cmp(&boxes[b as usize].distance2(obs))
                .unwrap()
                .then(a.cmp(&b))
        })
        .unwrap();

    let tick = ring.update(obs, &required2).unwrap();
    assert_eq!(
        tick.evicted_this_tick,
        vec![PageKey::new(art, expected_victim)],
        "eviction must take the FARTHEST non-required page"
    );
    assert!(ring.is_resident(PageKey::new(art, new_page)));
    assert!(ring.is_resident(PageKey::new(art, chosen[0])));
    assert!(ring.resident_bytes() <= four_bytes);
}

#[test]
fn ordeal_missing_artifact_is_typed_error() {
    let mut ring = Ring::new(1 << 20);
    let missing = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("nope.cbdg");
    match ring.mount(&missing) {
        Err(RingError::Io(_)) | Err(RingError::BadIndex(_)) => {}
        other => panic!("expected a typed error for a missing artifact, got {other:?}"),
    }
}

#[test]
fn ordeal_unknown_page_is_typed_error() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = generate_artifact(dir);
    let mut ring = Ring::new(1 << 20);
    let art = ring.mount(&path).unwrap();
    let n = ring.page_count(art).unwrap() as u32;
    match ring.update([0.0; 3], &[PageKey::new(art, n + 5)]) {
        Err(RingError::UnknownPage { page, .. }) => assert_eq!(page, n + 5),
        other => panic!("expected UnknownPage, got {other:?}"),
    }
}

#[test]
fn ordeal_torn_page_is_typed_error_no_panic() {
    use transmutation::{Directory, PageRef, FORMAT_VERSION, HEADER_LEN, MAGIC};

    // Hand-forge a valid header + directory whose single PageRef claims a byte
    // range running PAST EOF (no page payload written) — a torn artifact. The
    // index reads fine; the page load must return TornPage, never panic.
    let pr = PageRef {
        id: 0,
        offset: HEADER_LEN as u64, // pages region is empty → this range is torn
        len: 8_000_000,            // far past EOF
        level: 0,
        clusters: vec![0],
        deps: vec![],
    };
    let dir = Directory {
        input_tri_count: 0,
        partitioner: "greedy".into(),
        levels: vec![vec![0]],
        groups: vec![],
        pages: vec![pr],
        roots: vec![0],
        cluster_page: vec![0],
        cluster_count: 1,
    };
    let dir_bytes = bincode::serialize(&dir).unwrap();
    let dir_offset = HEADER_LEN as u64; // no pages region

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes()); // flags
    bytes.extend_from_slice(&dir_offset.to_le_bytes());
    bytes.extend_from_slice(&(dir_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes()); // root_page
    assert_eq!(bytes.len(), HEADER_LEN);
    bytes.extend_from_slice(&dir_bytes);

    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("torn.cbdg");
    std::fs::write(&path, &bytes).unwrap();

    let mut ring = Ring::new(1 << 30);
    let art = ring.mount(&path).expect("index still loads");
    match ring.update([0.0; 3], &[PageKey::new(art, 0)]) {
        Err(RingError::TornPage { page, .. }) => assert_eq!(page, 0),
        other => panic!("expected TornPage, got {other:?}"),
    }
}
