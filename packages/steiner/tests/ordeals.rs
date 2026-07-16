//! The determinism ordeals for Reading Steiner (ENTROPY.md: "byte-identical
//! builds and replays are not test hygiene — they are this law's trial").
//!
//! Each ordeal prints its verbatim numbers so the trial is auditable.

use crystal::{Op, OpBatch, SetOp};
use serde_json::{json, Value};
use steiner::{ReadOutcome, Recorder, TornKind};

/// A tiny deterministic LCG so op streams are reproducible without an RNG dep.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn range(&mut self, n: u64) -> u64 {
        (self.next() >> 33) % n.max(1)
    }
}

/// Build op batch `i` deterministically from `seed`: a `set` writing a JSON
/// component value onto one of a small pool of entities.
fn op_at(seed: u64, i: u64) -> OpBatch {
    let mut lcg = Lcg(seed ^ i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let entity = format!("e{}", lcg.range(16));
    let component = ["health", "pose", "wanted", "armor"][(lcg.range(4)) as usize];
    let value = json!({
        "a": lcg.range(1000) as i64,
        "b": (lcg.range(2000) as i64) - 1000,
        "tag": format!("v{}", lcg.range(64)),
    });
    OpBatch {
        dev: false,
        ops: vec![Op::Set(SetOp {
            id: entity,
            component: component.to_owned(),
            value,
            extra: Default::default(),
        })],
        from: Some(format!("caster{}", i % 3)),
        extra: Default::default(),
    }
}

/// Record `count` deterministic ops onto a fresh worldline, one op per tick.
fn record_run(seed: u64, count: u64) -> Recorder {
    let mut recorder = Recorder::new(seed);
    for tick in 0..count {
        recorder.record(&op_at(seed, tick), tick).unwrap();
    }
    recorder
}

#[test]
fn ordeal_roundtrip() {
    let seed = 0xDEAD_BEEF;
    let live = record_run(seed, 1000);
    let live_hash = live.state_hash();

    let (replayed, outcome) = Recorder::replay(live.journal_bytes(), None).unwrap();
    let replay_hash = replayed.state_hash();

    assert_eq!(outcome, ReadOutcome::Complete);
    assert_eq!(live_hash, replay_hash, "replayed ECS state must match live");
    println!(
        "ROUNDTRIP: ops=1000 live_hash=0x{live_hash:016x} replay_hash=0x{replay_hash:016x} outcome={outcome:?}"
    );
}

#[test]
fn ordeal_any_point_reconstruction() {
    let seed = 0x0C0F_FEE5;
    // Live: snapshot the hash the instant tick 500 has been applied.
    let mut live = Recorder::new(seed);
    let mut snapshot_500 = None;
    for tick in 0..1000 {
        live.record(&op_at(seed, tick), tick).unwrap();
        if tick == 500 {
            snapshot_500 = Some(live.state_hash());
        }
    }
    let snapshot_500 = snapshot_500.unwrap();

    // Replay to entropy T=500 and compare.
    let (partial, outcome) = Recorder::replay(live.journal_bytes(), Some(500)).unwrap();
    let replay_500 = partial.state_hash();

    assert_eq!(outcome, ReadOutcome::Complete);
    assert_eq!(
        snapshot_500, replay_500,
        "replay to T=500 must match live @500"
    );
    assert_ne!(
        snapshot_500,
        live.state_hash(),
        "state @500 must differ from state @1000"
    );
    println!(
        "ANY-POINT: T=500 live_snapshot=0x{snapshot_500:016x} replay=0x{replay_500:016x} final@1000=0x{:016x}",
        live.state_hash()
    );
}

#[test]
fn ordeal_worldline_fork() {
    let seed = 0xF00D_BA11;
    let parent = record_run(seed, 1000);

    // Two forks from T=500, each cast different subsequent ops.
    let mut a = parent.fork(500).unwrap();
    let mut b = parent.fork(500).unwrap();

    // Prefix hashes (before divergence) must be identical.
    let prefix_a = a.state_hash();
    let prefix_b = b.state_hash();
    assert_eq!(prefix_a, prefix_b, "forks share the same past @500");

    for tick in 501..600 {
        a.record(
            &OpBatch {
                dev: false,
                ops: vec![Op::Set(SetOp {
                    id: "hero".into(),
                    component: "path".into(),
                    value: Value::from(tick as i64),
                    extra: Default::default(),
                })],
                from: None,
                extra: Default::default(),
            },
            tick,
        )
        .unwrap();
        b.record(
            &OpBatch {
                dev: false,
                ops: vec![Op::Set(SetOp {
                    id: "hero".into(),
                    component: "path".into(),
                    value: Value::from(-(tick as i64)),
                    extra: Default::default(),
                })],
                from: None,
                extra: Default::default(),
            },
            tick,
        )
        .unwrap();
    }

    let final_a = a.state_hash();
    let final_b = b.state_hash();
    assert_ne!(final_a, final_b, "divergent worldlines must differ");

    // Identical shared prefix bytes: re-fork both at 500 and compare byte-for-byte.
    let shared_a = steiner::fork_journal(a.journal_bytes(), 500).unwrap();
    let shared_b = steiner::fork_journal(b.journal_bytes(), 500).unwrap();
    assert_eq!(shared_a, shared_b, "shared prefix bytes identical");
    // And that shared prefix hashes to the same ECS state.
    let (pa, _) = Recorder::replay(&shared_a, None).unwrap();
    let (pb, _) = Recorder::replay(&shared_b, None).unwrap();
    assert_eq!(pa.state_hash(), pb.state_hash());

    println!(
        "FORK: prefix@500=0x{prefix_a:016x} final_A=0x{final_a:016x} final_B=0x{final_b:016x} shared_prefix_bytes={} (identical)",
        shared_a.len()
    );
}

#[test]
fn ordeal_torn_write() {
    let seed = 0xBADC_0DE5;
    let full = record_run(seed, 200);
    let full_bytes = full.journal_bytes().to_vec();

    // Decode the intact journal to learn the valid frame count.
    let intact = steiner::read_journal(&full_bytes).unwrap();
    let intact_frames = intact.entries.len();
    assert_eq!(intact.outcome, ReadOutcome::Complete);

    // Truncate mid-final-frame (drop the last 5 bytes: cuts into CRC + payload).
    let torn_len = full_bytes.len() - 5;
    let torn = &full_bytes[..torn_len];

    let decoded = steiner::read_journal(torn).unwrap();
    let (recorder, outcome) = Recorder::replay(torn, None).unwrap();

    match outcome {
        ReadOutcome::Torn { kind, valid_frames } => {
            assert_eq!(kind, TornKind::Truncated);
            assert_eq!(valid_frames, intact_frames - 1, "exactly one frame lost");
            assert_eq!(decoded.entries.len(), valid_frames);
            // The recovered state equals a clean run of the surviving ticks.
            let clean = record_run_ticks(seed, valid_frames as u64);
            assert_eq!(recorder.state_hash(), clean.state_hash());
            println!(
                "TORN: intact_frames={intact_frames} torn_bytes={} kind={kind} valid_frames={valid_frames} (clean stop, no panic)",
                full_bytes.len() - torn_len
            );
        }
        other => panic!("expected torn tail, got {other:?}"),
    }

    // A mid-payload byte flip must be caught by the frame CRC, not applied.
    let mut flipped = full_bytes.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0xFF;
    let flipped_decode = steiner::read_journal(&flipped).unwrap();
    assert!(matches!(
        flipped_decode.outcome,
        ReadOutcome::Torn {
            kind: TornKind::FrameCrc,
            ..
        }
    ));
    println!(
        "TORN-CRC: byte flip at {mid} -> {:?}",
        flipped_decode.outcome
    );
}

/// Record the first `ticks` ops of the deterministic stream (for torn-recovery comparison).
fn record_run_ticks(seed: u64, ticks: u64) -> Recorder {
    let mut recorder = Recorder::new(seed);
    for tick in 0..ticks {
        recorder.record(&op_at(seed, tick), tick).unwrap();
    }
    recorder
}

#[test]
fn ordeal_determinism() {
    let seed = 0x5EED_5EED;
    let first = record_run(seed, 1000).journal_bytes().to_vec();
    let second = record_run(seed, 1000).journal_bytes().to_vec();
    assert_eq!(first, second, "re-recording is byte-identical");

    // Re-replay is byte-identical too: replaying a journal reproduces its bytes.
    let (replayed, outcome) = Recorder::replay(&first, None).unwrap();
    assert_eq!(outcome, ReadOutcome::Complete);
    let replay_bytes = replayed.journal_bytes().to_vec();
    assert_eq!(
        first, replay_bytes,
        "replay reproduces journal bytes exactly"
    );

    println!(
        "DETERMINISM: journal_bytes={} record==record:{} replay==record:{}",
        first.len(),
        first == second,
        first == replay_bytes
    );
}
