//! The LIVE-RECORD ordeal for Reading Steiner's live tap.
//!
//!   GAIA_PORT=8427 bun server/index.js         # from a GAIA-World-Engine checkout
//!   GAIA_PORT=8427 cargo run -p steiner --example live_ordeal --features live
//!
//! Drives a real wired client against a real server: seed a [`LiveTap`] from the
//! connect snapshot, script spawn/move/set batches, journal every applied batch,
//! then prove offline `(seed, journal)` replay reconstructs the LIVE state:
//!   - final replayed ECS hash == live WorldView hash
//!   - any-tick reconstruction differs from the final
//!   - the journal survives a disk round-trip (written under the worktree)
//!   - a torn tail stops cleanly (typed), never panics
//!   - re-replay is byte-identical (determinism)
//!
//! Every number prints verbatim.

use std::time::Duration;
use steiner::live::world_view_hash;
use steiner::{read_journal, LiveTap, ReadOutcome, Recorder, TornKind};
use wired::{Config, Wired};

const SEED: u64 = 0x57E1_7E12_0000_0001;
const DRIVEN_BATCHES: usize = 100;

/// Does this op reference `id` (spawn/despawn/set)? Used to spot the fence echo.
fn op_targets(op: &crystal::Op, id: &str) -> bool {
    match op {
        crystal::Op::Set(s) => s.id == id,
        crystal::Op::Other { fields, .. } => fields.get("id").and_then(|v| v.as_str()) == Some(id),
        _ => false,
    }
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("GAIA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8420);
    let out_dir = std::env::var("STEINER_ORDEAL_DIR")
        .unwrap_or_else(|_| "packages/steiner/ordeal-out".to_string());
    println!("[live-ordeal] port={port} out_dir={out_dir}");

    let client = Wired::connect(Config::with_port(port).presence("player-steiner"));
    assert!(client.wait_live().await, "client never went live");

    // Subscribe to the op stream, then quiesce so the connect snapshot and any
    // startup burst settle BEFORE we capture the base state. Measure ambient
    // traffic in the quiet window so a non-static hub can't corrupt the base.
    let mut batches = client.op_batches();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut ambient = 0usize;
    let quiet_until = tokio::time::Instant::now() + Duration::from_millis(700);
    while let Ok(Ok(_)) = tokio::time::timeout_at(quiet_until, batches.recv()).await {
        ambient += 1;
    }
    println!("[live-ordeal] ambient batches in quiet window: {ambient}");

    // Base state = the client's view right now → journal frame 0.
    let snapshot_view = client.view();
    println!(
        "[live-ordeal] snapshot entities={} -> LiveTap frame 0",
        snapshot_view.entities.len()
    );
    let mut tap = LiveTap::from_world_view(SEED, &snapshot_view).expect("seed tap from snapshot");

    // --- drive DRIVEN_BATCHES scripted op batches: spawn / move / set ---
    client.spawn_presence([0.0, 2.0, 22.0], 0.0).expect("spawn");
    for i in 0..(DRIVEN_BATCHES - 1) {
        match i % 3 {
            0 => {
                let id = format!("steiner-mob-{}", i % 8);
                client
                    .send_ops(vec![wired::spawn_presence_op(
                        &id,
                        [(i % 10) as f64, 2.0, 20.0 + (i % 5) as f64],
                        (i as f64) * 0.1,
                    )])
                    .expect("spawn op");
            }
            1 => {
                let x = (i % 13) as f64;
                client
                    .move_presence([x, 2.0, 18.0], (i as f64) * 0.05)
                    .expect("move op");
            }
            _ => {
                let id = format!("steiner-mob-{}", i % 8);
                client
                    .send_ops(vec![wired::set_op(
                        &id,
                        "health",
                        serde_json::json!({ "hp": (i % 100) as i64, "max": 100 }),
                    )])
                    .expect("set op");
            }
        }
        // Small spacing so the server echoes each batch discretely.
        tokio::time::sleep(Duration::from_millis(6)).await;
    }

    // A sentinel spawn fences the stream: the WS is ordered, so once its echo
    // is recorded EVERY prior batch has been recorded AND folded into the view.
    // That makes tap and live view provably in sync (ambient traffic is 0).
    // (A spawn always echoes; a set on an unspawned id would be dropped.)
    let fence_id = "steiner-fence";
    client
        .send_ops(vec![wired::spawn_presence_op(
            fence_id,
            [1.0, 2.0, 1.0],
            0.0,
        )])
        .expect("fence op");

    // --- drain every applied echo into the tap (with its entropy tick).
    //     Record past the fence, then keep going until the stream is idle: the
    //     server emits trailing ops (presence scene-stamps) AFTER a spawn echo,
    //     and the live view folds those too — the tap must capture them or the
    //     two diverge. Fence = lower bound; idle gap = upper bound. ambient=0
    //     guarantees the only traffic is our session's, so a quiet gap means
    //     both view and tap are fully caught up on the identical stream. ---
    let mut recorded = 0usize;
    let mut fenced = false;
    loop {
        let idle = if fenced {
            Duration::from_millis(800)
        } else {
            Duration::from_secs(5)
        };
        match tokio::time::timeout(idle, batches.recv()).await {
            Ok(Ok(batch)) => {
                if batch.ops.iter().any(|op| op_targets(op, fence_id)) {
                    fenced = true;
                }
                tap.record_batch(&batch).expect("journal batch");
                recorded += 1;
            }
            _ if fenced => break, // idle after the fence: fully caught up
            _ => panic!("fence echo never arrived; stream stalled"),
        }
    }
    println!(
        "[live-ordeal] recorded_batches={recorded} entropy_tick={} journal_bytes={}",
        tap.tick(),
        tap.journal_bytes().len()
    );

    // Stream idle and fully recorded → the view folded exactly the same batches.
    let live_hash = world_view_hash(&client.view());
    let tap_hash = tap.state_hash();

    // --- ORDEAL 1: offline replay of (seed, journal) == live ---
    let journal = tap.journal_bytes().to_vec();
    let (replayed, outcome) = Recorder::replay(&journal, None).expect("replay");
    let replay_hash = replayed.state_hash();
    println!(
        "LIVE-REPLAY: live_view=0x{live_hash:016x} tap=0x{tap_hash:016x} replay=0x{replay_hash:016x} outcome={outcome:?}"
    );
    if tap_hash != live_hash {
        let live_state = steiner::live::world_view_state(&client.view());
        let tap_state = tap.recorder().state_map();
        for (id, lc) in &live_state {
            match tap_state.get(id) {
                None => println!(
                    "  DIFF only-in-view id={id} comps={:?}",
                    lc.keys().collect::<Vec<_>>()
                ),
                Some(tc) => {
                    for (comp, lv) in lc {
                        let tv = tc.get(comp);
                        if tv != Some(lv) {
                            println!("  DIFF id={id} comp={comp}\n    view={lv}\n    tap ={tv:?}");
                        }
                    }
                    for comp in tc.keys() {
                        if !lc.contains_key(comp) {
                            println!("  DIFF id={id} comp={comp} only-in-tap");
                        }
                    }
                }
            }
        }
        for id in tap_state.keys() {
            if !live_state.contains_key(id) {
                println!("  DIFF only-in-tap id={id}");
            }
        }
    }
    assert_eq!(outcome, ReadOutcome::Complete, "clean journal");
    assert_eq!(
        tap_hash, live_hash,
        "tap must match the live view it recorded"
    );
    assert_eq!(replay_hash, live_hash, "offline replay must match live");

    // --- ORDEAL 2: any-tick reconstruction ---
    let mid = tap.tick() / 2;
    let (partial, _) = Recorder::replay(&journal, Some(mid)).expect("replay@mid");
    let mid_hash = partial.state_hash();
    println!("ANY-TICK: T={mid} hash=0x{mid_hash:016x} final=0x{replay_hash:016x}");
    assert_ne!(mid_hash, replay_hash, "mid state differs from final");

    // --- ORDEAL 3: disk round-trip (under the worktree, never /tmp) ---
    std::fs::create_dir_all(&out_dir).expect("mkdir out");
    let path = format!("{out_dir}/live_session.steinerj");
    std::fs::write(&path, &journal).expect("write journal");
    let from_disk = std::fs::read(&path).expect("read journal");
    let decoded = read_journal(&from_disk).expect("decode disk journal");
    let (disk_replay, _) = Recorder::replay(&from_disk, None).expect("replay disk");
    println!(
        "DISK: path={path} bytes={} snapshot_hash={:?} entries={} disk_replay=0x{:016x}",
        from_disk.len(),
        decoded.snapshot_hash.map(|h| format!("0x{h:016x}")),
        decoded.entries.len(),
        disk_replay.state_hash()
    );
    assert_eq!(
        disk_replay.state_hash(),
        live_hash,
        "disk replay matches live"
    );
    assert!(decoded.snapshot.is_some(), "v2 journal carries a snapshot");

    // --- ORDEAL 4: torn tail stops cleanly (typed, no panic) ---
    let torn = &journal[..journal.len() - 5];
    let torn_decode = read_journal(torn).expect("torn decode");
    let (torn_replay, torn_outcome) = Recorder::replay(torn, None).expect("torn replay");
    match torn_outcome {
        ReadOutcome::Torn { kind, valid_frames } => {
            assert_eq!(kind, TornKind::Truncated);
            println!(
                "TORN: dropped_bytes=5 kind={kind} valid_frames={valid_frames} entries={} replay=0x{:016x} (clean stop)",
                torn_decode.entries.len(),
                torn_replay.state_hash()
            );
        }
        other => panic!("expected torn tail, got {other:?}"),
    }

    // --- ORDEAL 5: determinism — re-replay is byte-identical ---
    let replay_bytes = replayed.journal_bytes().to_vec();
    println!(
        "DETERMINISM: journal_bytes={} replay==journal:{}",
        journal.len(),
        replay_bytes == journal
    );
    assert_eq!(replay_bytes, journal, "replay reproduces journal bytes");

    client.close().await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    println!("[live-ordeal] done. all ordeals passed.");
}
