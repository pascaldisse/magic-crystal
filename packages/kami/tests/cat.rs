//! RITE V · V2 ordeals — THE PINK CAT'S MIND.
//!
//! The cat's idle loop is a PURE, deterministic function of the world clock:
//! the same tick stream yields a byte-identical drive stream (the ENTROPY law),
//! the loop actually visits all three states in order, and the walk circuit
//! starts and ends at home (continuous into Sit — no teleport).

use kami::{CatMind, CatState};

fn mind() -> CatMind {
    // The realm's cat: home by the ramen stall, small 1.4 m circuit.
    CatMind {
        home: [-5.0, 0.359289, 23.0],
        radius: 1.4,
        ..CatMind::default()
    }
}

const DT: f64 = 1.0 / 60.0; // the renderer's fixed tick dt

/// Ticks spanning slightly more than one full loop (so the wrap is exercised).
fn loop_ticks(m: &CatMind) -> u64 {
    (m.loop_duration() / DT).ceil() as u64 + 5
}

// ---------------------------------------------------------------------------
// Ordeal V2-M1 — DETERMINISM: two independent runs over the full loop produce
// a BYTE-IDENTICAL drive stream (position, yaw, speed, tail, state all covered
// by `CatDrive::to_le_bytes`).
// ---------------------------------------------------------------------------
#[test]
fn cat_drive_stream_is_byte_identical() {
    let m = mind();
    let n = loop_ticks(&m);

    let stream = |mind: &CatMind| -> Vec<u8> {
        let mut bytes = Vec::new();
        for tick in 0..n {
            bytes.extend_from_slice(&mind.drive(tick as f64 * DT).to_le_bytes());
        }
        bytes
    };

    let a = stream(&m);
    let b = stream(&mind()); // freshly parsed mind, same params
    assert_eq!(
        a, b,
        "cat drive stream must be byte-identical across two runs of the full loop"
    );
    println!(
        "[v2-cat-determinism] ticks={n} bytes={} identical=true loop={:.4}s",
        a.len(),
        m.loop_duration()
    );
}

// ---------------------------------------------------------------------------
// Ordeal V2-M2 — the loop VISITS all three states in order Sit → TailFlick →
// Walk within one period, and returns to Sit at the wrap.
// ---------------------------------------------------------------------------
#[test]
fn cat_loop_visits_sit_then_flick_then_walk() {
    let m = mind();

    // Sample the phase boundaries (mid-window of each state).
    let at = |secs: f64| m.drive(secs).state;
    assert_eq!(at(m.sit * 0.5), CatState::Sit, "first: sitting");
    assert_eq!(
        at(m.sit + m.flick * 0.5),
        CatState::TailFlick,
        "then: tail flick"
    );
    assert_eq!(
        at(m.sit + m.flick + m.walk_duration() * 0.5),
        CatState::Walk,
        "then: walking the circuit"
    );
    // One full period later, back to Sit (the loop wraps).
    assert_eq!(
        at(m.loop_duration() + m.sit * 0.5),
        CatState::Sit,
        "wrap: back to sitting"
    );

    // The three states each actually occur over a tick sweep of one loop.
    let n = loop_ticks(&m);
    let mut seen = [false; 3];
    for tick in 0..n {
        seen[m.drive(tick as f64 * DT).state.tag() as usize] = true;
    }
    assert!(
        seen == [true, true, true],
        "all three states must occur in one loop, saw {seen:?}"
    );
    println!("[v2-cat-loop] Sit→TailFlick→Walk all present, wrap returns to Sit");
}

// ---------------------------------------------------------------------------
// Ordeal V2-M3 — the walk circuit STARTS and ENDS at home (continuous into
// Sit), stays within `radius` of home the whole way, and only the Walk phase
// commands a nonzero speed (Sit/TailFlick hold the idle pose).
// ---------------------------------------------------------------------------
#[test]
fn cat_walk_is_bounded_and_returns_home() {
    let m = mind();
    let n = loop_ticks(&m);

    // Walk start (first tick of the Walk phase) is essentially at home; the
    // phase end returns to home.
    let walk_start = m.sit + m.flick;
    let start = m.drive(walk_start + 1e-9).position;
    let d_start = ((start[0] - m.home[0]).powi(2) + (start[2] - m.home[2]).powi(2)).sqrt();
    assert!(d_start < 1e-6, "walk starts at home, off by {d_start}");

    let end = m.drive(m.loop_duration() - 1e-9).position;
    let d_end = ((end[0] - m.home[0]).powi(2) + (end[2] - m.home[2]).powi(2)).sqrt();
    assert!(d_end < 1e-3, "walk ends back at home, off by {d_end}");

    // Bounded: every tick stays within radius (+ f64 slack) of home; speed is
    // zero unless walking.
    for tick in 0..n {
        let d = m.drive(tick as f64 * DT);
        let r = ((d.position[0] - m.home[0]).powi(2) + (d.position[2] - m.home[2]).powi(2)).sqrt();
        assert!(
            r <= m.radius + 1e-9,
            "cat left its {radius} m circuit: r={r} at tick {tick}",
            radius = m.radius
        );
        let moving = d.state == CatState::Walk;
        assert_eq!(
            d.speed > 0.0,
            moving,
            "speed positive iff walking (tick {tick}, state {:?})",
            d.state
        );
    }
    println!(
        "[v2-cat-bounds] circuit r={:.2}m bounded, starts+ends home, speed>0 only when walking",
        m.radius
    );
}
