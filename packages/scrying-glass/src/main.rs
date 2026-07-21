use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use scrying_glass::player::{Ground, Key, Player, PlayerParams};
use scrying_glass::{input, player};

use serde_json::json;
use crystal::{Core, GaiaPackage, ImpulseOp, Op, OpBatch, load_world_dir};
use glam::Vec3;
use scrying_glass::ScryingGlassPackage;
use scrying_glass::bloodbend::{self, Bend, Bloodbend, BloodbendParams};
use scrying_glass::bvh::{Bvh, BvhParams, DynamicSplice, RefitParams};
use scrying_glass::denoiser::deserialize_weights as deserialize_denoiser_weights;
use scrying_glass::denoiser_gpu::GpuDenoiser;
use scrying_glass::integrator::{
    Integrator, IntegratorParams, IntegratorUniform, TemporalParams, resolve as resolve_accum,
    split_aov,
    trace_headless, trace_headless_aov,
};
// NEURAL-LIVE N0.c construction scaffold: the ONE net presented per live frame.
#[cfg(target_os = "macos")]
use scrying_glass::rdirect::ALBEDO_DEMOD_EPS;
#[cfg(target_os = "macos")]
use scrying_glass::rdirect_demod::DemodPass;
#[cfg(target_os = "macos")]
use scrying_glass::rdirect_gather::FeatureGather;
#[cfg(target_os = "macos")]
use scrying_glass::rdirect_live::RdirectLive;
use scrying_glass::scene::{
    Camera, RenderScene, SceneParameters, SunDefaults, SunLight, WalkerPose,
};
use scrying_glass::retina::{self, GeometryCache as RetinaGeometryCache, Layers as RetinaLayers};
use scrying_glass::upscaler::deserialize_weights as deserialize_upscaler_weights;
use scrying_glass::upscaler_gpu::GpuUpscaler;
use steiner::{AppliedBatch, WorldCore, WorldCoreParams, parse_op_batch};
use tauri::{Manager, PhysicalPosition, PhysicalSize, WebviewUrl};

const DEFAULT_NATIVE_PORT: u16 = 8430;
const BYTES_PER_PIXEL: u32 = 4;
const CAPTURE_SLOT_COUNT: usize = 3;

#[derive(Clone)]
struct ScryingGlassConfig {
    window_width: f64,
    window_height: f64,
    /// God's render canvas. Trace, accumulation, temporal, and offscreen
    /// present resources are permanently this size; only the OS surface moves.
    native_canvas_width: u32,
    native_canvas_height: u32,
    panel_width: f64,
    panel_height: f64,
    panel_margin: f64,
    fps: f64,
    native_port: u16,
    title: String,
    auto_test_ipc: bool,
    world_path: PathBuf,
    scene: SceneParameters,
    integrator: IntegratorParams,
    bvh: BvhParams,
    refit: RefitParams,
    /// Accumulation frames a /scry moving-eye capture integrates for a crisp shot.
    capture_frames: u32,
    /// LIGHT-NOT-DOTS: temporal accumulation with reprojection on the live
    /// present path (GAIA_NATIVE_TEMPORAL, default ON). When off the legacy
    /// reset-on-move accum path runs (the escape hatch).
    temporal_enabled: bool,
    temporal: TemporalParams,
    /// FPS COUNTER BURST — HUD toggle (GAIA_NATIVE_HUD, default ON) and its
    /// rolling-median sample window (GAIA_NATIVE_HUD_WINDOW, default 30).
    hud_enabled: bool,
    hud_window: usize,
    /// OWN-BODY CULL override (`GAIA_NATIVE_DRAW_OWN_BODY`, default off/false):
    /// force a walker-attached body (nari) to draw even from its own eye —
    /// the pre-fix behavior, kept as an escape hatch (debugging the vessel
    /// itself, a future third-person mode). Off is the normal weld: her body
    /// vanishes only from the exact eye it is attached to.
    draw_own_body: bool,
    /// Native realm authority: Steiner seed, bounded event view, HTTP limits.
    world_core: WorldCoreParams,
    authority_timeout: Duration,
    event_default_limit: usize,
    event_limit_max: usize,
    max_request_bytes: usize,
    /// WORKER WINDOW (`GAIA_NATIVE_WORKER_WINDOW`, default off/false — Nekromant
    /// case #1 fix): a worker instance's window is built non-activating/
    /// never-key (`focused(false)` + `focusable(false)`, which on macOS rides
    /// tao's `canBecomeKeyWindow`/`canBecomeMainWindow` override down to a
    /// permanent `false` — the portable equivalent of patching NSWindow's
    /// `canBecomeKeyWindow`, no raw objc2 subclassing needed). Such a window
    /// can never accept a keystroke (including Cmd+Q) no matter what GPU-load
    /// activation storm hits the app. Off is the pre-fix behavior (Architect's
    /// live window at :8430 stays exactly as before).
    worker_window: bool,
    /// NEURAL-LIVE N0.c CONSTRUCTION SCAFFOLD (`GAIA_NATIVE_NET_PRESENT`,
    /// default OFF on the branch). When on, the live window loop presents the
    /// ONE net's frame every frame: trace low radiance + native AOV → GPU
    /// feature gather (N0.b) → MPSGraph batched-GEMM forward (N0.a) → undo
    /// log-demod by the native albedo → the existing blit/present path. This
    /// flag DIES at lane cutover — the merged state presents the net
    /// unconditionally, no flag. macOS-only (the MPSGraph net is macOS-only).
    net_present: bool,
    /// WINDOW-BAN OFFSCREEN mode (`GAIA_NATIVE_OFFSCREEN`, default off). When
    /// on, NO NSWindow is ever created: the whole tauri/winit surface path is
    /// skipped, the render loop draws only to the offscreen texture (the
    /// presented eye), and `/scry` serves it over HTTP. The mandated proof
    /// surface — measurement runs never put a window on the Architect's
    /// desktop. Width/height come from the window-size config fields.
    offscreen: bool,
}

impl ScryingGlassConfig {
    fn from_env() -> Result<Self, String> {
        let number = |name: &str, default: f64| -> Result<f64, String> {
            match std::env::var(name) {
                Ok(value) => value
                    .parse::<f64>()
                    .map_err(|_| format!("{name} must be a number, got {value:?}")),
                Err(_) => Ok(default),
            }
        };
        let native_port = match std::env::var("GAIA_NATIVE_PORT") {
            Ok(value) => value
                .parse::<u16>()
                .map_err(|_| format!("GAIA_NATIVE_PORT must be a port, got {value:?}"))?,
            Err(_) => DEFAULT_NATIVE_PORT,
        };
        let integer = |name: &str, default: u32| -> Result<u32, String> {
            match std::env::var(name) {
                Ok(value) => value
                    .parse::<u32>()
                    .map_err(|_| format!("{name} must be an integer, got {value:?}")),
                Err(_) => Ok(default),
            }
        };
        let auto_test_ipc = match std::env::var("SPIKE_AUTOTEST_IPC") {
            Ok(value) => value
                .parse::<bool>()
                .map_err(|_| format!("SPIKE_AUTOTEST_IPC must be true or false, got {value:?}"))?,
            Err(_) => false,
        };
        let hud_enabled = match std::env::var("GAIA_NATIVE_HUD") {
            Ok(value) => value
                .parse::<bool>()
                .map_err(|_| format!("GAIA_NATIVE_HUD must be true or false, got {value:?}"))?,
            Err(_) => true,
        };
        let world_path = std::env::var_os("GAIA_WORLD")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko"));
        // Nekromant case #1 fix: a worker instance never activates/steals
        // focus (item 1 below, applied to the window builder) and defaults to
        // a visibly smaller window + a title suffix so the Architect can tell
        // a worker apart from the one live window at a glance (item 1, param).
        let worker_window = match std::env::var("GAIA_NATIVE_WORKER_WINDOW") {
            Ok(value) => value.parse::<bool>().map_err(|_| {
                format!("GAIA_NATIVE_WORKER_WINDOW must be true or false, got {value:?}")
            })?,
            Err(_) => false,
        };
        let default_window_width = if worker_window { 480.0 } else { 960.0 };
        let default_window_height = if worker_window { 320.0 } else { 640.0 };
        let config = Self {
            window_width: number("GAIA_NATIVE_WIDTH", default_window_width)?,
            window_height: number("GAIA_NATIVE_HEIGHT", default_window_height)?,
            native_canvas_width: integer("GAIA_NATIVE_CANVAS_W", 640)?,
            native_canvas_height: integer("GAIA_NATIVE_CANVAS_H", 480)?,
            panel_width: number("SPIKE_PANEL_WIDTH", 300.0)?,
            panel_height: number("SPIKE_PANEL_HEIGHT", 154.0)?,
            panel_margin: number("SPIKE_PANEL_MARGIN", 24.0)?,
            fps: number("GAIA_NATIVE_FPS", 60.0)?,
            native_port,
            title: {
                let base = std::env::var("GAIA_NATIVE_TITLE")
                    .unwrap_or_else(|_| "GAIA — Scrying Glass".into());
                if worker_window {
                    format!("{base} [worker]")
                } else {
                    base
                }
            },
            auto_test_ipc,
            world_path,
            scene: SceneParameters {
                fov_y_degrees: number("GAIA_NATIVE_FOV", 60.0)? as f32,
                near: number("GAIA_NATIVE_NEAR", 0.1)? as f32,
                far: number("GAIA_NATIVE_FAR", 4_000.0)? as f32,
                sky_top: std::env::var("GAIA_NATIVE_SKY_TOP").unwrap_or_else(|_| "#20152f".into()),
                sky_horizon: std::env::var("GAIA_NATIVE_SKY_HORIZON")
                    .unwrap_or_else(|_| "#9a627d".into()),
                mesh_color: std::env::var("GAIA_NATIVE_MESH_COLOR")
                    .unwrap_or_else(|_| "#9aa0a6".into()),
                radial_segments: integer("GAIA_NATIVE_RADIAL_SEGMENTS", 24)?,
                camera_position: [
                    number("GAIA_NATIVE_CAMERA_X", 0.0)? as f32,
                    number("GAIA_NATIVE_CAMERA_Y", 2.0)? as f32,
                    number("GAIA_NATIVE_CAMERA_Z", 22.0)? as f32,
                ],
                camera_yaw: number("GAIA_NATIVE_CAMERA_YAW", 0.0)? as f32,
                camera_pitch: number("GAIA_NATIVE_CAMERA_PITCH", 0.0)? as f32,
                tick_dt: number("GAIA_NATIVE_TICK_DT", 1.0 / 60.0)?,
                sun: SunDefaults {
                    sun_color: std::env::var("GAIA_NATIVE_SUN_COLOR")
                        .unwrap_or_else(|_| "#ffe2b0".into()),
                    sun_intensity: number("GAIA_NATIVE_SUN_INTENSITY", 1.1)? as f32,
                    sun_position: [
                        number("GAIA_NATIVE_SUN_X", 60.0)? as f32,
                        number("GAIA_NATIVE_SUN_Y", 90.0)? as f32,
                        number("GAIA_NATIVE_SUN_Z", 30.0)? as f32,
                    ],
                    ambient_intensity: number("GAIA_NATIVE_AMBIENT_INTENSITY", 0.32)? as f32,
                },
                emission_intensity: number("GAIA_NATIVE_EMISSIVE_INTENSITY", 2.5)? as f32,
            },
            integrator: IntegratorParams {
                spp: integer("GAIA_NATIVE_SPP", 2)?,
                max_bounces: integer("GAIA_NATIVE_MAX_BOUNCES", 4)?,
                rr_start: integer("GAIA_NATIVE_RR_START", 2)?,
                seed: integer("GAIA_NATIVE_SEED", 0x5eed)?,
                eps: number("GAIA_NATIVE_RAY_EPS", 1e-3)? as f32,
            },
            bvh: BvhParams {
                leaf_max: integer("GAIA_NATIVE_BVH_LEAF", 4)? as usize,
                max_depth: integer("GAIA_NATIVE_BVH_DEPTH", 64)? as usize,
                sah_bins: integer("GAIA_NATIVE_BVH_SAH_BINS", 16)? as usize,
            },
            refit: RefitParams {
                degrade_ratio: number(
                    "GAIA_NATIVE_BVH_REFIT_DEGRADE",
                    RefitParams::default().degrade_ratio as f64,
                )? as f32,
                max_refits: integer("GAIA_NATIVE_BVH_REFIT_MAX", 0)?,
            },
            capture_frames: integer("GAIA_NATIVE_CAPTURE_FRAMES", 48)?,
            temporal_enabled: match std::env::var("GAIA_NATIVE_TEMPORAL") {
                Ok(value) => value.parse::<bool>().map_err(|_| {
                    format!("GAIA_NATIVE_TEMPORAL must be true or false, got {value:?}")
                })?,
                // THE DESIGN IS THE LAW (Architect, 07-18): the shipped present
                // path is trace → THE NET → screen and nothing else. The temporal
                // accumulation machinery is LAB EQUIPMENT (training ground-truth
                // generator + history-buffer substrate) — default OFF; its
                // hand-heuristics (gates/clamps/thresholds) never ship. Until the
                // net lands in the present path, the window shows the one
                // integrator's young samples — the truth, not a stand-in.
                Err(_) => false,
            },
            temporal: TemporalParams {
                alpha_min: number("GAIA_NATIVE_TEMPORAL_ALPHA_MIN", 0.1)? as f32,
                depth_tol: number("GAIA_NATIVE_TEMPORAL_DEPTH_TOL", 0.05)? as f32,
                normal_tol: number("GAIA_NATIVE_TEMPORAL_NORMAL_TOL", 0.85)? as f32,
                clamp_k: number("GAIA_NATIVE_TEMPORAL_CLAMP_K", 1.5)? as f32,
                max_history: integer("GAIA_NATIVE_TEMPORAL_MAX_HISTORY", 512)?,
                still_px: number("GAIA_NATIVE_TEMPORAL_STILL_PX", 0.05)? as f32,
            },
            hud_enabled,
            hud_window: integer("GAIA_NATIVE_HUD_WINDOW", 30)? as usize,
            draw_own_body: match std::env::var("GAIA_NATIVE_DRAW_OWN_BODY") {
                Ok(value) => value.parse::<bool>().map_err(|_| {
                    format!("GAIA_NATIVE_DRAW_OWN_BODY must be true or false, got {value:?}")
                })?,
                Err(_) => false,
            },
            world_core: WorldCoreParams {
                seed: integer("GAIA_NATIVE_WORLD_SEED", 0x5eed)? as u64,
                event_capacity: integer("GAIA_NATIVE_EVENT_CAPACITY", 2_000)? as usize,
                ..WorldCoreParams::default()
            },
            authority_timeout: Duration::from_millis(u64::from(integer(
                "GAIA_NATIVE_OP_TIMEOUT_MS",
                5_000,
            )?)),
            event_default_limit: integer("GAIA_NATIVE_EVENT_DEFAULT_LIMIT", 200)? as usize,
            event_limit_max: integer("GAIA_NATIVE_EVENT_LIMIT_MAX", 500)? as usize,
            max_request_bytes: integer("GAIA_NATIVE_HTTP_MAX_BYTES", 1 << 20)? as usize,
            worker_window,
            net_present: match std::env::var("GAIA_NATIVE_NET_PRESENT") {
                Ok(value) => value.parse::<bool>().map_err(|_| {
                    format!("GAIA_NATIVE_NET_PRESENT must be true or false, got {value:?}")
                })?,
                Err(_) => false,
            },
            offscreen: match std::env::var("GAIA_NATIVE_OFFSCREEN") {
                Ok(value) => value.parse::<bool>().map_err(|_| {
                    format!("GAIA_NATIVE_OFFSCREEN must be true or false, got {value:?}")
                })?,
                Err(_) => false,
            },
        };
        if config.window_width <= 0.0
            || config.window_height <= 0.0
            || config.panel_width <= 0.0
            || config.panel_height <= 0.0
            || config.panel_margin < 0.0
            || config.fps <= 0.0
            || config.native_port == 0
            || config.native_canvas_width == 0
            || config.native_canvas_height == 0
            || config.world_core.event_capacity == 0
            || config.authority_timeout.is_zero()
            || config.event_default_limit == 0
            || config.event_limit_max == 0
            || config.max_request_bytes == 0
        {
            return Err(
                "window/canvas dimensions, FPS, port, authority/event/request limits must be positive (margin may be zero)"
                    .into(),
            );
        }
        if config.panel_width + config.panel_margin > config.window_width
            || config.panel_height + config.panel_margin > config.window_height
        {
            return Err("overlay panel plus SPIKE_PANEL_MARGIN must fit in the window".into());
        }
        Ok(config)
    }

    fn frame_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.fps)
    }

    fn panel_layout(&self, size: PhysicalSize<u32>) -> (PhysicalPosition<f64>, PhysicalSize<u32>) {
        let width = f64::from(size.width);
        let position = PhysicalPosition::new(
            (width - self.panel_width - self.panel_margin).max(0.0),
            self.panel_margin,
        );
        let panel = PhysicalSize::new(
            self.panel_width.round() as u32,
            self.panel_height.round() as u32,
        );
        (position, panel)
    }

    fn is_panel_point(&self, x: f64, y: f64, size: PhysicalSize<u32>) -> bool {
        let width = f64::from(size.width);
        let height = f64::from(size.height);
        let left = width - self.panel_width - self.panel_margin;
        let bottom = height - self.panel_height - self.panel_margin;
        x >= left
            && x <= width - self.panel_margin
            && y >= bottom
            && y <= height - self.panel_margin
    }
}

#[derive(Clone, Copy)]
enum PixelOrder {
    Rgba,
    Bgra,
}

struct CapturedFrame {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

type LatestFrame = Arc<RwLock<Option<Arc<CapturedFrame>>>>;

/// S12.5 AI DEBUG DOOR: the latest per-stage budget + forward-state JSON, kept
/// fresh by the render loop and served by `/budget` and `/state`.
#[derive(Default)]
struct DebugSnapshot {
    budget: String,
    state: String,
}
type DebugCell = Arc<RwLock<DebugSnapshot>>;

struct CaptureReady {
    result: Result<(), String>,
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    pixel_order: PixelOrder,
    busy: Arc<AtomicBool>,
}

fn spawn_capture_worker(latest: LatestFrame) -> mpsc::Sender<CaptureReady> {
    let (sender, receiver) = mpsc::channel::<CaptureReady>();
    thread::Builder::new()
        .name("gaia-frame-capture".into())
        .spawn(move || {
            while let Ok(capture) = receiver.recv() {
                if let Err(error) = &capture.result {
                    eprintln!("[screenshot] framebuffer map failed: {error}");
                    capture.busy.store(false, Ordering::Release);
                    continue;
                }
                let row_bytes = (capture.width * BYTES_PER_PIXEL) as usize;
                let mapped = match capture.buffer.get_mapped_range(..) {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        eprintln!("[screenshot] mapped framebuffer unavailable: {error}");
                        capture.buffer.unmap();
                        capture.busy.store(false, Ordering::Release);
                        continue;
                    }
                };
                let mut rgba = Vec::with_capacity(row_bytes * capture.height as usize);
                for row in mapped
                    .chunks(capture.padded_bytes_per_row as usize)
                    .take(capture.height as usize)
                {
                    rgba.extend_from_slice(&row[..row_bytes]);
                }
                if matches!(capture.pixel_order, PixelOrder::Bgra) {
                    for pixel in rgba.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
                        pixel.swap(0, 2);
                    }
                }
                drop(mapped);
                capture.buffer.unmap();
                if let Ok(mut frame) = latest.write() {
                    *frame = Some(Arc::new(CapturedFrame {
                        width: capture.width,
                        height: capture.height,
                        rgba,
                    }));
                }
                capture.busy.store(false, Ordering::Release);
            }
        })
        .expect("spawn framebuffer capture worker");
    sender
}

fn encode_png(frame: &CapturedFrame) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, frame.width, frame.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
        writer
            .write_image_data(&frame.rgba)
            .map_err(|error| error.to_string())?;
    }
    Ok(bytes)
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    extra_headers: &str,
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nCache-Control: no-store\r\nConnection: close\r\n{extra_headers}\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn respond_json(stream: &mut TcpStream, status: &str, value: serde_json::Value) {
    match serde_json::to_vec_pretty(&value) {
        Ok(body) => {
            let _ = write_response(
                stream,
                status,
                "application/json; charset=utf-8",
                &body,
                "",
            );
        }
        Err(error) => {
            let _ = write_response(
                stream,
                "500 Internal Server Error",
                "application/json; charset=utf-8",
                json!({ "ok": false, "error": error.to_string() })
                    .to_string()
                    .as_bytes(),
                "",
            );
        }
    }
}

fn respond_frame(stream: &mut TcpStream, frame: &CapturedFrame) {
    match encode_png(frame) {
        Ok(png) => {
            let dimensions = format!("X-GAIA-Framebuffer: {}x{}\r\n", frame.width, frame.height);
            let _ = write_response(stream, "200 OK", "image/png", &png, &dimensions);
        }
        Err(error) => {
            let _ = write_response(
                stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                error.as_bytes(),
                "",
            );
        }
    }
}

enum WorldRequest {
    Apply {
        batch: OpBatch,
        reply: mpsc::Sender<Result<AppliedBatch, String>>,
    },
    Snapshot {
        reply: mpsc::Sender<Result<serde_json::Value, String>>,
    },
    Events {
        since: u64,
        limit: usize,
        reply: mpsc::Sender<serde_json::Value>,
    },
}

/// HTTP ↔ render-authority channels + embodied debug state.
struct HttpContext {
    latest: LatestFrame,
    scry: mpsc::Sender<RenderRequest>,
    world: mpsc::Sender<WorldRequest>,
    authority_timeout: Duration,
    event_default_limit: usize,
    event_limit_max: usize,
    max_request_bytes: usize,
    player: Arc<Mutex<Player>>,
    ground: Arc<Ground>,
    tick_dt: f32,
    /// S12.5: live budget/state JSON for `/budget` and `/state`.
    debug: DebugCell,
}

/// Read one bounded HTTP request, including its Content-Length body.
fn read_request(stream: &mut TcpStream, max_bytes: usize) -> Option<(String, String)> {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break index + 4;
        }
        if buffer.len() > max_bytes {
            return None;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    if header_end.checked_add(content_length)? > max_bytes {
        return None;
    }
    while buffer.len() < header_end + content_length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = String::from_utf8_lossy(
        &buffer[header_end..header_end + content_length.min(buffer.len() - header_end)],
    )
    .into_owned();
    Some((headers, body))
}

fn handle_http(mut stream: TcpStream, ctx: &HttpContext) {
    let latest = &ctx.latest;
    let scry = &ctx.scry;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let Some((headers, body)) = read_request(&mut stream, ctx.max_request_bytes) else {
        return;
    };
    let first_line = headers.lines().next().unwrap_or_default().to_owned();
    let mut tokens = first_line.split_whitespace();
    let method = tokens.next().unwrap_or_default();
    let target = tokens.next().unwrap_or_default();
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    if method == "OPTIONS" {
        let _ = write_response(
            &mut stream,
            "204 No Content",
            "text/plain; charset=utf-8",
            b"",
            "Allow: GET, POST, OPTIONS\r\n",
        );
        return;
    }
    if path == "/op" && method == "POST" {
        let value = match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(value) => value,
            Err(error) => {
                respond_json(
                    &mut stream,
                    "400 Bad Request",
                    json!({ "ok": false, "error": format!("op body must be JSON: {error}") }),
                );
                return;
            }
        };
        let batch = match parse_op_batch(value) {
            Ok(batch) => batch,
            Err(error) => {
                respond_json(
                    &mut stream,
                    "400 Bad Request",
                    json!({ "ok": false, "error": error }),
                );
                return;
            }
        };
        let (reply, receive) = mpsc::channel();
        if ctx
            .world
            .send(WorldRequest::Apply { batch, reply })
            .is_err()
        {
            respond_json(
                &mut stream,
                "503 Service Unavailable",
                json!({ "ok": false, "error": "world authority unavailable" }),
            );
            return;
        }
        match receive.recv_timeout(ctx.authority_timeout) {
            Ok(Ok(report)) => respond_json(
                &mut stream,
                "200 OK",
                json!({
                    "ok": true,
                    "applied": report.applied,
                    "entropy": report.entropy,
                    "latest": report.latest,
                }),
            ),
            Ok(Err(error)) => respond_json(
                &mut stream,
                "400 Bad Request",
                json!({ "ok": false, "error": error }),
            ),
            Err(_) => respond_json(
                &mut stream,
                "504 Gateway Timeout",
                json!({ "ok": false, "error": "world authority timed out" }),
            ),
        }
        return;
    }
    if path == "/world" && method == "GET" {
        let (reply, receive) = mpsc::channel();
        if ctx.world.send(WorldRequest::Snapshot { reply }).is_err() {
            respond_json(
                &mut stream,
                "503 Service Unavailable",
                json!({ "ok": false, "error": "world authority unavailable" }),
            );
            return;
        }
        match receive.recv_timeout(ctx.authority_timeout) {
            Ok(Ok(snapshot)) => respond_json(&mut stream, "200 OK", snapshot),
            Ok(Err(error)) => respond_json(
                &mut stream,
                "500 Internal Server Error",
                json!({ "ok": false, "error": error }),
            ),
            Err(_) => respond_json(
                &mut stream,
                "504 Gateway Timeout",
                json!({ "ok": false, "error": "world authority timed out" }),
            ),
        }
        return;
    }
    if path == "/events" && method == "GET" {
        let query_value = |key: &str| {
            query.split('&').find_map(|pair| {
                let (name, value) = pair.split_once('=')?;
                (name == key).then_some(value)
            })
        };
        let since = query_value("since")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let limit = query_value("limit")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(ctx.event_default_limit)
            .min(ctx.event_limit_max);
        let (reply, receive) = mpsc::channel();
        if ctx
            .world
            .send(WorldRequest::Events {
                since,
                limit,
                reply,
            })
            .is_err()
        {
            respond_json(
                &mut stream,
                "503 Service Unavailable",
                json!({ "ok": false, "error": "world authority unavailable" }),
            );
            return;
        }
        match receive.recv_timeout(ctx.authority_timeout) {
            Ok(events) => respond_json(&mut stream, "200 OK", events),
            Err(_) => respond_json(
                &mut stream,
                "504 Gateway Timeout",
                json!({ "ok": false, "error": "world authority timed out" }),
            ),
        }
        return;
    }

    // Embodiment debug organs (param-gated by their presence, no keyboard needed):
    // GET /pose returns the body's eye pose; POST /walk injects held keys for N ticks.
    if path == "/pose" && method == "GET" {
        respond_pose(&mut stream, ctx);
        return;
    }
    if path == "/walk" && method == "POST" {
        respond_walk(&mut stream, ctx, &body);
        return;
    }
    if path == "/push" && method == "POST" {
        respond_push(&mut stream, ctx, &body);
        return;
    }
    // S12.5 AI DEBUG DOOR: live per-stage budget + forward state as JSON.
    if path == "/budget" && method == "GET" {
        let json = ctx
            .debug
            .read()
            .ok()
            .map(|d| d.budget.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "{\"frames\":0}".to_string());
        let _ = write_response(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            json.as_bytes(),
            "",
        );
        return;
    }
    if path == "/state" && method == "GET" {
        let json = ctx
            .debug
            .read()
            .ok()
            .map(|d| d.state.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "{\"note\":\"warming up\"}".to_string());
        let _ = write_response(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            json.as_bytes(),
            "",
        );
        return;
    }

    if method != "GET" {
        let _ = write_response(
            &mut stream,
            "405 Method Not Allowed",
            "text/plain; charset=utf-8",
            b"method not allowed\n",
            "Allow: GET, POST\r\n",
        );
        return;
    }
    // GET /scry — the true name (GRIMOIRE: a screenshot is a scrying).
    // GET /screenshot is kept as an alias for tool compatibility.
    if path == "/retina" {
        let params = match parse_retina_query(query) {
            Ok(params) => params,
            Err(error) => { let _ = write_response(&mut stream, "400 Bad Request", "text/plain; charset=utf-8", error.as_bytes(), ""); return; }
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if scry.send(RenderRequest::Retina { params, reply: reply_tx }).is_err() {
            let _ = write_response(&mut stream, "503 Service Unavailable", "text/plain; charset=utf-8", b"render thread unavailable\n", "");
            return;
        }
        match reply_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(json)) => { let _ = write_response(&mut stream, "200 OK", "application/json; charset=utf-8", json.as_bytes(), ""); }
            Ok(Err(error)) => { let _ = write_response(&mut stream, "500 Internal Server Error", "text/plain; charset=utf-8", error.as_bytes(), ""); }
            Err(_) => { let _ = write_response(&mut stream, "504 Gateway Timeout", "text/plain; charset=utf-8", b"retina trace timed out\n", ""); }
        }
        return;
    }
    if path != "/scry" && path != "/screenshot" {
        let _ = write_response(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
            "",
        );
        return;
    }

    // Parse the query (pose overrides and/or the S12.5 eye selector). An empty
    // query yields the default params (the bare live-frame request).
    let mut params = match parse_scry_query(query) {
        Ok(params) => params,
        Err(error) => {
            let _ = write_response(
                &mut stream,
                "400 Bad Request",
                "text/plain; charset=utf-8",
                error.as_bytes(),
                "",
            );
            return;
        }
    };
    // S12.5: `eye=presented` alone (no pose/resolve/size override) serves the
    // live net-present frame — same as an empty query. Only the belief eye and
    // real pose captures round-trip to the render thread.
    let no_capture_override = params.pos.is_none()
        && params.yaw.is_none()
        && params.pitch.is_none()
        && params.fov.is_none()
        && params.resolve.is_none()
        && params.width.is_none()
        && params.height.is_none();
    // N0.j S13.2 ON-DEMAND READBACK: a bare `/scry` (or `eye=presented` with no
    // pose override) is served by reading the CURRENT offscreen texture back to
    // the CPU on the render thread ONLY when asked — the per-frame readback that
    // fed `latest` is gone. Flag the request so the loop calls
    // `capture_presented`. `latest` stays a fallback under the A/B toggle
    // (`GAIA_NATIVE_PERFRAME_READBACK=1`), read first when populated.
    if !params.belief && no_capture_override {
        if let Some(frame) = latest.read().ok().and_then(|frame| frame.clone()) {
            respond_frame(&mut stream, &frame);
            return;
        }
        params.presented = true;
    }
    let (reply_tx, reply_rx) = mpsc::channel();
    if scry
        .send(RenderRequest::Scry(ScryRequest {
            params,
            reply: reply_tx,
        }))
        .is_err()
    {
        let _ = write_response(
            &mut stream,
            "503 Service Unavailable",
            "text/plain; charset=utf-8",
            b"render thread unavailable\n",
            "",
        );
        return;
    }
    match reply_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(frame)) => respond_frame(&mut stream, &frame),
        Ok(Err(error)) => {
            let _ = write_response(
                &mut stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                error.as_bytes(),
                "",
            );
        }
        Err(_) => {
            let _ = write_response(
                &mut stream,
                "504 Gateway Timeout",
                "text/plain; charset=utf-8",
                b"scry render timed out\n",
                "",
            );
        }
    }
}

/// Format a pose as the `/pose` — and `/walk` stream — JSON object.
fn pose_json(pose: &player::Pose) -> String {
    format!(
        "{{\"position\":[{},{},{}],\"yaw\":{},\"pitch\":{},\"eyeHeight\":{},\"feetY\":{},\"grounded\":{},\"vy\":{}}}",
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.yaw,
        pose.pitch,
        pose.eye_height,
        pose.position.y - pose.eye_height,
        pose.grounded,
        pose.vy,
    )
}

/// GET /pose — the body's current eye pose (debug organ).
fn respond_pose(stream: &mut TcpStream, ctx: &HttpContext) {
    let pose = match ctx.player.lock() {
        Ok(player) => player.pose(),
        Err(_) => {
            let _ = write_response(
                stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                b"player state poisoned\n",
                "",
            );
            return;
        }
    };
    let _ = write_response(
        stream,
        "200 OK",
        "application/json; charset=utf-8",
        pose_json(&pose).as_bytes(),
        "",
    );
}

/// POST /walk — inject held keys for N deterministic ticks (debug organ). Body
/// is `{\"keys\":[...], \"yaw\"?, \"pitch\"?, \"ticks\"?}`. Returns the final pose
/// plus the full per-tick pose stream so play-tests read exactly what moved.
fn respond_walk(stream: &mut TcpStream, ctx: &HttpContext, body: &str) {
    let request: serde_json::Value = match serde_json::from_str(body.trim()) {
        Ok(value) => value,
        Err(error) => {
            let _ = write_response(
                stream,
                "400 Bad Request",
                "text/plain; charset=utf-8",
                format!("walk body must be JSON: {error}").as_bytes(),
                "",
            );
            return;
        }
    };
    let ticks = request
        .get("ticks")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1)
        .min(100_000) as u32;
    let keys: std::collections::HashSet<Key> = request
        .get("keys")
        .and_then(serde_json::Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|item| item.as_str().and_then(Key::from_token))
                .collect()
        })
        .unwrap_or_default();
    let yaw = request.get("yaw").and_then(serde_json::Value::as_f64);
    let pitch = request.get("pitch").and_then(serde_json::Value::as_f64);

    let mut player = match ctx.player.lock() {
        Ok(player) => player,
        Err(_) => {
            let _ = write_response(
                stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                b"player state poisoned\n",
                "",
            );
            return;
        }
    };
    if let Some(yaw) = yaw {
        player.yaw = yaw as f32;
    }
    if let Some(pitch) = pitch {
        player.pitch = (pitch as f32).clamp(-player.params.pitch_limit, player.params.pitch_limit);
    }
    player.keys = keys;
    let mut poses = Vec::with_capacity(ticks as usize);
    for _ in 0..ticks {
        player.step(ctx.tick_dt, &ctx.ground);
        poses.push(pose_json(&player.pose()));
    }
    // Injected keys are transient: clear them so the render loop doesn't keep
    // walking after the organ returns.
    player.keys.clear();
    let final_pose = pose_json(&player.pose());
    drop(player);

    let body = format!(
        "{{\"ticks\":{ticks},\"pose\":{final_pose},\"stream\":[{}]}}",
        poses.join(",")
    );
    let _ = write_response(
        stream,
        "200 OK",
        "application/json; charset=utf-8",
        body.as_bytes(),
        "",
    );
}

/// POST /push — fire ONE push from the current view ray, the exact keyboard
/// path without a keyboard: flip the shared player's `push_pending` flag (the
/// same flag the F key and a pointer-locked click set) so the render loop
/// casts the ray, picks the nearest aimed-at body, and shoves it with an
/// `Op::Impulse` on its next tick. Optional body `{yaw?, pitch?}` aims first.
fn respond_push(stream: &mut TcpStream, ctx: &HttpContext, body: &str) {
    let request: serde_json::Value = serde_json::from_str(body.trim()).unwrap_or(serde_json::Value::Null);
    let yaw = request.get("yaw").and_then(serde_json::Value::as_f64);
    let pitch = request.get("pitch").and_then(serde_json::Value::as_f64);
    let mut player = match ctx.player.lock() {
        Ok(player) => player,
        Err(_) => {
            let _ = write_response(
                stream,
                "500 Internal Server Error",
                "text/plain; charset=utf-8",
                b"player state poisoned\n",
                "",
            );
            return;
        }
    };
    if let Some(yaw) = yaw {
        player.yaw = yaw as f32;
    }
    if let Some(pitch) = pitch {
        player.pitch = (pitch as f32).clamp(-player.params.pitch_limit, player.params.pitch_limit);
    }
    player.push_pending = true;
    let pose = pose_json(&player.pose());
    drop(player);
    let body = format!("{{\"pushed\":true,\"pose\":{pose}}}");
    let _ = write_response(
        stream,
        "200 OK",
        "application/json; charset=utf-8",
        body.as_bytes(),
        "",
    );
}

fn start_screenshot_server(port: u16, ctx: HttpContext) -> Result<(), String> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))
        .map_err(|error| format!("bind GAIA_NATIVE_PORT {port}: {error}"))?;
    thread::Builder::new()
        .name("gaia-native-http".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => handle_http(stream, &ctx),
                    Err(error) => eprintln!("[scry] HTTP accept failed: {error}"),
                }
            }
        })
        .map_err(|error| format!("spawn scrying HTTP server: {error}"))?;
    eprintln!(
        "[scry] GET http://127.0.0.1:{port}/scry (alias: /screenshot; optional pos/yaw/pitch/fov/w/h; lab-only chain: lab=teacher-benchmark)"
    );
    eprintln!(
        "[world-core] POST http://127.0.0.1:{port}/op · GET /world · GET /events"
    );
    eprintln!(
        "[embodiment] GET http://127.0.0.1:{port}/pose · POST http://127.0.0.1:{port}/walk {{keys,yaw?,pitch?,ticks?}}"
    );
    Ok(())
}

struct CaptureSlot {
    buffer: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}

struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    slots: Vec<CaptureSlot>,
    next_slot: usize,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
}

impl OffscreenTarget {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot framebuffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let unpadded = width * BYTES_PER_PIXEL;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(alignment) * alignment;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);
        let slots = (0..CAPTURE_SLOT_COUNT)
            .map(|index| CaptureSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(match index {
                        0 => "frame readback 0",
                        1 => "frame readback 1",
                        _ => "frame readback 2",
                    }),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                busy: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Self {
            texture,
            view,
            slots,
            next_slot: 0,
            padded_bytes_per_row,
            width,
            height,
        }
    }

    fn claim_slot(&mut self) -> Option<usize> {
        for offset in 0..self.slots.len() {
            let index = (self.next_slot + offset) % self.slots.len();
            if self.slots[index]
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.next_slot = (index + 1) % self.slots.len();
                return Some(index);
            }
        }
        None
    }
}

/// NEURAL-LIVE N0.j S13 THE OUTSIDE-9ms HUNT — the frame-loop work that lives
/// OUTSIDE the per-stage net budget (N0.i named ~9 ms of it): world advance
/// (skin·tick·splice + BVH re-upload), the per-frame offscreen readback that
/// feeds `/scry`, and the http/debug servicing on the render thread. Measured
/// in the render loop (not `NetPresent`, which only sees the GPU stages) and
/// merged into `/budget` so the throughput gap is finally VISIBLE, not implied.
/// NEURAL-LIVE S14 (shift 14): the sub-breakdown INSIDE the ~7 ms world
/// advance — one timer per stage of `advance_world`, so the thief is split
/// (skin·tick vs gather vs splice vs upload) before it is cut. Filled by
/// `advance_world` into `Renderer::last_world_stages`, drained by the loop.
#[derive(Default, Clone, Copy)]
struct WorldStages {
    /// `command_bodies_walked` + `tick_with_ops` (skin the bodies, advance solver).
    skin: f64,
    /// `command_bodies_walked` alone (SAMA gait + re-skin the body meshes).
    command: f64,
    /// `tick_with_ops` alone (dynamics solver step + op application).
    tick: f64,
    /// S15 sub-split of `tick`: KAMI decorative eval / apply ops / physics.step
    /// / re-derive models (`scene.last_tick_breakdown`).
    kami: f64,
    apply: f64,
    physics: f64,
    rederive: f64,
    solver_step: f64,
    poll: f64,
    /// `dynamic_leaf_triangles_for_eye` (gather the dynamic partition's tris).
    gather: f64,
    /// `splice.update` — dynamic refit/rebuild + CPU merge onto the static tree.
    splice: f64,
    /// `integrator.update_bvh` — (re)build the GPU node/tri buffers.
    upload: f64,
}

#[derive(Default)]
struct OutsideBudget {
    /// player.step + set_view_pose + advance_world (skin·tick·splice·upload).
    world: Vec<f64>,
    /// S14 sub-breakdown of `world` (skin·tick / gather / splice / upload).
    w_skin: Vec<f64>,
    w_command: Vec<f64>,
    w_tick: Vec<f64>,
    w_kami: Vec<f64>,
    w_apply: Vec<f64>,
    w_physics: Vec<f64>,
    w_rederive: Vec<f64>,
    w_solver_step: Vec<f64>,
    w_poll: Vec<f64>,
    w_gather: Vec<f64>,
    w_splice: Vec<f64>,
    w_upload: Vec<f64>,
    /// the per-frame offscreen copy_texture_to_buffer + map submit (the
    /// measurement tax — S13.2 makes it on-demand, so this collapses to ~0).
    readback: Vec<f64>,
    /// scry drain + the /budget + /state JSON write on the render thread.
    http: Vec<f64>,
    /// the whole iteration wall (frame-start to frame-start), sans deadline
    /// sleep — the honest per-frame cost the wall-clock fps derives from.
    loop_total: Vec<f64>,
    frames: u64,
}

impl OutsideBudget {
    fn record(&mut self, world: f64, readback: f64, http: f64, loop_total: f64) {
        self.world.push(world);
        self.readback.push(readback);
        self.http.push(http);
        self.loop_total.push(loop_total);
        self.frames += 1;
    }

    /// S14: record the sub-stage breakdown of the frame's world advance.
    fn record_world(&mut self, s: WorldStages) {
        self.w_skin.push(s.skin);
        self.w_command.push(s.command);
        self.w_tick.push(s.tick);
        self.w_kami.push(s.kami);
        self.w_apply.push(s.apply);
        self.w_physics.push(s.physics);
        self.w_rederive.push(s.rederive);
        self.w_solver_step.push(s.solver_step);
        self.w_poll.push(s.poll);
        self.w_gather.push(s.gather);
        self.w_splice.push(s.splice);
        self.w_upload.push(s.upload);
    }

    /// The `"outside"` block spliced into `/budget` (median/p95 per segment).
    fn json(&self) -> String {
        format!(
            "\"outside\":{{\"world\":[{:.3},{:.3}],\"readback\":[{:.3},{:.3}],\
             \"http\":[{:.3},{:.3}],\"loop_total\":[{:.3},{:.3}]}}",
            pct(&self.world, 0.5), pct(&self.world, 0.95),
            pct(&self.readback, 0.5), pct(&self.readback, 0.95),
            pct(&self.http, 0.5), pct(&self.http, 0.95),
            pct(&self.loop_total, 0.5), pct(&self.loop_total, 0.95),
        )
    }

    /// S14: the `"world_stages"` block (median/p95 per advance sub-stage).
    fn world_stages_json(&self) -> String {
        format!(
            "\"world_stages\":{{\"skin\":[{:.3},{:.3}],\"command\":[{:.3},{:.3}],\
             \"tick\":[{:.3},{:.3}],\"kami\":[{:.3},{:.3}],\"apply\":[{:.3},{:.3}],\
             \"physics\":[{:.3},{:.3}],\"rederive\":[{:.3},{:.3}],\
             \"solver_step\":[{:.3},{:.3}],\"poll\":[{:.3},{:.3}],\"gather\":[{:.3},{:.3}],\
             \"splice\":[{:.3},{:.3}],\"upload\":[{:.3},{:.3}]}}",
            pct(&self.w_skin, 0.5), pct(&self.w_skin, 0.95),
            pct(&self.w_command, 0.5), pct(&self.w_command, 0.95),
            pct(&self.w_tick, 0.5), pct(&self.w_tick, 0.95),
            pct(&self.w_kami, 0.5), pct(&self.w_kami, 0.95),
            pct(&self.w_apply, 0.5), pct(&self.w_apply, 0.95),
            pct(&self.w_physics, 0.5), pct(&self.w_physics, 0.95),
            pct(&self.w_rederive, 0.5), pct(&self.w_rederive, 0.95),
            pct(&self.w_solver_step, 0.5), pct(&self.w_solver_step, 0.95),
            pct(&self.w_poll, 0.5), pct(&self.w_poll, 0.95),
            pct(&self.w_gather, 0.5), pct(&self.w_gather, 0.95),
            pct(&self.w_splice, 0.5), pct(&self.w_splice, 0.95),
            pct(&self.w_upload, 0.5), pct(&self.w_upload, 0.95),
        )
    }
}

/// NEURAL-LIVE N0.c per-frame stage budget (ms). One frame's cost split across
/// the pipeline stages the budget table (N0.d) reports.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct NetTimings {
    trace: f64,
    gather: f64,
    net: f64,
    /// CUT 2 GPU demod pass (undo-log-demod), split from the surface blit so
    /// the budget table can name it separately (S3).
    demod: f64,
    present: f64,
    total: f64,
}

/// NEURAL-LIVE N0.c CONSTRUCTION SCAFFOLD (dies at lane cutover). Presents the
/// ONE net's frame in the live window loop. EVERYTHING is pooled ONCE here at
/// construction — the low-res radiance accum, the native-res AOV, the AOV
/// readback stage, the net's zero-copy feature/output MTLBuffers (inside
/// `RdirectLive`), the surface-res present accum, and the CPU upload scratch.
/// The per-frame path allocates nothing on the heap except the forward's
/// output `Vec` (owned by N0.a's `forward_shared`, out of this shift's scope)
/// and two lightweight compute bind groups (the integrator reallocates its
/// node/tri storage buffers each dynamic tick, so bind groups over them MUST
/// be rebuilt per frame — wgpu bind groups are handle-weight, not buffer
/// churn). Sized to the boot surface; the net path self-disables if the
/// surface ever exceeds the pooled ceiling (resize rebuild is out of scope
/// for a scaffold that dies at cutover).
#[cfg(target_os = "macos")]
struct NetPresent {
    live: RdirectLive,
    gather: FeatureGather,
    /// CUT 2: GPU demod pass (undo-log-demod on the GPU, no CPU round-trip).
    demod: DemodPass,
    /// Low-res noisy radiance accum (STORAGE|COPY_SRC|COPY_DST — cleared each
    /// frame so a moving camera never smears progressive samples).
    net_accum: wgpu::Buffer,
    /// Native-res AOV G-buffer (albedo/normal/depth, 2 cells/px). N0.i S13:
    /// ONE PER SET — the frame overlap demods the PREVIOUS frame's net output,
    /// so its albedo must be that frame's, not the one trace just wrote. Trace
    /// writes `net_aov[set]`, gather reads `net_aov[set]`, demod reads
    /// `net_aov[demod_set]` — albedo stays matched to the radiance's frame.
    net_aov: Vec<wgpu::Buffer>,
    /// Surface-res present accum the blit resolves to screen (linear rgb, w=1).
    present_accum: wgpu::Buffer,
    present_blit_bg: wgpu::BindGroup,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    n: usize,
    /// S12.5: the buffer set the last frame's net wrote (for the belief eye).
    last_set: usize,
    // Rolling per-stage budget samples for the N0.d table.
    s_trace: Vec<f64>,
    s_gather: Vec<f64>,
    s_net: Vec<f64>,
    /// S3 instrument: the net forward's GPU-only ms (MTLCommandBuffer GPU
    /// timestamps), split from `s_net` (the wall around the blocking call).
    s_net_gpu: Vec<f64>,
    /// N0.i S13 probe: the net stage's wall split — commit CPU ms (critical
    /// path) and downstream wait ms (overlap-hidden). Where the ~8.5 ms hides.
    s_net_commit: Vec<f64>,
    s_net_wait: Vec<f64>,
    s_demod: Vec<f64>,
    s_present: Vec<f64>,
    s_total: Vec<f64>,
    frames: u64,
    /// N0.i S13 THROUGHPUT: wall-clock start of the first recorded frame, for
    /// the frames/second-over-the-whole-run figure (the throughput truth).
    wall_start: Option<Instant>,
}

// SAFETY: `NetPresent` embeds `RdirectLive` (an MPSGraph handle + Metal
// buffers), which is not auto-`Send` because Objective-C objects are
// thread-affine. This rig is built lazily ON the render thread and every
// method that touches it (`net_present_frame` → `resolve_frame`) runs ONLY on
// that thread — `Renderer` is moved once into the render worker and never
// shared or accessed from another thread afterward (the HTTP/screenshot
// threads read the shared `latest` frame, never the `Renderer`). So the
// MPSGraph is never used from two threads; marking the field `Send` (needed
// only so the one-time `Renderer` move compiles) is sound. Scaffold: dies at
// cutover.
#[cfg(target_os = "macos")]
unsafe impl Send for NetPresent {}

#[cfg(target_os = "macos")]
impl NetPresent {
    /// Pool everything once. `low_*` is the trace resolution, `target_*` the
    /// surface (present) resolution the net upscales to.
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        integrator: &Integrator,
        low_w: u32,
        low_h: u32,
        target_w: u32,
        target_h: u32,
    ) -> Result<Self, String> {
        let n = (target_w as usize) * (target_h as usize);
        let weights = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v1.bin"),
        )
        .map_err(|e| format!("read rdirect weights: {e}"))?;
        let live = RdirectLive::from_wgpu_queue(device, queue, &weights, n)?;
        // S9: spin up the encode thread so the ~14 ms MPSGraph per-frame encode
        // rides a background thread while the render thread does GPU work. After
        // this the net is driven via `begin_frame` (pick set) + `commit_net`
        // (commit the pre-encoded buffer + wait) instead of `forward_shared_gpu`.
        live.start_pipeline()?;
        let gather = FeatureGather::new(device);
        let demod = DemodPass::new(device);

        let low_cells = (low_w as u64) * (low_h as u64);
        let net_accum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("net-present low accum"),
            size: low_cells.max(1) * 16, // ACCUM_CELL = vec4<f32>
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // N0.i S13: one AOV per buffer set (frame-overlap demod matches albedo
        // to the radiance's frame). Same count as the net's double-buffer sets.
        let net_aov: Vec<wgpu::Buffer> = (0..live.set_count())
            .map(|_| integrator.make_aov_buffer(device, target_w, target_h))
            .collect();
        let present_accum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("net-present surface accum"),
            size: (n as u64).max(1) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let present_blit_bg = integrator.blit_bind_group(device, &present_accum);

        Ok(Self {
            live,
            gather,
            demod,
            net_accum,
            net_aov,
            present_accum,
            present_blit_bg,
            low_w,
            low_h,
            target_w,
            target_h,
            n,
            last_set: 0,
            s_trace: Vec::new(),
            s_gather: Vec::new(),
            s_net: Vec::new(),
            s_net_gpu: Vec::new(),
            s_net_commit: Vec::new(),
            s_net_wait: Vec::new(),
            s_demod: Vec::new(),
            s_present: Vec::new(),
            s_total: Vec::new(),
            frames: 0,
            wall_start: None,
        })
    }

    /// Trace → gather → forward → undo-log-demod, leaving `present_accum`
    /// filled and `integrator.uniform_buf` set for a 1:1 nearest present blit.
    /// Returns (trace, gather, net, resolve) ms; the final blit+present is timed
    /// by the caller and folded into the `present` stage.
    #[allow(clippy::too_many_arguments)]
    fn resolve_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        integrator: &Integrator,
        uni_low: &IntegratorUniform,
        uni_target: &IntegratorUniform,
        blit_uniform: &IntegratorUniform,
    ) -> (f64, f64, f64, f64) {
        // S9: claim this frame's pre-encoded net command buffer. Returns the
        // buffer SET the gather must fill; the net (committed in the net stage
        // below) reads THIS set, so evidence stays this frame's own (0 latency).
        // Near-instant in steady state (the pipeline stays primed).
        let set = self.live.begin_frame();

        // Bind groups over the integrator's node/tri buffers (reallocated each
        // dynamic tick) — rebuilt per frame, handle-weight.
        let accum_bg = integrator.compute_bind_group(device, &self.net_accum);
        // N0.i S13: trace + gather touch THIS frame's set's AOV.
        let aov_bg = integrator.aov_bind_group(device, &self.net_aov[set]);

        // —— STAGE: trace (low radiance + native AOV) ——
        let t0 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("net trace: clear+accum"),
        });
        enc.clear_buffer(&self.net_accum, 0, None);
        integrator.dispatch(queue, &mut enc, uni_low, &accum_bg, self.low_w, self.low_h);
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("net trace: aov"),
        });
        integrator.dispatch_aov(
            queue,
            &mut enc,
            uni_target,
            &accum_bg,
            &aov_bg,
            self.target_w,
            self.target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // —— STAGE: gather (GPU feature build → pooled shared MTLBuffer) ——
        let feats = self
            .live
            .feature_buffer_set(set)
            .expect("net-present pooled feature buffer");
        let t1 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("net gather"),
        });
        self.gather.encode(
            device,
            queue,
            &mut enc,
            &self.net_accum,
            &self.net_aov[set],
            feats,
            self.low_w,
            self.low_h,
            self.target_w,
            self.target_h,
        );
        queue.submit(Some(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let gather_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // S12: release the gather→net fence on the render queue now the gather
        // is done, so the net command buffer waiting on it (on the dedicated
        // net queue) can run. This is the ONE cross-queue hazard the queue
        // split introduces (net→demod stays a CPU fence via commit_net).
        self.live.signal_gather_ready();

        // —— STAGE: net (N0.i S13 FRAME OVERLAP) ———————————————————————————
        // `commit_net` commits THIS frame's pre-encoded buffer (for `set`, which
        // the gather just filled) WITHOUT blocking — its GPU forward now overlaps
        // the NEXT frame's trace+gather — and WAITS the PREVIOUS frame's buffer,
        // whose net ran during THIS frame's trace+gather and is (near-)complete.
        // So `net_ms` (wall) now measures ≈ commit(cur) + wait(prev, overlapped).
        // GPU-only ms is the completed PREVIOUS buffer's timestamps. The demod
        // consumes that finished buffer's set — one frame of DISPLAY latency, and
        // the presented image is always the COMPLETE image of its own frame's
        // evidence (output-or-nothing). `None` only on the first frame.
        let t2 = Instant::now();
        let demod_set = self.live.commit_net().expect("net commit_net");
        let net_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let net_gpu_ms = self.live.last_gpu_ms();

        // —— STAGE: resolve (CUT 2 GPU demod — no AOV readback, no CPU loop) ——
        // One compute dispatch: reads the FINISHED net output MTLBuffer (the
        // previous frame's `demod_set`, zero-copy wgpu view) + its AOV albedo,
        // undoes the log-demod, writes present_accum. Skipped on the first frame
        // (demod_set None) — present_accum stays as-is (black on boot).
        let t3 = Instant::now();
        if let Some(dset) = demod_set {
            let net_out = self
                .live
                .output_buffer_set(dset)
                .expect("net-present pooled output buffer");
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("net demod"),
            });
            self.demod.encode(
                device,
                queue,
                &mut enc,
                net_out,
                &self.net_aov[dset],
                &self.present_accum,
                self.n as u32,
                false, // presented (undo albedo demod); belief eye is /scry?eye=belief
            );
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            // S12.5: remember which set the presented frame came from, so a
            // /scry?eye=belief capture re-demods THAT net output in belief mode.
            self.last_set = dset;
        }
        // The present blit resolves present_accum (w=1) 1:1 nearest to screen.
        queue.write_buffer(&integrator.uniform_buf, 0, bytemuck::bytes_of(blit_uniform));
        let resolve_ms = t3.elapsed().as_secs_f64() * 1000.0;
        self.s_net_gpu.push(net_gpu_ms);
        // N0.i S13 probe: the net wall = commit(cur, critical) + wait(prev,
        // overlap-hidden). Split them so the doc can name where the gap lives.
        self.s_net_commit.push(self.live.last_commit_ms());
        self.s_net_wait.push(self.live.last_wait_ms());

        (trace_ms, gather_ms, net_ms, resolve_ms)
    }

    /// Record one frame's stage budget and print a rolling median/p95 summary
    /// every 60 frames (the N0.d budget table's live source).
    fn record(&mut self, t: NetTimings) {
        self.s_trace.push(t.trace);
        self.s_gather.push(t.gather);
        self.s_net.push(t.net);
        // s_net_gpu is pushed inside resolve_frame (GPU timestamps live there).
        self.s_demod.push(t.demod);
        self.s_present.push(t.present);
        self.s_total.push(t.total);
        if self.wall_start.is_none() {
            self.wall_start = Some(Instant::now());
        }
        self.frames += 1;
        if self.frames % 60 == 0 {
            // N0.i S13 throughput truth: frames / wall seconds over the run.
            let wall_fps = self
                .wall_start
                .map(|s| self.frames as f64 / s.elapsed().as_secs_f64().max(1e-9))
                .unwrap_or(0.0);
            eprintln!(
                "[n0i] frames={} {}x{}→{}x{} (ms median/p95 vs 16.67): \
                 trace {:.2}/{:.2} gather {:.2}/{:.2} \
                 net[wall {:.2}/{:.2} gpu {:.2}/{:.2} commit {:.2}/{:.2} wait {:.2}/{:.2}] \
                 demod {:.2}/{:.2} present {:.2}/{:.2} TOTAL {:.2}/{:.2} | WALL-FPS {:.1}",
                self.frames,
                self.low_w,
                self.low_h,
                self.target_w,
                self.target_h,
                pct(&self.s_trace, 0.5),
                pct(&self.s_trace, 0.95),
                pct(&self.s_gather, 0.5),
                pct(&self.s_gather, 0.95),
                pct(&self.s_net, 0.5),
                pct(&self.s_net, 0.95),
                pct(&self.s_net_gpu, 0.5),
                pct(&self.s_net_gpu, 0.95),
                pct(&self.s_net_commit, 0.5),
                pct(&self.s_net_commit, 0.95),
                pct(&self.s_net_wait, 0.5),
                pct(&self.s_net_wait, 0.95),
                pct(&self.s_demod, 0.5),
                pct(&self.s_demod, 0.95),
                pct(&self.s_present, 0.5),
                pct(&self.s_present, 0.95),
                pct(&self.s_total, 0.5),
                pct(&self.s_total, 0.95),
                wall_fps,
            );
        }
    }

    /// S12.5 AI DEBUG DOOR — `/budget` JSON: the latest rolling per-stage
    /// median/p95 (ms) plus the frame count and the 16.67 ms wall.
    fn budget_json(&self) -> String {
        let path = if self.live.use_mpsgraph_now() { "mpsgraph" } else { "chain" };
        let wall_fps = self
            .wall_start
            .map(|s| self.frames as f64 / s.elapsed().as_secs_f64().max(1e-9))
            .unwrap_or(0.0);
        format!(
            "{{\"frames\":{},\"wall_ms\":16.67,\"wall_fps\":{:.2},\"path\":\"{path}\",\
             \"canvas\":[{},{}],\"stages\":{{\
             \"trace\":[{:.3},{:.3}],\"gather\":[{:.3},{:.3}],\
             \"net_wall\":[{:.3},{:.3}],\"net_gpu\":[{:.3},{:.3}],\
             \"net_commit\":[{:.3},{:.3}],\"net_wait\":[{:.3},{:.3}],\
             \"demod\":[{:.3},{:.3}],\"present\":[{:.3},{:.3}],\
             \"total\":[{:.3},{:.3}]}}}}",
            self.frames,
            wall_fps,
            self.target_w,
            self.target_h,
            pct(&self.s_trace, 0.5), pct(&self.s_trace, 0.95),
            pct(&self.s_gather, 0.5), pct(&self.s_gather, 0.95),
            pct(&self.s_net, 0.5), pct(&self.s_net, 0.95),
            pct(&self.s_net_gpu, 0.5), pct(&self.s_net_gpu, 0.95),
            pct(&self.s_net_commit, 0.5), pct(&self.s_net_commit, 0.95),
            pct(&self.s_net_wait, 0.5), pct(&self.s_net_wait, 0.95),
            pct(&self.s_demod, 0.5), pct(&self.s_demod, 0.95),
            pct(&self.s_present, 0.5), pct(&self.s_present, 0.95),
            pct(&self.s_total, 0.5), pct(&self.s_total, 0.95),
        )
    }

    /// S12.5 `/state` JSON: forward path, canvas res, frame count, weights id.
    fn state_json(&self) -> String {
        let path = if self.live.use_mpsgraph_now() { "mpsgraph" } else { "chain" };
        format!(
            "{{\"path\":\"{path}\",\"canvas\":[{},{}],\"pixels\":{},\
             \"frames\":{},\"weights\":\"rdirect-weights-v1\",\
             \"in_features\":{},\"out_channels\":{},\"max_pixels\":{}}}",
            self.target_w, self.target_h, self.n, self.frames,
            self.live.in_features(), self.live.out_channels(), self.live.max_pixels(),
        )
    }
}

/// undo the net's log-demod residual by the native albedo (bit-identical to
/// `examples/rdirect_live_frame.rs` / VIII-1's `undo_log_demod`). CPU parity
/// reference; the live present path does this on the GPU (`rdirect_demod.wgsl`,
/// same math), so this stays only as the correctness anchor.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn undo_log_demod_px(dl: Vec3, albedo: Vec3) -> Vec3 {
    let divisor = if albedo.length_squared() > 1e-8 {
        albedo + Vec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        Vec3::ONE
    };
    let e = Vec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
    Vec3::new(e.x.max(0.0), e.y.max(0.0), e.z.max(0.0)) * divisor
}

/// The `q` quantile (0..=1) of `samples` by the nearest-rank method (budget
/// table helper). Returns 0.0 for an empty slice.
fn pct(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = ((q * (s.len() as f64 - 1.0)).round() as usize).min(s.len() - 1);
    s[rank]
}

struct Renderer {
    // Safety: created from the native Tauri Window's raw handles; the app owns that Window
    // until shutdown, and the render worker stops before process exit.
    // `None` in GAIA_NATIVE_OFFSCREEN mode — no NSWindow, no surface; the render
    // loop draws only to `offscreen` and `/scry` serves that.
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// The ONE traced integrator (Rite IV) — replaces the deleted raster path.
    integrator: Integrator,
    /// The realm's render scene, kept so the render loop can tick the world clock
    /// and re-splice the living layer each frame (Rite IV dynamics).
    scene: RenderScene,
    /// The STATIC BVH — built once over the non-behavior leaf triangles and
    /// cached; only the dynamic partition changes per tick, spliced onto this.
    static_bvh: Bvh,
    /// The dynamic-partition SAH params (`.dynamic()` derives the splice's own
    /// `BvhParams`) — kept so `capture_pose` can build a one-off foreign-eye
    /// splice (OWN-EYE CULL) without disturbing the persistent `splice`/`static_bvh`.
    bvh_params: BvhParams,
    /// Refit tuning for the same one-off foreign-eye splice.
    refit_params: RefitParams,
    /// OWN-EYE CULL override (`GAIA_NATIVE_DRAW_OWN_BODY`, default off) — see
    /// `ScryingGlassConfig::draw_own_body`.
    draw_own_body: bool,
    /// The persistent two-level splice (LEVER 1): refits the dynamic partition
    /// per tick when the set is unchanged, rebuilds only on set change / bound
    /// degradation. Its `merged` tree is what gets uploaded.
    splice: DynamicSplice,
    /// `/retina` CPU trees + source IDs; cache invalidates with scene geometry.
    retina_cache: RetinaGeometryCache,
    retina_epoch: u64,
    /// The dynamic model transforms uploaded last frame; when they change the
    /// BVH is re-spliced and accumulation resets (the honest 2spp-live tradeoff).
    last_models: Vec<[f32; 16]>,
    /// Current window camera (the moving eye follows the embodied body).
    camera: Camera,
    sun: SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    int_params: IntegratorParams,
    /// Accumulation frames a /scry moving-eye capture integrates.
    capture_frames: u32,
    /// God's fixed render canvas. Trace, accumulation, temporal buffers, and
    /// offscreen present remain this size across every window resize.
    canvas_width: u32,
    canvas_height: u32,
    /// Persistent accumulation at God's fixed canvas resolution.
    surface_accum: wgpu::Buffer,
    surface_compute_bg: wgpu::BindGroup,
    surface_blit_bg: wgpu::BindGroup,
    samples_before: u32,
    /// LIGHT-NOT-DOTS live temporal accumulation state (see `render`).
    temporal_enabled: bool,
    temporal_params: TemporalParams,
    /// Ping-pong PACKED frame buffers (radiance + primary gbuffer, 2 cells/px)
    /// and history (rgb + accumulated frame count). Parity flips each frame.
    /// Owned here to keep the GPU buffers alive for the lifetime of `t_bind`.
    #[allow(dead_code)]
    t_packed: [wgpu::Buffer; 2],
    #[allow(dead_code)]
    t_hist: [wgpu::Buffer; 2],
    t_bind: [wgpu::BindGroup; 2],
    /// Frame parity for the ping-pong and the previous-frame camera uniform
    /// (None until the first temporal frame has run / after an invalidation).
    t_parity: usize,
    t_prev: Option<IntegratorUniform>,
    /// Eye pose the current fixed-canvas accumulation belongs to.
    last_view: Option<([f32; 3], f32, f32)>,
    offscreen: OffscreenTarget,
    pixel_order: PixelOrder,
    capture_sender: mpsc::Sender<CaptureReady>,
    /// N0.j S13: does the live present path do the offscreen readback EVERY
    /// frame (the old measurement tax, kept as an A/B via
    /// `GAIA_NATIVE_PERFRAME_READBACK=1`) or ON-DEMAND when `/scry` actually
    /// asks (the S13 default)? On-demand serves the current offscreen texture
    /// via `capture_presented`, so the render loop never pays the copy.
    perframe_readback: bool,
    /// N0.j S13: the last frame's per-frame-readback ms (0 in on-demand mode),
    /// set by `net_present_frame`, read by the loop's `outside` accounting.
    last_readback_ms: f64,
    /// N0.j S13 THE OUTSIDE-9ms HUNT: the non-net frame-loop budget.
    outside: OutsideBudget,
    /// S14: the last frame's `advance_world` sub-stage breakdown (skin/gather/
    /// splice/upload), filled by `advance_world`, drained by the loop into
    /// `outside.record_world`.
    last_world_stages: WorldStages,
    /// DAS BLUTBÄNDIGEN — B0 data door state. `None` when the master switch is
    /// off; `Some` carries the world/scene params + last-good snapshot the live
    /// scene/shader bends re-materialize and journal against.
    bloodbend: Option<Bloodbend>,
    /// NEURAL-LIVE N0.c: master switch for the net-present scaffold (config).
    net_present_enabled: bool,
    /// NEURAL-LIVE N0.c: the pooled net-present rig, built lazily on the first
    /// frame once the boot surface size is known (`None` until then / when the
    /// flag is off). macOS-only.
    #[cfg(target_os = "macos")]
    net_present: Option<NetPresent>,
}

impl Renderer {
    /// `window` is `Some` for the normal on-screen surface path and `None` in
    /// GAIA_NATIVE_OFFSCREEN mode (no NSWindow, no wgpu surface). `fallback_dims`
    /// sizes the offscreen present/capture surface when `window` is `None`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        window: Option<&tauri::Window>,
        fallback_dims: (u32, u32),
        capture_sender: mpsc::Sender<CaptureReady>,
        scene: RenderScene,
        int_params: IntegratorParams,
        bvh_params: &BvhParams,
        refit_params: RefitParams,
        capture_frames: u32,
        draw_own_body: bool,
        temporal_enabled: bool,
        temporal_params: TemporalParams,
        canvas_width: u32,
        canvas_height: u32,
        net_present_enabled: bool,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // Build the surface (windowed) or none (offscreen); pick the adapter
        // compatible with whichever we have.
        let surface = match window {
            Some(window) => {
                let target = unsafe {
                    wgpu::SurfaceTargetUnsafe::from_display_and_window(window, window)
                        .map_err(|error| format!("raw-window-handle target: {error}"))?
                };
                let surface = unsafe {
                    instance
                        .create_surface_unsafe(target)
                        .map_err(|error| format!("wgpu surface: {error}"))?
                };
                Some(surface)
            }
            None => None,
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface.as_ref(),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|error| format!("wgpu adapter: {error}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|error| format!("wgpu device: {error}"))?;
        // Size + format: from the surface when windowed; from `fallback_dims`
        // with a fixed BGRA8-sRGB (M1's native surface format) when offscreen,
        // so the captured PNGs match the on-screen path byte-for-byte.
        let (size, format, pixel_order, alpha_mode) = match (&surface, window) {
            (Some(surface), Some(window)) => {
                let size = window.inner_size().map_err(|error| error.to_string())?;
                let capabilities = surface.get_capabilities(&adapter);
                let format = [
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    wgpu::TextureFormat::Bgra8Unorm,
                    wgpu::TextureFormat::Rgba8Unorm,
                ]
                .into_iter()
                .find(|candidate| capabilities.formats.contains(candidate))
                .ok_or_else(|| {
                    "surface has no 8-bit RGBA/BGRA format for framebuffer capture".to_string()
                })?;
                let pixel_order = match format {
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                        PixelOrder::Bgra
                    }
                    _ => PixelOrder::Rgba,
                };
                (size, format, pixel_order, capabilities.alpha_modes[0])
            }
            _ => {
                let (w, h) = fallback_dims;
                (
                    PhysicalSize { width: w.max(1), height: h.max(1) },
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                    PixelOrder::Bgra,
                    wgpu::CompositeAlphaMode::Auto,
                )
            }
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
            desired_maximum_frame_latency: 2,
        };
        if let Some(surface) = &surface {
            surface.configure(&device, &config);
        }

        // The acceleration: a STATIC BVH over the Great Chain's EXACT non-behavior
        // leaf triangles (built once, cached), with the living layer's dynamic
        // partition spliced on top. Load budget printed, never gated (RENDER:
        // cost ∝ pixels, not FLOPs).
        let build_start = Instant::now();
        let static_bvh = Bvh::build(&scene.leaf_triangles(), bvh_params);
        // OWN-EYE CULL doesn't apply yet at construction — no walker pose has
        // fed `command_bodies_walked` (`scene.last_walker_eye` is `None`), so
        // this is identical to `dynamic_leaf_triangles_for_eye` here; the first
        // frame loop's `advance_world` re-splices with the real walker eye.
        let dynamic_tris = scene.dynamic_leaf_triangles();
        let splice = DynamicSplice::build(
            &static_bvh,
            &dynamic_tris,
            &bvh_params.dynamic(),
            refit_params,
        );
        let build_millis = build_start.elapsed().as_secs_f64() * 1e3;
        let integrator = Integrator::new(&device, format, &splice.merged, None);
        let last_models = scene.dynamics.model_matrices();
        eprintln!(
            "[pleroma] BVH nodes={} triangles={} (static {} + dynamic {}) build={build_millis:.1}ms; {} dynamic entit(ies) — living layer",
            integrator.node_count,
            integrator.tri_count,
            static_bvh.tris.len(),
            dynamic_tris.len(),
            scene.dynamics.entities().len(),
        );
        eprintln!(
            "[pleroma] traced integrator spp={} bounces={} rr_start={} — first_light is dead",
            int_params.spp, int_params.max_bounces, int_params.rr_start,
        );

        let camera = scene.camera;
        let sun = scene.sun;
        let sky_top = scene.sky_top;
        let sky_horizon = scene.sky_horizon;

        // ★ THE RESOLUTION IS 640×480: all rendering resources live at the
        // fixed IRON canvas. The window surface is display-only.
        let surface_accum = integrator.make_accum(&device, canvas_width, canvas_height);
        let surface_compute_bg = integrator.compute_bind_group(&device, &surface_accum);
        let surface_blit_bg = integrator.blit_bind_group(&device, &surface_accum);
        let t_packed = [
            integrator.make_temporal_packed(&device, canvas_width, canvas_height),
            integrator.make_temporal_packed(&device, canvas_width, canvas_height),
        ];
        let t_hist = [
            integrator.make_temporal_buffer(&device, canvas_width, canvas_height),
            integrator.make_temporal_buffer(&device, canvas_width, canvas_height),
        ];
        let t_bind = [
            integrator.temporal_bind_group(
                &device,
                &t_packed[0],
                &t_packed[1],
                &t_hist[0],
                &t_hist[1],
            ),
            integrator.temporal_bind_group(
                &device,
                &t_packed[1],
                &t_packed[0],
                &t_hist[1],
                &t_hist[0],
            ),
        ];
        let offscreen = OffscreenTarget::new(&device, format, canvas_width, canvas_height);
        eprintln!(
            "[wgpu] traced God's canvas {canvas_width}x{canvas_height}; surface {}x{} = nearest integer display scale ({format:?})",
            config.width, config.height,
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            integrator,
            scene,
            static_bvh,
            bvh_params: *bvh_params,
            refit_params,
            draw_own_body,
            splice,
            retina_cache: RetinaGeometryCache::default(),
            retina_epoch: 0,
            last_models,
            camera,
            sun,
            sky_top,
            sky_horizon,
            int_params,
            capture_frames,
            canvas_width,
            canvas_height,
            surface_accum,
            surface_compute_bg,
            surface_blit_bg,
            samples_before: 0,
            temporal_enabled,
            temporal_params,
            t_packed,
            t_hist,
            t_bind,
            t_parity: 0,
            t_prev: None,
            last_view: None,
            offscreen,
            pixel_order,
            capture_sender,
            perframe_readback: matches!(
                std::env::var("GAIA_NATIVE_PERFRAME_READBACK").as_deref(),
                Ok("1" | "true")
            ),
            last_readback_ms: 0.0,
            outside: OutsideBudget::default(),
            last_world_stages: WorldStages::default(),
            bloodbend: None,
            net_present_enabled,
            #[cfg(target_os = "macos")]
            net_present: None,
        })
    }

    /// DAS BLUTBÄNDIGEN — SCENE BEND. A watched scene JSON file changed: run the
    /// full Zauberpolizei inspection (loader + render-scene materialization into
    /// a THROWAWAY world) BEFORE touching living tissue. On rejection the world
    /// stays byte-identical and a police report is logged. On success the
    /// PREVIOUS good bytes are journaled (Traumdeuter-Vorritt), the entity diff
    /// is reported (law 4 blast radius), and the scene tier rebuilds live —
    /// window/device/surface/pipelines all persist.
    fn bend_scene(&mut self) {
        let Some(bb) = self.bloodbend.as_ref() else {
            return;
        };
        let world_path = bb.world_path.clone();
        let scene_params = bb.scene_params.clone();
        let journal_dir = bb.params.journal_dir.clone();
        let scene_paths = bb.params.scene_paths.clone();
        let previous = bb.last_good.clone();
        let next = bloodbend::read_scene_bytes(&scene_paths);

        // ADVISORY 3 — no-op bend: the watcher can fire on a touch with no
        // entity-level change (save-without-edit, whitespace-only diff, a
        // broken-JSON write that briefly round-trips back to the same text).
        // Skip journal + rebuild + accumulation-reset entirely.
        //
        // RE-PASS ADVISORY (corner a): do NOT advance `last_good` to `next`
        // here. A duplicate-id write can be value-identical at the entity
        // level (diff empty) while its raw bytes are still loader-rejectable
        // (e.g. a repeated key the loader itself would refuse) — those bytes
        // never passed INSPECTION 1+2 below. Leaving `last_good` pointed at
        // the last VALIDATED bytes keeps the invariant "last_good always
        // loads" true across a no-op; the next real (non-empty) diff still
        // computes correctly against those same last-validated bytes.
        let diff = bloodbend::diff_scenes(&previous, &next);
        if diff.is_empty() {
            eprintln!("[bloodbend] no-op bend ignored · scene · entity diff empty");
            return;
        }

        // INSPECTION 1+2 — TOCTOU-SAFE (bloodbend-b0 fix pass, adversary
        // MUST-FIX 2): validate the EXACT bytes just captured in `next` by
        // materializing them into a private validation dir and loading FROM
        // THAT — never re-reading the live world dir a second time. `last_good`
        // is set to this SAME `next` below, so validated bytes == stored
        // last_good bytes BY CONSTRUCTION; no window remains for a concurrent
        // write to slip unvalidated bytes into `last_good` (ordeal f).
        let validate_dir = match bloodbend::write_validation_dir(&journal_dir, &world_path, &next)
        {
            Ok(dir) => dir,
            Err(error) => {
                bloodbend::police_report("scene", &format!("validation snapshot: {error}"));
                return;
            }
        };
        // Parse + deserialize through the sigil structs. NOTE: the loader's
        // `deny_unknown_fields` is loud on a component's OWN fields (e.g.
        // physics::RigidBody), but the data-driven component model tolerates
        // an unrecognized COMPONENT KEY on an entity by design — that is a
        // flagged design question, not a loader bug; do not read this comment
        // as "any unknown key is rejected".
        let mut core = Core::default();
        ScryingGlassPackage.register(&mut core);
        if let Err(error) = load_world_dir(&validate_dir, &mut core.world) {
            let _ = std::fs::remove_dir_all(&validate_dir);
            bloodbend::police_report("scene", &error);
            return;
        }
        // INSPECTION 2 — materialize the render scene (geometry/material laws).
        let new_scene = match RenderScene::from_ecs(core.world, &scene_params) {
            Ok(scene) => scene,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&validate_dir);
                bloodbend::police_report("scene", &format!("materialize: {error}"));
                return;
            }
        };
        let _ = std::fs::remove_dir_all(&validate_dir);

        // TRAUMDEUTER-VORRITT — snapshot the previous good bytes BEFORE apply.
        match bloodbend::journal_previous(&journal_dir, &previous) {
            Ok(dir) => eprintln!("[bloodbend] 📜 journaled previous scene → {}", dir.display()),
            Err(error) => {
                bloodbend::police_report(
                    "scene",
                    &format!("journal failed, refusing to apply (undo would be lost): {error}"),
                );
                return;
            }
        }

        self.rebuild_scene(new_scene);
        if let Some(bb) = self.bloodbend.as_mut() {
            bb.last_good = next;
        }
        bloodbend::bend_applied("scene", &diff.summary());
    }

    /// Re-project authoritative crystal state through the one render scene path.
    fn rebuild_world_core(
        &mut self,
        world_core: &WorldCore,
        scene_params: &SceneParameters,
    ) -> Result<(), String> {
        let mut core = Core::default();
        ScryingGlassPackage.register(&mut core);
        world_core.materialize_into(&mut core.world)?;
        let scene = RenderScene::from_ecs(core.world, scene_params)
            .map_err(|error| format!("materialize authority state: {error}"))?;
        self.rebuild_scene(scene);
        Ok(())
    }

    /// The scene tier of the blast-radius ladder (law 4): swap the render scene,
    /// rebuild the static BVH + dynamic splice over the new leaf triangles, re-
    /// upload the acceleration structure, refresh sun/sky, and reset the window
    /// accumulation. The device, surface, integrator pipelines and uniform
    /// buffer all persist (law 1 — stable substrate outlives the swapped unit).
    fn rebuild_scene(&mut self, new_scene: RenderScene) {
        self.scene = new_scene;
        let tris = self.scene.leaf_triangles();
        self.static_bvh = Bvh::build(&tris, &self.bvh_params);
        let dynamic_tris = self.scene.dynamic_leaf_triangles();
        self.splice = DynamicSplice::build(
            &self.static_bvh,
            &dynamic_tris,
            &self.bvh_params.dynamic(),
            self.refit_params,
        );
        self.integrator.update_bvh(&self.device, &self.splice.merged);
        self.retina_epoch = self.retina_epoch.wrapping_add(1);
        self.retina_cache.clear();
        self.last_models = self.scene.dynamics.model_matrices();
        self.sun = self.scene.sun;
        self.sky_top = self.scene.sky_top;
        self.sky_horizon = self.scene.sky_horizon;
        self.camera.fov_y_radians = self.scene.camera.fov_y_radians;
        self.reset_surface_accum();
    }

    /// DAS BLUTBÄNDIGEN — SHADER BEND. The watched WGSL source changed: read it
    /// and hand it to `Integrator::reload_shader`, which recompiles + rebuilds
    /// the pipelines under a wgpu Validation error scope. A bad shader keeps the
    /// OLD pipeline rendering and yields a police report; a clean one swaps the
    /// pipelines and resets accumulation. Buffers/layouts/bind groups persist.
    fn bend_shader(&mut self) {
        let Some(bb) = self.bloodbend.as_ref() else {
            return;
        };
        let path = bb.params.shader_path.clone();
        let journal_dir = bb.params.journal_dir.clone();
        let previous_shader = bb.last_good_shader.clone();
        let source = match std::fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) => {
                bloodbend::police_report("shader", &format!("read {}: {error}", path.display()));
                return;
            }
        };

        // ADVISORY 3 — no-op bend: identical source (a touch with no edit).
        if source == previous_shader {
            eprintln!(
                "[bloodbend] no-op bend ignored · shader · {} unchanged",
                path.display()
            );
            return;
        }

        // TRAUMDEUTER-VORRITT — SHADER JOURNAL (bloodbend-b0 fix pass,
        // adversary MUST-FIX 1): snapshot the previous good WGSL source BEFORE
        // the swap is attempted, mirroring the scene tier. A journal-write
        // failure REFUSES the bend — undo lost = no bend.
        match bloodbend::journal_previous_shader(&journal_dir, &previous_shader) {
            Ok(dir) => eprintln!("[bloodbend] 📜 journaled previous shader → {}", dir.display()),
            Err(error) => {
                bloodbend::police_report(
                    "shader",
                    &format!("journal failed, refusing to apply (undo would be lost): {error}"),
                );
                return;
            }
        }

        let format = self.config.format;
        match self.integrator.reload_shader(&self.device, &source, format) {
            Ok(()) => {
                self.reset_surface_accum();
                if let Some(bb) = self.bloodbend.as_mut() {
                    bb.last_good_shader = source;
                }
                bloodbend::bend_applied("shader", &format!("{} recompiled", path.display()));
            }
            Err(error) => bloodbend::police_report("shader", &error),
        }
    }

    /// Rebuild the fixed-canvas accumulation buffer and drop its samples.
    fn reset_surface_accum(&mut self) {
        let accum = self
            .integrator
            .make_accum(&self.device, self.canvas_width, self.canvas_height);
        self.surface_compute_bg = self.integrator.compute_bind_group(&self.device, &accum);
        self.surface_blit_bg = self.integrator.blit_bind_group(&self.device, &accum);
        self.surface_accum = accum;
        self.samples_before = 0;
    }

    /// MEASURE: honest per-frame GPU cost at God's fixed render canvas.
    /// Dispatches `frames` accumulation passes into a throwaway
    /// accum from the live spawn camera, force-flushing the GPU
    /// (`poll(wait)`) after each so the timing is real GPU work, not an
    /// async submit. Returns (median_ms, mean_ms). Runs once at startup off
    /// the frame loop, so it never perturbs live frames.
    fn measure_trace_ms(&mut self, frames: u32) -> (f64, f64) {
        let (width, height) = (self.canvas_width, self.canvas_height);
        let accum = self.integrator.make_accum(&self.device, width, height);
        let compute_bg = self.integrator.compute_bind_group(&self.device, &accum);
        let mut samples_before = 0u32;
        let mut times = Vec::with_capacity(frames as usize);
        for _ in 0..frames.max(1) {
            let uniform = IntegratorUniform::build(
                &self.camera,
                &self.sun,
                self.sky_top,
                self.sky_horizon,
                width,
                height,
                self.integrator.node_count,
                self.integrator.tri_count,
                samples_before,
                &self.int_params,
                None,
            );
            let start = Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("trace timing"),
                });
            self.integrator.dispatch(
                &self.queue,
                &mut encoder,
                &uniform,
                &compute_bg,
                width,
                height,
            );
            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            times.push(start.elapsed().as_secs_f64() * 1e3);
            samples_before += self.int_params.spp;
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        (median, mean)
    }

    /// Advance the world clock one tick and, when the living layer actually
    /// moved, re-splice the dynamic partition onto the cached static BVH, re-
    /// upload it, and reset accumulation. STRATEGY (DYNAMICS): two-level splice
    /// — the static hierarchy never re-sorts; only the tiny dynamic partition
    /// rebuilds, fused under a new root by `Bvh::merge` (O(Sn+Dn) linear).
    /// ACCUMULATION: continues progressively while every dynamic transform is
    /// unchanged; the instant one changes (a bobbing lantern, every tick) the
    /// BVH re-splices and accumulation resets — so a continuously moving world
    /// renders live at `spp` samples/frame (no ghosting), and pauses converge.
    /// PLAYGROUND — push reach (m), speed (m/s, the Op::Impulse velocity
    /// delta) and aim radius (m, perpendicular tolerance off the view ray)
    /// read from the environment, never hardcoded: `GAIA_PUSH_REACH`
    /// (default 4 m — arm's length plus a step), `GAIA_PUSH_SPEED`
    /// (default 5 m/s — a few m/s, enough to topple a rigid stack and, on a
    /// weakly-bonded crate, tear it apart) and `GAIA_PUSH_AIM_RADIUS`
    /// (default 0.9 m — a crate's own reach radius, so the crosshair need
    /// not be pixel-perfect on a 0.8 m box).
    fn push_params() -> (f32, f32, f32) {
        let num = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(default)
        };
        (
            num("GAIA_PUSH_REACH", 4.0),
            num("GAIA_PUSH_SPEED", 5.0),
            num("GAIA_PUSH_AIM_RADIUS", 0.9),
        )
    }

    /// PLAYGROUND — build the push op for a view ray: pick the nearest
    /// pushable body the ray is aimed at within reach and name it in an
    /// `Op::Impulse` carrying a `speed` m/s velocity delta along the ray.
    /// Empty when nothing physical is under the crosshair (a silent miss) or
    /// the realm has no physics. This is the WHOLE push door: the F key / a
    /// locked click / the `/push` organ all funnel through the identical
    /// Op::Impulse an agent would send.
    ///
    /// ADVISORY (merge-conductor #12): `examples/playground_push.rs` carries
    /// its own `pick()` — a byte-for-byte copy of this ray/AIM_RADIUS logic,
    /// kept verbatim on purpose so the example proves the window's actual
    /// door rather than a stub. Extracting a shared fn was considered and
    /// parked: the example lives outside the crate's public surface (no
    /// clean import path today) and a shared helper would need its own
    /// pub(crate) plumbing for one call site. Noted as copy-drift risk on
    /// record — if `build_push_ops` changes, `pick()` must change with it.
    fn build_push_ops(&self, eye: Vec3, yaw: f32, pitch: f32) -> Vec<Op> {
        let Some(physics) = self.scene.physics() else {
            return Vec::new();
        };
        let (reach, speed, aim_radius) = Self::push_params();
        let cos_pitch = pitch.cos();
        let dir = Vec3::new(-yaw.sin() * cos_pitch, pitch.sin(), -yaw.cos() * cos_pitch);
        let mut best: Option<(f32, String)> = None;
        for (gaia_id, centroid) in physics.push_targets() {
            let c = Vec3::new(centroid[0] as f32, centroid[1] as f32, centroid[2] as f32);
            let v = c - eye;
            let t = v.dot(dir);
            if t <= 0.0 || t > reach {
                continue; // behind the eye or past arm's reach
            }
            let perp = (v - dir * t).length();
            if perp > aim_radius {
                continue; // the ray does not pass through this body
            }
            if best.as_ref().is_none_or(|(bt, _)| t < *bt) {
                best = Some((t, gaia_id));
            }
        }
        match best {
            Some((_, id)) => {
                let dv = dir * speed;
                vec![Op::Impulse(ImpulseOp {
                    id,
                    delta_velocity: [dv.x as f64, dv.y as f64, dv.z as f64],
                    extra: Default::default(),
                })]
            }
            None => Vec::new(),
        }
    }

    fn advance_world(&mut self, body_speed: f32, walker: Option<WalkerPose>, push_ops: &[Op]) {
        // S14: time each sub-stage of the ~7 ms advance so the thief is split.
        self.last_world_stages = WorldStages::default();
        let has_bodies = !self.scene.bodies.is_empty();
        if self.scene.dynamics.entities().is_empty() && !has_bodies {
            return; // a still realm never pays the living-layer cost
        }
        let t_skin = Instant::now();
        // RITE V·V1 — drive the embodied bodies from the walker's velocity: the
        // commanded speed feeds each body's SAMA state machine, its pose re-skins
        // the body per tick. A walking body changes the dynamic partition every
        // tick even when the living models are still, so it forces a re-splice.
        // RITE V FINAL WELD — `walker` (its world pose) drives walker-ATTACHED
        // bodies (`follows: "walker"`): they TRACK the walker, gait derived from
        // displacement, instead of gaiting in place off the broadcast.
        let bodies_animating = self.scene.command_bodies_walked(body_speed, walker);
        self.last_world_stages.command = t_skin.elapsed().as_secs_f64() * 1000.0;
        let t_tick = Instant::now();
        self.scene.tick_with_ops(push_ops);
        self.last_world_stages.tick = t_tick.elapsed().as_secs_f64() * 1000.0;
        let [kami, apply, physics, rederive, solver_step, poll] =
            self.scene.last_tick_breakdown();
        self.last_world_stages.kami = kami;
        self.last_world_stages.apply = apply;
        self.last_world_stages.physics = physics;
        self.last_world_stages.rederive = rederive;
        self.last_world_stages.solver_step = solver_step;
        self.last_world_stages.poll = poll;
        self.last_world_stages.skin =
            self.last_world_stages.command + self.last_world_stages.tick;
        let models = self.scene.dynamics.model_matrices();
        if models == self.last_models && !bodies_animating {
            return; // nothing moved — keep accumulating
        }
        // OWN-EYE CULL — the window camera IS the walker's own eye every frame
        // (`set_view_pose` and this tick's `walker` share one pose, wired in the
        // run loop below), so a walker-attached body never renders inside it.
        let t_gather = Instant::now();
        let dynamic_tris = self.scene.dynamic_leaf_triangles_for_eye(
            self.camera.eye,
            scrying_glass::scene::OWN_EYE_EPSILON_M,
            self.draw_own_body,
        );
        self.last_world_stages.gather = t_gather.elapsed().as_secs_f64() * 1000.0;
        let t_splice = Instant::now();
        self.splice.update(&self.static_bvh, &dynamic_tris);
        self.last_world_stages.splice = t_splice.elapsed().as_secs_f64() * 1000.0;
        let t_upload = Instant::now();
        self.integrator
            .update_bvh(&self.device, &self.splice.merged);
        self.retina_epoch = self.retina_epoch.wrapping_add(1);
        self.retina_cache.clear();
        self.last_world_stages.upload = t_upload.elapsed().as_secs_f64() * 1000.0;
        // The node/tri buffers changed — rebuild the bind groups (they bind them)
        // and drop the stale samples (moved geometry invalidates the mean).
        self.reset_surface_accum();
        self.last_models = models;
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0
            && size.height > 0
            && (size.width != self.config.width || size.height != self.config.height)
        {
            self.config.width = size.width;
            self.config.height = size.height;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, &self.config);
            }
            // The surface (window-size) alone changed. God's canvas is fixed
            // resolution (canvas_width/canvas_height); accumulation and
            // offscreen capture ride the canvas, not the window, and remain
            // untouched here.
        }
    }

    /// Point the windowed camera at the embodied player's eye. Movement resets
    /// the accumulation on the next frame (detected by `last_view`).
    fn set_view_pose(&mut self, eye: Vec3, yaw: f32, pitch: f32) {
        self.camera.eye = eye;
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;
    }

    fn view_key(&self) -> ([f32; 3], f32, f32) {
        (self.camera.eye.to_array(), self.camera.yaw, self.camera.pitch)
    }

    /// Submit ONE traced frame (dispatch → offscreen/surface blit → capture
    /// copy → present) WITHOUT waiting on the GPU, returning the submission's
    /// `SubmissionIndex`. The pipelined `run_render_loop` completes the PREVIOUS
    /// frame's submission (explicit `Wait`) only AFTER the NEXT frame's CPU
    /// stages (`advance_world`) have run — so frame N+1's skin/tick/splice/
    /// upload overlap frame N's GPU trace (LEVER 2, the shape `perf_audit`'s
    /// ATOM B / `live_loop_audit` proved bit-identical to serial). Scheduling
    /// only: `update_bvh` allocates FRESH node/tri buffers each frame, so frame
    /// N's in-flight trace keeps reading its own (wgpu tracks GPU-side lifetime
    /// by the in-flight command buffer), and dispatch+blit ride ONE submission
    /// so each frame's blit reads exactly its own trace — content is unchanged.
    fn render(&mut self, size: PhysicalSize<u32>) -> Option<wgpu::SubmissionIndex> {
        self.resize(size);

        // NEURAL-LIVE N0.c SCAFFOLD: present the ONE net's frame instead of the
        // integrator's raw blit. On build failure the flag self-clears and the
        // normal present below runs (never a black window).
        #[cfg(target_os = "macos")]
        if self.net_present_enabled {
            if let Ok(idx) = self.net_present_frame() {
                return idx;
            }
        }

        // LIGHT-NOT-DOTS: with temporal accumulation the eye moving is NOT a
        // reset — the resolve reprojects last frame's light into the new view.
        // Only the legacy escape-hatch path throws the samples away on move.
        let key = self.view_key();
        if !self.temporal_enabled && self.last_view != Some(key) {
            self.reset_surface_accum();
        }
        self.last_view = Some(key);

        // The canvas is God's fixed render resolution; the mutable surface is
        // display-only and receives a nearest integer-scale blit.
        let (width, height) = (self.canvas_width, self.canvas_height);
        let (surface_w, surface_h) = (self.config.width, self.config.height);
        let mut uniform = IntegratorUniform::build(
            &self.camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            width,
            height,
            self.integrator.node_count,
            self.integrator.tri_count,
            self.samples_before,
            &self.int_params,
            None,
        );
        // `IntegratorUniform::build` already derives camera aspect from the
        // fixed canvas. The shader letterboxes/pillarboxes this result using
        // nearest integer display scaling; it never re-renders for the window.
        uniform.surface = [surface_w, surface_h, 1, 0];

        // LIGHT-NOT-DOTS: hand the resolve the previous frame's camera + dials.
        if self.temporal_enabled {
            let t = &self.temporal_params;
            uniform.temporal = [t.alpha_min, t.depth_tol, t.normal_tol, t.clamp_k];
            match self.t_prev {
                Some(prev) => {
                    uniform.prev_eye = prev.eye;
                    uniform.prev_right = prev.right;
                    uniform.prev_up = prev.up;
                    uniform.prev_forward = prev.forward;
                    uniform.temporal_flags = [1, t.max_history, t.still_px.to_bits(), 0];
                }
                None => {
                    uniform.temporal_flags = [0, t.max_history, t.still_px.to_bits(), 0];
                }
            }
        }

        let surface_frame = match self.surface.as_ref().map(|s| s.get_current_texture()) {
            Some(
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame),
            ) => Some(frame),
            Some(wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost) => {
                if let Some(surface) = &self.surface {
                    surface.configure(&self.device, &self.config);
                }
                None
            }
            // Offscreen mode (surface None) or any transient state → no surface
            // present; the offscreen present + capture below still runs.
            _ => None,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("traced frame + capture"),
            });
        // LIGHT-NOT-DOTS: temporal path traces THIS frame + reprojects/blends
        // last frame's accumulated light into the accum the blit reads; the
        // legacy path dispatches one accumulation frame in place. Either way the
        // blit below presents `surface_accum` unchanged.
        if self.temporal_enabled {
            self.integrator.dispatch_temporal(
                &self.queue,
                &mut encoder,
                &uniform,
                &self.surface_compute_bg,
                &self.t_bind[self.t_parity],
                width,
                height,
            );
        } else {
            self.integrator.dispatch(
                &self.queue,
                &mut encoder,
                &uniform,
                &self.surface_compute_bg,
                width,
                height,
            );
        }
        self.integrator.blit(
            &mut encoder,
            &self.offscreen.view,
            &self.surface_blit_bg,
            "offscreen present",
        );
        if let Some(frame) = &surface_frame {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.integrator.blit(
                &mut encoder,
                &view,
                &self.surface_blit_bg,
                "surface present",
            );
        }

        if let Some(index) = self.offscreen.claim_slot() {
            let slot = &self.offscreen.slots[index];
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.offscreen.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &slot.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.offscreen.padded_bytes_per_row),
                        rows_per_image: Some(self.offscreen.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.offscreen.width,
                    height: self.offscreen.height,
                    depth_or_array_layers: 1,
                },
            );
            let sender = self.capture_sender.clone();
            let buffer = slot.buffer.clone();
            let callback_buffer = buffer.clone();
            let busy = slot.busy.clone();
            let callback_busy = busy.clone();
            let width = self.offscreen.width;
            let height = self.offscreen.height;
            let padded_bytes_per_row = self.offscreen.padded_bytes_per_row;
            let pixel_order = self.pixel_order;
            encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
                let capture = CaptureReady {
                    result: result.map_err(|error| error.to_string()),
                    buffer: callback_buffer,
                    width,
                    height,
                    padded_bytes_per_row,
                    pixel_order,
                    busy: callback_busy,
                };
                if let Err(error) = sender.send(capture) {
                    let capture = error.0;
                    if capture.result.is_ok() {
                        capture.buffer.unmap();
                    }
                    capture.busy.store(false, Ordering::Release);
                }
            });
        }
        let submission = self.queue.submit(Some(encoder.finish()));
        self.samples_before += self.int_params.spp;
        // LIGHT-NOT-DOTS: this frame's camera becomes next frame's reprojection
        // source, and the ping-pong parity flips (hist_out→hist_prev, curr_gbuf
        // →gbuf_prev). Only after temporal actually ran.
        if self.temporal_enabled {
            self.t_prev = Some(uniform);
            self.t_parity ^= 1;
        }
        if let Some(frame) = surface_frame {
            self.queue.present(frame);
        }
        Some(submission)
    }

    /// NEURAL-LIVE N0.c: one live frame through the ONE net. Builds/rebuilds the
    /// pooled rig on demand, traces low radiance + native AOV, gathers features
    /// on the GPU, runs the MPSGraph forward, undoes the log-demod, and presents
    /// the result 1:1 to both the surface and the offscreen capture target (so
    /// `/screenshot` reads the net's frame). Records the per-stage budget.
    /// `Err(())` means the rig could not be built (flag self-cleared) — the
    /// caller falls through to the normal present.
    #[cfg(target_os = "macos")]
    fn net_present_frame(&mut self) -> Result<Option<wgpu::SubmissionIndex>, ()> {
        let (surface_w, surface_h) = (self.config.width, self.config.height);
        // RESOLUTION OF GOD (law 0a25530): trace, net AND present all run at the
        // 640×480 canvas — the net NEVER enlarges a small trace to the window.
        // low == target == render res; the window gets it by a nearest/integer
        // display blit only. Anamorphic camera framing (surface aspect below)
        // maps the canvas onto the window without geometric distortion, exactly
        // as the normal `render` path does.
        let (low_w, low_h) = (self.canvas_width, self.canvas_height);
        let (target_w, target_h) = (self.canvas_width, self.canvas_height);

        let rebuild = match &self.net_present {
            Some(np) => np.target_w != target_w || np.target_h != target_h,
            None => true,
        };
        if rebuild {
            match NetPresent::new(
                &self.device,
                &self.queue,
                &self.integrator,
                low_w,
                low_h,
                target_w,
                target_h,
            ) {
                Ok(np) => {
                    eprintln!(
                        "[n0c] net-present rig pooled (God's res): trace {low_w}x{low_h} → net {target_w}x{target_h} → nearest blit → surface {surface_w}x{surface_h} ({} px)",
                        (target_w as usize) * (target_h as usize)
                    );
                    self.net_present = Some(np);
                }
                Err(e) => {
                    eprintln!("[n0c] net-present disabled (build failed): {e}");
                    self.net_present_enabled = false;
                    return Err(());
                }
            }
        }

        // Uniforms: low radiance trace, native AOV, and the 1:1 nearest present
        // blit — all framed to the SURFACE aspect (mirrors the normal `render`).
        let (right, up, _forward) = self.camera.basis();
        let surface_aspect = surface_w as f32 / surface_h.max(1) as f32;
        let half = (self.camera.fov_y_radians * 0.5).tan();
        let r = right * (half * surface_aspect);
        let u = up * half;
        let apply_aspect = |uni: &mut IntegratorUniform| {
            uni.right = [r.x, r.y, r.z, 0.0];
            uni.up = [u.x, u.y, u.z, 0.0];
        };
        let noisy = IntegratorParams {
            spp: 1,
            ..self.int_params.clone()
        };
        let mut uni_low = IntegratorUniform::build(
            &self.camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            low_w,
            low_h,
            self.integrator.node_count,
            self.integrator.tri_count,
            0,
            &noisy,
            None,
        );
        apply_aspect(&mut uni_low);
        let mut uni_target = IntegratorUniform::build(
            &self.camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            target_w,
            target_h,
            self.integrator.node_count,
            self.integrator.tri_count,
            0,
            &self.int_params,
            None,
        );
        apply_aspect(&mut uni_target);
        // Present: params.xy = the 640×480 canvas (present_accum dims);
        // surface.xy = the window; mode 1 = nearest — the display blit scales
        // God's res onto any surface with no interpolation (integer when the
        // window is a whole multiple, nearest otherwise). No neural enlarge.
        let mut blit_uniform = uni_target;
        blit_uniform.surface = [surface_w, surface_h, 1, 0]; // nearest canvas→window.

        // Take the rig out to avoid borrowing `self` twice; put it back after.
        let mut np = self.net_present.take().expect("net-present rig present");
        let (trace_ms, gather_ms, net_ms, resolve_ms) = np.resolve_frame(
            &self.device,
            &self.queue,
            &self.integrator,
            &uni_low,
            &uni_target,
            &blit_uniform,
        );

        // —— STAGE: present (blit the net frame to surface + offscreen, capture) ——
        let t_present = Instant::now();
        let surface_frame = match self.surface.as_ref().map(|s| s.get_current_texture()) {
            Some(
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame),
            ) => Some(frame),
            Some(wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost) => {
                if let Some(surface) = &self.surface {
                    surface.configure(&self.device, &self.config);
                }
                None
            }
            _ => None,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("net present + capture"),
            });
        self.integrator.blit(
            &mut encoder,
            &self.offscreen.view,
            &np.present_blit_bg,
            "net offscreen present",
        );
        if let Some(frame) = &surface_frame {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.integrator
                .blit(&mut encoder, &view, &np.present_blit_bg, "net surface present");
        }
        // N0.j S13.2 KILL THE MEASUREMENT TAX: the per-frame offscreen readback
        // (copy_texture_to_buffer + map submit) fed `latest` so a bare `/scry`
        // could be served cheaply — but it ran EVERY frame whether anyone looked
        // or not, ~measurement tax on the render thread. It is now ON-DEMAND
        // (`capture_presented` reads the current offscreen texture when `/scry`
        // asks); the per-frame copy runs only under the A/B toggle
        // `GAIA_NATIVE_PERFRAME_READBACK=1`. The offscreen BLIT above still runs
        // every frame, so the texture always holds the latest presented image
        // for the on-demand path to read.
        let t_readback = Instant::now();
        if self.perframe_readback {
            if let Some(index) = self.offscreen.claim_slot() {
                let slot = &self.offscreen.slots[index];
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.offscreen.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &slot.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(self.offscreen.padded_bytes_per_row),
                            rows_per_image: Some(self.offscreen.height),
                        },
                    },
                    wgpu::Extent3d {
                        width: self.offscreen.width,
                        height: self.offscreen.height,
                        depth_or_array_layers: 1,
                    },
                );
                let sender = self.capture_sender.clone();
                let buffer = slot.buffer.clone();
                let callback_buffer = buffer.clone();
                let busy = slot.busy.clone();
                let callback_busy = busy.clone();
                let width = self.offscreen.width;
                let height = self.offscreen.height;
                let padded_bytes_per_row = self.offscreen.padded_bytes_per_row;
                let pixel_order = self.pixel_order;
                encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
                    let capture = CaptureReady {
                        result: result.map_err(|error| error.to_string()),
                        buffer: callback_buffer,
                        width,
                        height,
                        padded_bytes_per_row,
                        pixel_order,
                        busy: callback_busy,
                    };
                    if let Err(error) = sender.send(capture) {
                        let capture = error.0;
                        if capture.result.is_ok() {
                            capture.buffer.unmap();
                        }
                        capture.busy.store(false, Ordering::Release);
                    }
                });
            }
        }
        self.last_readback_ms = t_readback.elapsed().as_secs_f64() * 1000.0;
        let submission = self.queue.submit(Some(encoder.finish()));
        if let Some(frame) = surface_frame {
            self.queue.present(frame);
        }
        let blit_ms = t_present.elapsed().as_secs_f64() * 1000.0;
        // S3: demod (resolve_ms) and the surface blit are now separate columns.
        let total = trace_ms + gather_ms + net_ms + resolve_ms + blit_ms;
        np.record(NetTimings {
            trace: trace_ms,
            gather: gather_ms,
            net: net_ms,
            demod: resolve_ms,
            present: blit_ms,
            total,
        });
        self.net_present = Some(np);
        Ok(Some(submission))
    }

    /// S12.5 AI DEBUG DOOR — the BELIEF eye. Re-demods THIS frame's net output
    /// (the last committed set, still resident in the pooled MTLBuffer) in
    /// belief mode (raw `exp(dl)-1`, NO albedo multiply) into a fresh canvas-res
    /// offscreen and reads it back — the accum-belief PNG owed since n0e. Runs
    /// on the render thread (like `capture_pose`); does not disturb the live
    /// present accum's next frame (it recomputes it). macOS-only / net-present.
    #[cfg(target_os = "macos")]
    fn capture_belief(&mut self) -> Result<CapturedFrame, String> {
        if self.net_present.is_none() {
            return Err(
                "belief eye needs the net-present rig (GAIA_NATIVE_NET_PRESENT=true)".into(),
            );
        }
        let np = self.net_present.take().expect("net-present rig present");
        let out = self.capture_belief_inner(&np);
        self.net_present = Some(np);
        out
    }

    #[cfg(target_os = "macos")]
    fn capture_belief_inner(&mut self, np: &NetPresent) -> Result<CapturedFrame, String> {
        let (w, h) = (np.target_w, np.target_h);
        let net_out = np
            .live
            .output_buffer_set(np.last_set)
            .ok_or_else(|| "belief: no pooled net output buffer".to_string())?;
        // Belief demod: raw net radiance into the present accum (overwritten
        // next live frame). One dispatch, canvas-res.
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("belief demod"),
            });
        np.demod.encode(
            &self.device,
            &self.queue,
            &mut enc,
            net_out,
            &np.net_aov[np.last_set],
            &np.present_accum,
            np.n as u32,
            true, // BELIEF
        );
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // 1:1 nearest blit present_accum → a fresh canvas-res sRGB target.
        let mut blit_uniform = IntegratorUniform::build(
            &self.camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            w,
            h,
            self.integrator.node_count,
            self.integrator.tri_count,
            0,
            &self.int_params,
            None,
        );
        blit_uniform.surface = [w, h, 1, 0]; // nearest 1:1 canvas→target
        self.queue
            .write_buffer(&self.integrator.uniform_buf, 0, bytemuck::bytes_of(&blit_uniform));

        let target = OffscreenTarget::new(&self.device, self.config.format, w, h);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("belief present + capture"),
            });
        self.integrator
            .blit(&mut encoder, &target.view, &np.present_blit_bg, "belief present");
        let slot = &target.slots[0];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let buffer = slot.buffer.clone();
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = done_tx.send(result.map(|_| ()).map_err(|e| e.to_string()));
        });
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        done_rx
            .recv()
            .map_err(|e| format!("belief readback channel closed: {e}"))??;
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|e| format!("belief framebuffer map: {e}"))?;
        let row_bytes = (w * BYTES_PER_PIXEL) as usize;
        let mut rgba = Vec::with_capacity(row_bytes * h as usize);
        for row in mapped
            .chunks(target.padded_bytes_per_row as usize)
            .take(h as usize)
        {
            rgba.extend_from_slice(&row[..row_bytes]);
        }
        if matches!(self.pixel_order, PixelOrder::Bgra) {
            for pixel in rgba.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
                pixel.swap(0, 2);
            }
        }
        drop(mapped);
        buffer.unmap();
        Ok(CapturedFrame { width: w, height: h, rgba })
    }

    /// N0.j S13.2 ON-DEMAND READBACK: read the CURRENT offscreen texture (the
    /// last presented net frame — the offscreen blit runs every frame) back to
    /// the CPU, only when a bare `/scry` actually asks. This replaces the old
    /// per-frame readback that fed `latest`: no re-trace, no demod, just the one
    /// copy+map the viewer needs. Runs on the render thread (owns the device).
    fn capture_presented(&mut self) -> Result<CapturedFrame, String> {
        let (w, h) = (self.offscreen.width, self.offscreen.height);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("presented on-demand readback"),
            });
        let slot = &self.offscreen.slots[0];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.offscreen.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.offscreen.padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let buffer = slot.buffer.clone();
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = done_tx.send(result.map(|_| ()).map_err(|e| e.to_string()));
        });
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        done_rx
            .recv()
            .map_err(|e| format!("presented readback channel closed: {e}"))??;
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|e| format!("presented framebuffer map: {e}"))?;
        let row_bytes = (w * BYTES_PER_PIXEL) as usize;
        let mut rgba = Vec::with_capacity(row_bytes * h as usize);
        for row in mapped
            .chunks(self.offscreen.padded_bytes_per_row as usize)
            .take(h as usize)
        {
            rgba.extend_from_slice(&row[..row_bytes]);
        }
        if matches!(self.pixel_order, PixelOrder::Bgra) {
            for pixel in rgba.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
                pixel.swap(0, 2);
            }
        }
        drop(mapped);
        buffer.unmap();
        Ok(CapturedFrame { width: w, height: h, rgba })
    }

    /// S12.5: the live per-stage budget JSON (`/budget`) and forward-state JSON
    /// (`/state`), or an honest "net-present off" stub.
    fn debug_budget_json(&self) -> String {
        // N0.j S13: splice the OUTSIDE-work block into the net budget JSON so
        // `/budget` carries both the GPU stage table AND the ~9 ms non-net
        // frame-loop segments (world/readback/http/loop_total) in one door.
        #[cfg(target_os = "macos")]
        if let Some(np) = &self.net_present {
            let base = np.budget_json();
            let outside = self.outside.json();
            let stages = self.outside.world_stages_json();
            return match base.strip_suffix('}') {
                Some(head) => format!("{head},{outside},{stages}}}"),
                None => base,
            };
        }
        format!(
            "{{\"frames\":0,\"note\":\"net-present off\",{},{}}}",
            self.outside.json(),
            self.outside.world_stages_json()
        )
    }

    fn debug_state_json(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(np) = &self.net_present {
            return np.state_json();
        }
        format!(
            "{{\"path\":\"raster\",\"canvas\":[{},{}],\"note\":\"net-present off\"}}",
            self.config.width, self.config.height
        )
    }

    /// The moving eye: integrate `capture_frames` accumulation frames from an
    /// arbitrary pose to a per-request offscreen target and read it back. Runs on
    /// the render thread; the surface loop's own accumulation is untouched.
    fn capture_pose(&mut self, params: &ScryParams) -> Result<CapturedFrame, String> {
        let width = self.canvas_width;
        let height = self.canvas_height;
        let fov = match params.fov {
            Some(degrees) => {
                if !(degrees > 0.0 && degrees < 180.0) {
                    return Err("fov must be between 0 and 180 degrees".into());
                }
                degrees.to_radians()
            }
            None => self.camera.fov_y_radians,
        };
        let camera = Camera {
            eye: params.pos.map(Vec3::from_array).unwrap_or(self.camera.eye),
            yaw: params.yaw.unwrap_or(self.camera.yaw),
            pitch: params.pitch.unwrap_or(self.camera.pitch),
            fov_y_radians: fov,
            near: self.camera.near,
            far: self.camera.far,
        };

        // ITEM 16 (de-charter): the trace→denoise→upscale chain exists only as
        // an explicitly named teacher/benchmark LAB surface. Neither a missing
        // query parameter nor a `resolve` selector can enter it.
        if params.teacher_benchmark {
            return self.capture_pose_teacher_benchmark(&camera, width, height);
        }
        // Fixed canvas capture is the same nearest present path as the window.

        // OWN-EYE CULL — the persistent `self.integrator` buffers already carry
        // the OWN-eye-culled geometry (`advance_world` keeps them in lockstep
        // with `self.camera.eye`, the walker's own eye). A default `/scry` (no
        // `pos` override) IS that same eye, so the fast path below needs no
        // rebuild. An EXPLICIT moving eye (`?pos=...`) may be a FOREIGN eye —
        // any eye that is not the walker's own must still see her — so when one
        // is given we build a throwaway splice for THIS capture only and
        // restore the persistent (own-eye) buffers before returning, so the
        // live window's next frame is unaffected.
        let foreign_splice = params.pos.map(|_| {
            let tris = self.scene.dynamic_leaf_triangles_for_eye(
                camera.eye,
                scrying_glass::scene::OWN_EYE_EPSILON_M,
                self.draw_own_body,
            );
            DynamicSplice::build(
                &self.static_bvh,
                &tris,
                &self.bvh_params.dynamic(),
                self.refit_params,
            )
        });
        if let Some(foreign) = &foreign_splice {
            self.integrator.update_bvh(&self.device, &foreign.merged);
        }
        // Whatever happens below (success or an early `?` error), put the
        // persistent own-eye-culled buffers back before this function returns
        // — the live window's next frame must never see the foreign geometry.
        let result = self.capture_pose_fixed(&camera, width, height);
        if foreign_splice.is_some() {
            self.integrator
                .update_bvh(&self.device, &self.splice.merged);
        }
        result
    }

    /// `/retina`: exact primary rays over the tracer's post-transmute leaf
    /// geometry; no framebuffer, radiance, or secondary-ray path is involved.
    fn capture_retina(&mut self, params: &RetinaParams) -> Result<String, String> {
        let width = params.width;
        let height = params.height;
        let fov = match params.pose.fov {
            Some(degrees) if degrees > 0.0 && degrees < 180.0 => degrees.to_radians(),
            Some(_) => return Err("fov must be between 0 and 180 degrees".into()),
            None => self.camera.fov_y_radians,
        };
        let camera = Camera {
            eye: params.pose.pos.map(Vec3::from_array).unwrap_or(self.camera.eye),
            yaw: params.pose.yaw.unwrap_or(self.camera.yaw), pitch: params.pose.pitch.unwrap_or(self.camera.pitch),
            fov_y_radians: fov, near: self.camera.near, far: self.camera.far,
        };
        let culls_own_body = self.scene.retina_culls_own_body(camera.eye, scrying_glass::scene::OWN_EYE_EPSILON_M, self.draw_own_body);
        let scene = &self.scene;
        let (bvh, ordered_tags) = self.retina_cache.get_or_build(
            self.retina_epoch, culls_own_body, &self.bvh_params,
            || scene.retina_triangles_for_eye(camera.eye, scrying_glass::scene::OWN_EYE_EPSILON_M, self.draw_own_body),
        );
        let base = retina::trace(bvh, ordered_tags, &camera, width, height, params.layers);
        let fovea = params.fovea.iter().map(|level| serde_json::json!({
            "center": level.center, "radius": level.radius, "scale": level.scale,
            "image": retina::trace_window(bvh, ordered_tags, &camera, width.saturating_mul(level.scale), height.saturating_mul(level.scale), params.layers, level.center, level.radius),
        })).collect::<Vec<_>>();
        serde_json::to_string(&serde_json::json!({"base": base, "fovea": fovea})).map_err(|error| error.to_string())
    }

    /// Fixed-canvas `/scry` dispatch + readback — split out of
    /// [`Renderer::capture_pose`] so its OWN-EYE CULL restore always runs.
    fn capture_pose_fixed(
        &mut self,
        camera: &Camera,
        width: u32,
        height: u32,
    ) -> Result<CapturedFrame, String> {
        let accum = self.integrator.make_accum(&self.device, width, height);
        let compute_bg = self.integrator.compute_bind_group(&self.device, &accum);
        let blit_bg = self.integrator.blit_bind_group(&self.device, &accum);

        let mut samples_before = 0u32;
        for _ in 0..self.capture_frames.max(1) {
            let uniform = IntegratorUniform::build(
                &camera,
                &self.sun,
                self.sky_top,
                self.sky_horizon,
                width,
                height,
                self.integrator.node_count,
                self.integrator.tri_count,
                samples_before,
                &self.int_params,
                None,
            );
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scry integrate"),
                });
            self.integrator.dispatch(
                &self.queue,
                &mut encoder,
                &uniform,
                &compute_bg,
                width,
                height,
            );
            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            samples_before += self.int_params.spp;
        }

        // Present the converged mean to a fresh fixed-canvas sRGB target.
        let mut uniform = IntegratorUniform::build(
            camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            width,
            height,
            self.integrator.node_count,
            self.integrator.tri_count,
            samples_before,
            &self.int_params,
            None,
        );
        uniform.surface = [width, height, 1, 0];
        self.queue.write_buffer(
            &self.integrator.uniform_buf,
            0,
            bytemuck::bytes_of(&uniform),
        );
        let target = OffscreenTarget::new(&self.device, self.config.format, width, height);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scry present + capture"),
            });
        self.integrator
            .blit(&mut encoder, &target.view, &blit_bg, "scry present");
        let slot = &target.slots[0];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let buffer = slot.buffer.clone();
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
        let callback_buffer = buffer.clone();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let mapped = result.map_err(|error| error.to_string());
            if mapped.is_err() {
                let _ = callback_buffer;
            }
            let _ = done_tx.send(mapped.map(|_| ()));
        });
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        done_rx
            .recv()
            .map_err(|error| format!("scry readback channel closed: {error}"))??;
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|error| format!("scry framebuffer map: {error}"))?;
        let row_bytes = (width * BYTES_PER_PIXEL) as usize;
        let mut rgba = Vec::with_capacity(row_bytes * height as usize);
        for row in mapped
            .chunks(target.padded_bytes_per_row as usize)
            .take(height as usize)
        {
            rgba.extend_from_slice(&row[..row_bytes]);
        }
        if matches!(self.pixel_order, PixelOrder::Bgra) {
            for pixel in rgba.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
                pixel.swap(0, 2);
            }
        }
        drop(mapped);
        buffer.unmap();
        Ok(CapturedFrame {
            width,
            height,
            rgba,
        })
    }

    /// TEACHER/BENCHMARK LAB SURFACE (ITEM 16): trace(low, 1 spp) → GPU
    /// denoise → GPU neural upscale → 1:1 present → readback. This historical
    /// chain is de-chartered: only `GET /scry?lab=teacher-benchmark` enters it;
    /// no present-path or resolve default can select it. The sequence remains
    /// available for the headless proofs in `examples/onepath_proof.rs` and the
    /// viii2/viii3 ordeals. It traces the STATIC BVH (geometry-only AOV guide +
    /// radiance); dynamics are absent, appropriate for a resolve-quality lab
    /// comparison and never represented as live output.
    fn capture_pose_teacher_benchmark(
        &mut self,
        camera: &Camera,
        width: u32,
        height: u32,
    ) -> Result<CapturedFrame, String> {
        let (low_w, low_h) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let denoiser = GpuDenoiser::new(
            &self.device,
            &deserialize_denoiser_weights(
                &std::fs::read(data_dir.join("denoiser-weights-v1.bin"))
                    .map_err(|e| format!("read denoiser weights: {e}"))?,
            )
            .ok_or_else(|| "deserialize denoiser weights".to_string())?,
        );
        let upscaler = GpuUpscaler::new(
            &self.device,
            &deserialize_upscaler_weights(
                &std::fs::read(data_dir.join("upscaler-weights-v1.bin"))
                    .map_err(|e| format!("read upscaler weights: {e}"))?,
            )
            .ok_or_else(|| "deserialize upscaler weights".to_string())?,
        );

        // trace(low, 1 spp) noisy radiance + low/hi geometry AOVs.
        let noisy_params = IntegratorParams {
            spp: 1,
            ..self.int_params.clone()
        };
        let low_noisy = resolve_accum(&trace_headless(
            &self.device,
            &self.queue,
            &self.static_bvh,
            camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            low_w,
            low_h,
            1,
            &noisy_params,
            None,
        ));
        let (low_alb, low_nrm, low_dep) = split_aov(&trace_headless_aov(
            &self.device,
            &self.queue,
            &self.static_bvh,
            camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            low_w,
            low_h,
        ));
        let (hi_alb, hi_nrm, hi_dep) = split_aov(&trace_headless_aov(
            &self.device,
            &self.queue,
            &self.static_bvh,
            camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            width,
            height,
        ));

        // denoise(low) → upscale(→ surface) — the neural resolve.
        let denoised = denoiser.denoise(
            &self.device,
            &self.queue,
            &low_noisy,
            &low_alb,
            &low_nrm,
            &low_dep,
            low_w,
            low_h,
        );
        let neural = upscaler.upscale(
            &self.device,
            &self.queue,
            &denoised,
            low_w,
            low_h,
            &hi_alb,
            &hi_nrm,
            &hi_dep,
            width,
            height,
        );

        // Present: upload the full-res linear image into a surface-sized accum
        // (w = 1 sample) and 1:1 nearest-blit it to a fresh sRGB target — the
        // SAME colour pipeline (linear accum → sRGB target OETF) the plain
        // capture uses, so the A/B differs only in the resolve.
        let cells: Vec<[f32; 4]> = neural.iter().map(|c| [c.x, c.y, c.z, 1.0]).collect();
        let present = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("teacher benchmark present accum"),
            size: (cells.len() * 16).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&present, 0, bytemuck::cast_slice(&cells));
        let blit_bg = self.integrator.blit_bind_group(&self.device, &present);
        let mut uniform = IntegratorUniform::build(
            camera,
            &self.sun,
            self.sky_top,
            self.sky_horizon,
            width,
            height,
            self.integrator.node_count,
            self.integrator.tri_count,
            0,
            &self.int_params,
            None,
        );
        uniform.surface = [width, height, 1, 0]; // nearest, 1:1 — no re-scale.
        self.queue.write_buffer(
            &self.integrator.uniform_buf,
            0,
            bytemuck::bytes_of(&uniform),
        );

        let target = OffscreenTarget::new(&self.device, self.config.format, width, height);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("teacher benchmark present + capture"),
            });
        self.integrator.blit(
            &mut encoder,
            &target.view,
            &blit_bg,
            "teacher benchmark present",
        );
        let slot = &target.slots[0];
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let buffer = slot.buffer.clone();
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();
        let callback_buffer = buffer.clone();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let mapped = result.map_err(|error| error.to_string());
            if mapped.is_err() {
                let _ = callback_buffer;
            }
            let _ = done_tx.send(mapped.map(|_| ()));
        });
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        done_rx
            .recv()
            .map_err(|error| format!("teacher benchmark readback channel closed: {error}"))??;
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|error| format!("teacher benchmark framebuffer map: {error}"))?;
        let row_bytes = (width * BYTES_PER_PIXEL) as usize;
        let mut rgba = Vec::with_capacity(row_bytes * height as usize);
        for row in mapped
            .chunks(target.padded_bytes_per_row as usize)
            .take(height as usize)
        {
            rgba.extend_from_slice(&row[..row_bytes]);
        }
        if matches!(self.pixel_order, PixelOrder::Bgra) {
            for pixel in rgba.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
                pixel.swap(0, 2);
            }
        }
        drop(mapped);
        buffer.unmap();
        Ok(CapturedFrame {
            width,
            height,
            rgba,
        })
    }
}

/// Optional moving-eye overrides parsed from `GET /scry?...`.
/// All absent = exactly the default spawn-pose capture.
#[derive(Clone, Debug, Default)]
struct ScryParams {
    pos: Option<[f32; 3]>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    fov: Option<f32>,
    /// Explicit lab gate for the de-chartered teacher/benchmark chain.
    teacher_benchmark: bool,
    width: Option<u32>,
    height: Option<u32>,
    /// S12.5 AI DEBUG DOOR: which eye to serve. `false`/absent = presented (the
    /// live net-present frame / a pose capture); `true` = belief (the net's raw
    /// radiance, re-demodded from THIS frame's net output with no albedo).
    belief: bool,
    /// The resolve to capture with: 0 bilinear, 1 nearest, 2 neural. Absent =
    /// the window's GAIA_NATIVE_UPSCALE default. THE ONE RENDER PATH A/B knob.
    resolve: Option<u32>,
    /// N0.j S13.2: serve the CURRENT presented frame by an ON-DEMAND readback of
    /// the offscreen texture (no re-trace) — set by the http handler for a bare
    /// `/scry`, consumed by the render loop (`capture_presented`).
    presented: bool,
}

struct ScryRequest {
    params: ScryParams,
    reply: mpsc::Sender<Result<CapturedFrame, String>>,
}

enum RenderRequest {
    Scry(ScryRequest),
    Retina { params: RetinaParams, reply: mpsc::Sender<Result<String, String>> },
}

#[derive(Clone, Debug)]
struct FoveaParams {
    center: [f32; 2],
    radius: f32,
    scale: u32,
}

#[derive(Clone, Debug)]
struct RetinaParams {
    pose: ScryParams,
    width: u32,
    height: u32,
    layers: RetinaLayers,
    fovea: Vec<FoveaParams>,
}

fn parse_retina_query(query: &str) -> Result<RetinaParams, String> {
    let mut pose = ScryParams::default();
    let mut width = 64;
    let mut height = 64;
    let mut layers = RetinaLayers { depth: true, normal: true, entity_id: true, material_id: true, world_pos: true };
    let mut fovea = Vec::new();
    for pair in query.split('&').filter(|part| !part.is_empty()) {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("query segment {pair:?} must be key=value"))?;
        if key == "fovea" {
            for level in value.split(';') {
                let values = level.split(',').collect::<Vec<_>>();
                if values.len() != 4 { return Err("fovea must be center_x,center_y,radius,scale (semicolon separates levels)".into()); }
                let center = [parse_finite_f32(values[0], "fovea.center_x")?, parse_finite_f32(values[1], "fovea.center_y")?];
                let radius = parse_finite_f32(values[2], "fovea.radius")?;
                let scale = values[3].parse::<u32>().map_err(|_| format!("fovea.scale must be a positive integer, got {:?}", values[3]))?;
                if !(0.0..=1.0).contains(&center[0]) || !(0.0..=1.0).contains(&center[1]) || !(radius > 0.0 && radius <= 1.0) || scale == 0 { return Err("fovea needs center in 0..=1, radius in 0..=1, scale > 0".into()); }
                fovea.push(FoveaParams { center, radius, scale });
            }
            continue;
        }
        if key == "layers" {
            layers = RetinaLayers::default();
            for layer in value.split(',') {
                match layer { "depth" => layers.depth = true, "normal" => layers.normal = true, "entity-id" | "entity_id" => layers.entity_id = true, "material-id" | "material_id" => layers.material_id = true, "world-pos" | "world_pos" => layers.world_pos = true, "motion" => return Err("motion is UNVERIFIED: no previous-frame plumbing".into()), other => return Err(format!("unknown retina layer {other:?}")) }
            }
            continue;
        }
        if key == "w" || key == "h" {
            let dimension = value
                .parse::<u32>()
                .ok()
                .filter(|dimension| *dimension > 0)
                .ok_or_else(|| format!("{key} must be a positive integer, got {value:?}"))?;
            if key == "w" { width = dimension; } else { height = dimension; }
            continue;
        }
        let one = parse_scry_query(pair)?;
        if one.pos.is_some() { pose.pos = one.pos; }
        if one.yaw.is_some() { pose.yaw = one.yaw; }
        if one.pitch.is_some() { pose.pitch = one.pitch; }
        if one.fov.is_some() { pose.fov = one.fov; }
    }
    Ok(RetinaParams { pose, width, height, layers, fovea })
}

fn parse_finite_f32(value: &str, name: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("{name} must be a number, got {value:?}"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("{name} must be finite, got {value:?}"))
    }
}

fn parse_scry_query(query: &str) -> Result<ScryParams, String> {
    let mut params = ScryParams::default();
    for pair in query.split('&').filter(|segment| !segment.is_empty()) {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("query segment {pair:?} must be key=value"))?;
        match key {
            "pos" => {
                let coords: Vec<&str> = value.split(',').collect();
                if coords.len() != 3 {
                    return Err(format!("pos must be x,y,z, got {value:?}"));
                }
                params.pos = Some([
                    parse_finite_f32(coords[0], "pos.x")?,
                    parse_finite_f32(coords[1], "pos.y")?,
                    parse_finite_f32(coords[2], "pos.z")?,
                ]);
            }
            "yaw" => params.yaw = Some(parse_finite_f32(value, "yaw")?),
            "pitch" => params.pitch = Some(parse_finite_f32(value, "pitch")?),
            "fov" => params.fov = Some(parse_finite_f32(value, "fov")?),
            "lab" => match value.trim().to_ascii_lowercase().as_str() {
                "teacher-benchmark" => params.teacher_benchmark = true,
                other => {
                    return Err(format!("lab must be teacher-benchmark, got {other:?}"));
                }
            },
            "resolve" => {
                params.resolve = Some(match value.trim().to_ascii_lowercase().as_str() {
                    "bilinear" => 0,
                    "nearest" => 1,
                    "neural" => 2,
                    other => {
                        return Err(format!(
                            "resolve must be bilinear, nearest, or neural, got {other:?}"
                        ));
                    }
                });
            }
            "w" => {
                params.width = Some(
                    value
                        .parse::<u32>()
                        .ok()
                        .filter(|width| *width > 0)
                        .ok_or_else(|| format!("w must be a positive integer, got {value:?}"))?,
                )
            }
            "h" => {
                params.height = Some(
                    value
                        .parse::<u32>()
                        .ok()
                        .filter(|height| *height > 0)
                        .ok_or_else(|| format!("h must be a positive integer, got {value:?}"))?,
                )
            }
            "eye" => match value.trim().to_ascii_lowercase().as_str() {
                "presented" | "present" => params.belief = false,
                "belief" => params.belief = true,
                other => {
                    return Err(format!("eye must be presented or belief, got {other:?}"));
                }
            },
            other => return Err(format!("unknown scry parameter {other:?}")),
        }
    }
    Ok(params)
}

struct RuntimeState {
    running: Arc<AtomicBool>,
}

#[tauri::command]
fn panel_pressed() {
    eprintln!("[ipc] transparent overlay button -> Rust command");
}

#[cfg(target_os = "macos")]
fn install_passthrough_monitor(
    window: tauri::Window,
    config: ScryingGlassConfig,
) -> Result<(), String> {
    use block2::RcBlock;
    use objc2_app_kit::{NSEvent, NSEventMask};

    let click_window = window.clone();
    let block = RcBlock::new(move |event: std::ptr::NonNull<NSEvent>| -> *mut NSEvent {
        let event = unsafe { event.as_ref() };
        let point = event.locationInWindow();
        if let Ok(size) = click_window.inner_size()
            && !config.is_panel_point(point.x, point.y, size)
        {
            eprintln!(
                "[wgpu-input] passthrough click x={:.1} y={:.1}",
                point.x, point.y
            );
        }
        event as *const NSEvent as *mut NSEvent
    });
    let monitor = unsafe {
        NSEvent::addLocalMonitorForEventsMatchingMask_handler(NSEventMask::LeftMouseDown, &block)
    }
    .ok_or_else(|| "failed to install macOS local mouse monitor".to_string())?;
    Box::leak(Box::new(monitor));
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn install_passthrough_monitor(
    _window: tauri::Window,
    _config: ScryingGlassConfig,
) -> Result<(), String> {
    Err("this package's physical native click monitor is macOS-only".into())
}

fn main() {
    let config = ScryingGlassConfig::from_env()
        .unwrap_or_else(|error| panic!("invalid scrying-glass config: {error}"));
    let render_interval = config.frame_interval();
    let native_port = config.native_port;
    let mut core = Core::default();
    ScryingGlassPackage.register(&mut core);
    eprintln!(
        "[package] {} v{} registered",
        core.package("scrying-glass").unwrap().name,
        core.package("scrying-glass").unwrap().version
    );
    let world_core = WorldCore::open(&config.world_path, config.world_core.clone())
        .unwrap_or_else(|error| panic!("open GAIA_WORLD {}: {error}", config.world_path.display()));
    world_core
        .materialize_into(&mut core.world)
        .unwrap_or_else(|error| panic!("materialize crystal authority: {error}"));
    let scene_names: Vec<String> = world_core
        .realm()
        .scene_names()
        .map(str::to_owned)
        .collect();
    let entity_count = world_core.realm().authored_entity_count();
    let chain_start = Instant::now();
    let render_scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &config.scene)
        .unwrap_or_else(|error| panic!("materialize GAIA world render: {error}"));
    let chain_millis = chain_start.elapsed().as_secs_f64() * 1e3;
    let cluster_count: usize = render_scene
        .chains
        .iter()
        .map(|chain| chain.dag.clusters.len())
        .sum();
    // Load budget: time to transmute the whole realm into the Great Chain
    // (printed, never gated — Rite III ordeal item 5).
    eprintln!(
        "[world] {} scene(s)={:?} entities={} chains={} clusters={} transmute={chain_millis:.1}ms",
        config.world_path.display(),
        scene_names,
        entity_count,
        render_scene.chains.len(),
        cluster_count,
    );

    // WINDOW-BAN OFFSCREEN mode: no NSWindow, no tauri/winit surface. Build the
    // renderer headless, serve /scry over HTTP off the offscreen texture, and
    // drive the render loop on this thread until killed. This is the mandated
    // proof surface — measurement runs never open a window on the desktop.
    if config.offscreen {
        run_offscreen(config, render_scene);
    }

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![panel_pressed])
        .setup(move |app| {
            let window = tauri::window::WindowBuilder::new(app, "wgpu-surface")
                .title(config.title.clone())
                .inner_size(config.window_width, config.window_height)
                // Nekromant case #1 fix: `focused(false)` skips the initial
                // makeKeyAndOrderFront (window shows via orderFront only, never
                // key at creation); `focusable(false)` rides tao's
                // canBecomeKeyWindow/canBecomeMainWindow override down to a
                // permanent `false` (packages/scrying-glass Cargo.toml pins
                // tauri 2.11.5 -> tao 0.35.3; see
                // tao-0.35.3/src/platform_impl/macos/window.rs WINDOW_CLASS) —
                // no NSWindow subclass of our own needed, no keystroke
                // (including Cmd+Q) can ever land on this window again,
                // however hard a GPU-load activation storm hits the app.
                .focused(!config.worker_window)
                .focusable(!config.worker_window)
                .build()?;
            if config.worker_window {
                eprintln!(
                    "[worker-window] GAIA_NATIVE_WORKER_WINDOW=true: window built focused=false \
                     focusable=false (never-key) title={:?}",
                    config.title
                );
            }
            let size = window.inner_size()?;
            let (position, panel_size) = config.panel_layout(size);
            let auto_test_ipc = config.auto_test_ipc;
            let overlay = tauri::webview::WebviewBuilder::new(
                "overlay-panel",
                WebviewUrl::App("index.html".into()),
            )
            .transparent(true)
            .on_page_load(move |webview, payload| {
                if auto_test_ipc
                    && matches!(payload.event(), tauri::webview::PageLoadEvent::Finished)
                {
                    let _ = webview.eval("document.querySelector('#ipc')?.click()");
                }
            });
            let overlay = window.add_child(overlay, position, panel_size)?;
            let resize_overlay = overlay.clone();
            let resize_config = config.clone();
            window.on_window_event(move |event| {
                match event {
                    tauri::WindowEvent::Resized(size) => {
                        let (position, panel_size) = resize_config.panel_layout(*size);
                        let _ = resize_overlay.set_position(position);
                        let _ = resize_overlay.set_size(panel_size);
                    }
                    // ALWAYS-ON instrumentation (both worker_window modes): every
                    // future quit-by-stolen-focus now has a named sender — a
                    // wall-clock-stamped log line for the exact moment this
                    // window gained/lost key status.
                    tauri::WindowEvent::Focused(focused) => {
                        let stamp_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis())
                            .unwrap_or(0);
                        eprintln!(
                            "[focus] t={stamp_ms}ms window=\"wgpu-surface\" focused={focused} \
                             worker_window={}",
                            resize_config.worker_window
                        );
                    }
                    _ => {}
                }
            });
            install_passthrough_monitor(window.clone(), config.clone())
                .map_err(std::io::Error::other)?;

            let latest = Arc::new(RwLock::new(None));
            let capture_sender = spawn_capture_worker(latest.clone());

            // The Embodiment: the world's own leaf triangles become the floor
            // (exact geometry, view-independent — never a camera's coarse cut),
            // and the world spawn pose becomes a walking body. IRON SWEEP:
            // floor cutoff / probe count / column epsilon are `PlayerParams`
            // fields (env-overridable), so `player_params` is read before the
            // floor set is built and threaded through explicitly — defaults
            // reproduce the old `Ground::from_positions` behavior exactly.
            let player_params = PlayerParams::from_env().map_err(std::io::Error::other)?;
            let ground = Arc::new(Ground::from_positions_with_params(
                &render_scene.leaf_positions(),
                &player_params,
            ));
            // The spawn eye pose defaults to the world's own spawn component; each
            // axis + yaw may be overridden by an explicit env param so the window
            // the Architect opens faces the realm (item 4 vantage). No frozen
            // world edit — the override is window-local and param-driven (unset =
            // the world spawn, unchanged). The body still FALLS to the floor from
            // whatever eye Y is given, so a spawn point on the plaza reads naruko
            // (lighthouse/pier/city) instead of the occluded default corner.
            let spawn_axis = |name: &str, world: f32| -> Result<f32, String> {
                match std::env::var(name) {
                    Ok(value) => value
                        .parse::<f32>()
                        .map_err(|_| format!("{name} must be a number, got {value:?}"))
                        .and_then(|parsed| {
                            if parsed.is_finite() {
                                Ok(parsed)
                            } else {
                                Err(format!("{name} must be finite, got {value:?}"))
                            }
                        }),
                    Err(_) => Ok(world),
                }
            };
            let world_eye = render_scene.camera.eye;
            let spawn_eye = Vec3::new(
                spawn_axis("GAIA_NATIVE_SPAWN_X", world_eye.x).map_err(std::io::Error::other)?,
                spawn_axis("GAIA_NATIVE_SPAWN_Y", world_eye.y).map_err(std::io::Error::other)?,
                spawn_axis("GAIA_NATIVE_SPAWN_Z", world_eye.z).map_err(std::io::Error::other)?,
            );
            let spawn_yaw =
                spawn_axis("GAIA_NATIVE_SPAWN_YAW", render_scene.camera.yaw)
                    .map_err(std::io::Error::other)?;
            let player = Arc::new(Mutex::new(Player::new(
                player_params,
                spawn_eye,
                spawn_yaw,
            )));
            let tick_dt = (1.0 / config.fps) as f32;
            eprintln!(
                "[embodiment] spawn eye={spawn_eye:?} yaw={spawn_yaw} floor_triangles={} tick_dt={tick_dt}",
                ground.triangle_count()
            );
            input::install_player_input(player.clone()).map_err(std::io::Error::other)?;

            let renderer = Renderer::new(
                Some(&window),
                (config.window_width as u32, config.window_height as u32),
                capture_sender,
                render_scene,
                config.integrator,
                &config.bvh,
                config.refit,
                config.capture_frames,
                config.draw_own_body,
                config.temporal_enabled,
                config.temporal,
                config.native_canvas_width,
                config.native_canvas_height,
                config.net_present,
            )
            .map_err(std::io::Error::other)?;
            // DAS BLUTBÄNDIGEN — B0 DATA DOOR. Seed the live bend state from the
            // boot scene bytes and, when the master switch is on, spawn the
            // mtime file-watch (law 1: polling, no new crate). The receiver is
            // drained on the render thread, which owns the device + scene.
            let bloodbend_params = BloodbendParams::from_env(&config.world_path)
                .map_err(std::io::Error::other)?;
            let bend_rx = if bloodbend_params.enabled {
                Some(bloodbend::spawn_watcher(&bloodbend_params))
            } else {
                eprintln!("[bloodbend] master switch GAIA_NATIVE_BLOODBEND=false — data door closed");
                None
            };
            let bloodbend_state = bloodbend_params.enabled.then(|| {
                Bloodbend::seed(
                    bloodbend_params.clone(),
                    config.world_path.clone(),
                    config.scene.clone(),
                )
            });
            {
                // MEASURE: print fixed God's-canvas trace cost.
                let mut renderer = renderer;
                renderer.bloodbend = bloodbend_state;
                let (median, mean) = renderer.measure_trace_ms(60);
                eprintln!(
                    "[frame] trace {}x{} God's canvas: median {median:.2}ms mean {mean:.2}ms/frame (spp={}, 60-frame sample)",
                    renderer.canvas_width,
                    renderer.canvas_height,
                    config.integrator.spp,
                );
                let renderer_moved = renderer;
                let (scry_tx, scry_rx) = mpsc::channel::<RenderRequest>();
                let (world_tx, world_rx) = mpsc::channel::<WorldRequest>();
                let debug: DebugCell = Arc::new(RwLock::new(DebugSnapshot::default()));
                start_screenshot_server(
                    native_port,
                    HttpContext {
                        latest,
                        scry: scry_tx,
                        world: world_tx,
                        authority_timeout: config.authority_timeout,
                        event_default_limit: config.event_default_limit,
                        event_limit_max: config.event_limit_max,
                        max_request_bytes: config.max_request_bytes,
                        player: player.clone(),
                        ground: ground.clone(),
                        tick_dt,
                        debug: debug.clone(),
                    },
                )
                .map_err(std::io::Error::other)?;
                let running = Arc::new(AtomicBool::new(true));
                app.manage(RuntimeState {
                    running: running.clone(),
                });
                let render_player = player.clone();
                let render_ground = ground.clone();
                let hud_overlay = overlay.clone();
                let hud_enabled = config.hud_enabled;
                let hud_window = config.hud_window;
                let authority_scene_params = config.scene.clone();
                thread::Builder::new()
                    .name("gaia-render".into())
                    .spawn(move || {
                        let mut renderer = renderer_moved;
                        let mut world_core = world_core;
                        run_render_loop(
                            &mut renderer,
                            &mut world_core,
                            &authority_scene_params,
                            &world_rx,
                            &window,
                            &render_player,
                            &render_ground,
                            tick_dt,
                            render_interval,
                            &scry_rx,
                            bend_rx.as_ref(),
                            &running,
                            &hud_overlay,
                            hud_enabled,
                            hud_window,
                            &debug,
                        );
                    })
                    .map_err(std::io::Error::other)?;
            }
            eprintln!(
                "[scrying-glass] child webview overlay created; render, capture, PNG, and HTTP run off the main thread"
            );
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Tauri app build failed")
        .run(|app, event| {
            match event {
                tauri::RunEvent::Exit => {
                    app.state::<RuntimeState>()
                        .running
                        .store(false, Ordering::Release);
                }
                // ALWAYS-ON instrumentation: applicationShouldHandleReopen is
                // the app-activation signal Tauri's RunEvent exposes on macOS
                // (dock-icon/Cmd+Tab reactivation) — the closest named sender
                // to "setApplicationIsActive" the public API surfaces (proven
                // seam: tauri-2.11.5/src/app.rs RunEvent::Reopen). Per-window
                // key/focus transitions are logged in the window's own
                // on_window_event above (Focused).
                #[cfg(target_os = "macos")]
                tauri::RunEvent::Reopen { has_visible_windows, .. } => {
                    let stamp_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    eprintln!(
                        "[activation] t={stamp_ms}ms event=Reopen has_visible_windows={has_visible_windows}"
                    );
                }
                _ => {}
            }
        });
}

/// WINDOW-BAN OFFSCREEN driver: no NSWindow, no tauri/winit. Builds a
/// surface-less renderer, serves `/scry` (+ the S12.5 door) off the offscreen
/// texture, and runs the render loop on this thread until the process is
/// killed. The mandated proof surface for measurement runs.
fn run_offscreen(config: ScryingGlassConfig, render_scene: RenderScene) -> ! {
    let native_port = config.native_port;
    let render_interval = config.frame_interval();
    let tick_dt = (1.0 / config.fps) as f32;
    let dims = (config.window_width as u32, config.window_height as u32);

    let ground = Arc::new(Ground::from_positions(&render_scene.leaf_positions()));
    let spawn_axis = |name: &str, world: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|p| p.is_finite())
            .unwrap_or(world)
    };
    let world_eye = render_scene.camera.eye;
    let spawn_eye = Vec3::new(
        spawn_axis("GAIA_NATIVE_SPAWN_X", world_eye.x),
        spawn_axis("GAIA_NATIVE_SPAWN_Y", world_eye.y),
        spawn_axis("GAIA_NATIVE_SPAWN_Z", world_eye.z),
    );
    let spawn_yaw = spawn_axis("GAIA_NATIVE_SPAWN_YAW", render_scene.camera.yaw);
    let player_params =
        PlayerParams::from_env().unwrap_or_else(|e| panic!("offscreen player params: {e}"));
    let player = Arc::new(Mutex::new(Player::new(player_params, spawn_eye, spawn_yaw)));
    eprintln!(
        "[embodiment] spawn eye={spawn_eye:?} yaw={spawn_yaw} floor_triangles={} tick_dt={tick_dt}",
        ground.triangle_count()
    );

    let latest: LatestFrame = Arc::new(RwLock::new(None));
    let capture_sender = spawn_capture_worker(latest.clone());
    let mut renderer = Renderer::new(
        None,
        dims,
        capture_sender,
        render_scene,
        config.integrator,
        &config.bvh,
        config.refit,
        config.capture_frames,
        config.draw_own_body,
        config.temporal_enabled,
        config.temporal,
        config.native_canvas_width,
        config.native_canvas_height,
        config.net_present,
    )
    .unwrap_or_else(|e| panic!("offscreen renderer: {e}"));

    let (median, mean) = renderer.measure_trace_ms(60);
    eprintln!(
        "[frame] trace {}x{} → offscreen {}x{}: median {median:.2}ms mean {mean:.2}ms/frame (spp={}, 60-frame sample)",
        renderer.canvas_width, renderer.canvas_height, dims.0, dims.1, config.integrator.spp,
    );

    let (scry_tx, scry_rx) = mpsc::channel::<RenderRequest>();
    // WINDOW-BAN offscreen is a headless measurement/proof surface: it serves
    // /scry but not live world ops. The channel exists only to satisfy
    // HttpContext (so /world endpoints don't 500) — world_rx is intentionally
    // never drained here, matching this mode's original (pre-remap) scope.
    let (world_tx, _world_rx) = mpsc::channel::<WorldRequest>();
    let debug: DebugCell = Arc::new(RwLock::new(DebugSnapshot::default()));
    start_screenshot_server(
        native_port,
        HttpContext {
            latest,
            scry: scry_tx,
            world: world_tx,
            authority_timeout: config.authority_timeout,
            event_default_limit: config.event_default_limit,
            event_limit_max: config.event_limit_max,
            max_request_bytes: config.max_request_bytes,
            player: player.clone(),
            ground: ground.clone(),
            tick_dt,
            debug: debug.clone(),
        },
    )
    .unwrap_or_else(|e| panic!("offscreen http server: {e}"));
    eprintln!(
        "[offscreen] GAIA_NATIVE_OFFSCREEN=true: NO NSWindow — rendering to offscreen {}x{}; \
         /scry (?eye=belief|presented), /budget, /state on http://127.0.0.1:{native_port}",
        dims.0, dims.1,
    );

    let size = PhysicalSize { width: dims.0.max(1), height: dims.1.max(1) };
    let mut deadline = Instant::now();
    let mut pending: Option<wgpu::SubmissionIndex> = None;
    // N0.j S13.3 OVERLAP THE REAL WORK — TRIED, MEASURED, DOES NOT HELP.
    // The ~7 ms world advance (skin·tick·splice + fresh BVH upload) is the
    // dominant OUTSIDE-work thief. The intent was to advance the NEXT frame's
    // world AFTER this frame's GPU submit so its CPU cost hides under the
    // in-flight GPU trace. But `trace` is SYNCHRONOUS on the render thread (it
    // submits+POLLS the GPU for the AOV that feeds the gather), so by the time
    // the deferred advance runs the GPU is already idle — nothing to hide under.
    // A/B measured it neutral-to-slightly-worse (47.5 vs 48.4 fps) AND it costs
    // one frame of world-state latency, so SERIAL is the default. The overlap
    // path stays behind `GAIA_NATIVE_WORLD_OVERLAP=1` for the record (it becomes
    // a real win only once trace stops blocking the render thread — the net
    // pipeline's next charter). `update_bvh` allocs FRESH buffers each tick and
    // the in-flight submission retains its own, so the overlap order is SAFE.
    let world_overlap = matches!(
        std::env::var("GAIA_NATIVE_WORLD_OVERLAP").as_deref(),
        Ok("1" | "true")
    );
    // Closure-free helper (borrow rules): advance one frame's world, timed.
    macro_rules! advance_timed {
        () => {{
            let t_world = Instant::now();
            let mut body_speed = 0.0f32;
            let mut walker_pose = None;
            if let Ok(mut body) = player.lock() {
                body.step(tick_dt, &ground);
                let pose = body.pose();
                body_speed = body.velocity.length();
                walker_pose = Some(WalkerPose { position: pose.position, yaw: pose.yaw });
                drop(body);
                renderer.set_view_pose(pose.position, pose.yaw, pose.pitch);
            }
            renderer.advance_world(body_speed, walker_pose, &[]);
            t_world.elapsed().as_secs_f64() * 1000.0
        }};
    }
    loop {
        let _ = renderer.device.poll(wgpu::PollType::Poll);
        let t_http = Instant::now();
        while let Ok(request) = scry_rx.try_recv() {
            match request {
                RenderRequest::Scry(request) => {
                    let frame = if request.params.belief {
                        #[cfg(target_os = "macos")]
                        {
                            renderer.capture_belief()
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            Err("belief eye is macOS-only".to_string())
                        }
                    } else if request.params.presented {
                        // N0.j S13.2 on-demand readback of the current offscreen frame.
                        renderer.capture_presented()
                    } else {
                        renderer.capture_pose(&request.params)
                    };
                    let _ = request.reply.send(frame);
                }
                RenderRequest::Retina { params, reply } => {
                    let _ = reply.send(renderer.capture_retina(&params));
                }
            }
        }
        let mut http_ms = t_http.elapsed().as_secs_f64() * 1000.0;
        // N0.j S13 THE OUTSIDE-9ms HUNT: time the non-net frame-loop segments.
        let t_iter = Instant::now();
        // SERIAL mode: advance BEFORE render (the old order).
        let mut world_ms = if world_overlap { 0.0 } else { advance_timed!() };
        let idx = renderer.render(size);
        // OVERLAP mode: advance the NEXT frame's world while THIS frame's GPU
        // flies (before waiting the previous frame below).
        if world_overlap {
            world_ms = advance_timed!();
        }
        if let Some(prev) = pending.take() {
            let _ = renderer.device.poll(wgpu::PollType::Wait {
                submission_index: Some(prev),
                timeout: None,
            });
        }
        pending = idx;
        let t_debug = Instant::now();
        if let Ok(mut d) = debug.write() {
            d.budget = renderer.debug_budget_json();
            d.state = renderer.debug_state_json();
        }
        http_ms += t_debug.elapsed().as_secs_f64() * 1000.0;
        // Record the outside-work AFTER render set `last_readback_ms` this frame.
        let readback_ms = renderer.last_readback_ms;
        let loop_total_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
        renderer
            .outside
            .record(world_ms, readback_ms, http_ms, loop_total_ms);
        let stages = renderer.last_world_stages;
        renderer.outside.record_world(stages);
        deadline += render_interval;
        let now = Instant::now();
        if deadline > now {
            thread::sleep(deadline - now);
        } else {
            deadline = now;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_render_loop(
    renderer: &mut Renderer,
    world_core: &mut WorldCore,
    authority_scene_params: &SceneParameters,
    world_rx: &mpsc::Receiver<WorldRequest>,
    window: &tauri::Window,
    render_player: &Arc<Mutex<Player>>,
    render_ground: &Arc<Ground>,
    tick_dt: f32,
    render_interval: Duration,
    scry_rx: &mpsc::Receiver<RenderRequest>,
    bend_rx: Option<&mpsc::Receiver<Bend>>,
    running: &Arc<AtomicBool>,
    hud_overlay: &tauri::webview::Webview<tauri::Wry>,
    hud_enabled: bool,
    hud_window: usize,
    debug: &DebugCell,
) {
    let mut deadline = Instant::now();
    // FPS COUNTER BURST — the REAL frame clock: the same std::time::Instant
    // style measure_trace_ms uses at startup, applied per delivered frame
    // (measured after present, i.e. across the Fifo vsync wait too — this is
    // the cadence the Architect's eyes actually see). Never a second timer:
    // the overlay DOM only renders numbers Rust pushes it, no JS rAF loop.
    let mut last_tick = Instant::now();
    let mut frame_times: std::collections::VecDeque<f64> =
        std::collections::VecDeque::with_capacity(hud_window.max(1));
    let mut hud_logged = 0u32;
    // Steady-state HUD sampling to stderr: default logs only the first 5 frames
    // (warm-up), but GAIA_NATIVE_HUD_LOG=<N> also logs every N-th delivered
    // frame — the honest way to read the LIVE frame clock at a settled vista
    // (LEVER 2 before/after) without a webview readback API.
    let hud_log_every: u32 = std::env::var("GAIA_NATIVE_HUD_LOG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut hud_frame = 0u32;
    // LEVER 2 — CPU/GPU overlap: the PREVIOUS frame's GPU submission, completed
    // only after THIS frame's CPU stages (body.step + advance_world) have run,
    // so frame N+1's skin/tick/splice/upload overlap frame N's trace. Proven
    // bit-identical to serial by `live_loop_hash_identity` (mirrors perf_audit's
    // ATOM B FNV hash-identity). Carried across iterations; drained on exit.
    let mut pending: Option<wgpu::SubmissionIndex> = None;
    while running.load(Ordering::Acquire) {
        // Service the map callbacks of the frame completed last iteration
        // (non-blocking) — keeps the /scry capture ring draining.
        let _ = renderer.device.poll(wgpu::PollType::Poll);
        // Incantations apply + journal on the render owner, then rebuild the
        // derived scene before HTTP receives success.
        while let Ok(request) = world_rx.try_recv() {
            match request {
                WorldRequest::Apply { batch, reply } => {
                    let result = world_core.apply(batch).and_then(|report| {
                        let rebuild = report.applied.iter().any(|op| match op {
                            Op::Set(_) => true,
                            Op::Other { op, .. } => {
                                matches!(op.as_str(), "spawn" | "despawn" | "clear")
                            }
                            _ => false,
                        });
                        if rebuild {
                            renderer.rebuild_world_core(world_core, authority_scene_params)?;
                        }
                        eprintln!(
                            "[world-core] entropy={} latest={} applied={} Steiner frames={}",
                            report.entropy,
                            report.latest,
                            report.applied.len(),
                            world_core.journal_frame_count().unwrap_or(0),
                        );
                        Ok(report)
                    });
                    let _ = reply.send(result);
                }
                WorldRequest::Snapshot { reply } => {
                    let _ = reply.send(world_core.snapshot_json());
                }
                WorldRequest::Events {
                    since,
                    limit,
                    reply,
                } => {
                    let _ = reply.send(world_core.events_json(since, limit));
                }
            }
        }
        // Service moving-eye requests off the frame loop's hot path.
        // /scry's wait_indefinitely also completes `pending` — each scry
        // momentarily collapses the overlap; harmless (verification organ).
        while let Ok(request) = scry_rx.try_recv() {
            match request {
                RenderRequest::Scry(request) => {
                    let frame = if request.params.belief {
                        #[cfg(target_os = "macos")]
                        {
                            renderer.capture_belief()
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            Err("belief eye is macOS-only".to_string())
                        }
                    } else if request.params.presented {
                        // N0.j S13.2 on-demand readback of the current offscreen frame.
                        renderer.capture_presented()
                    } else {
                        renderer.capture_pose(&request.params)
                    };
                    let _ = request.reply.send(frame);
                }
                RenderRequest::Retina { params, reply } => { let _ = reply.send(renderer.capture_retina(&params)); }
            }
        }
        // DAS BLUTBÄNDIGEN — drain the file-watch. Coalesce a burst of mtime
        // bumps into ONE apply per surface this frame (an editor may touch a
        // file several times); scene rebuild + shader recompile both run here on
        // the render thread that owns the device.
        if let Some(bend_rx) = bend_rx {
            let (mut scene_dirty, mut shader_dirty) = (false, false);
            while let Ok(bend) = bend_rx.try_recv() {
                match bend {
                    Bend::Scene => scene_dirty = true,
                    Bend::Shader => shader_dirty = true,
                }
            }
            if scene_dirty {
                renderer.bend_scene();
            }
            if shader_dirty {
                renderer.bend_shader();
            }
        }
        // Step the body one fixed tick and aim the window camera at its eye.
        let mut body_speed = 0.0f32;
        let mut walker_pose = None;
        // PLAYGROUND — the pushed view ray this tick, taken (edge-fired) from
        // the shared player: F key, a pointer-locked click, or the /push organ
        // all set `push_pending`; we consume it here and cast the ray below.
        let mut push_ray: Option<(Vec3, f32, f32)> = None;
        if let Ok(mut body) = render_player.lock() {
            body.step(tick_dt, render_ground);
            let pose = body.pose();
            body_speed = body.velocity.length();
            if body.push_pending {
                body.push_pending = false;
                push_ray = Some((pose.position, pose.yaw, pose.pitch));
            }
            walker_pose = Some(WalkerPose {
                position: pose.position,
                yaw: pose.yaw,
            });
            drop(body);
            renderer.set_view_pose(pose.position, pose.yaw, pose.pitch);
        }
        let push_ops = match push_ray {
            Some((eye, yaw, pitch)) => renderer.build_push_ops(eye, yaw, pitch),
            None => Vec::new(),
        };
        renderer.advance_world(body_speed, walker_pose, &push_ops);
        // Submit THIS frame's GPU work WITHOUT waiting. Its trace now runs on
        // the GPU while the NEXT iteration's CPU stages execute above.
        if let Ok(size) = window.inner_size() {
            let idx = renderer.render(size);
            // Complete the PREVIOUS frame — its GPU work has been overlapping
            // THIS frame's CPU stages since last iteration's submit. Explicit
            // per-submission Wait (not wait_indefinitely, which would also block
            // on the frame just submitted and collapse the overlap).
            if let Some(prev) = pending.take() {
                let _ = renderer.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(prev),
                    timeout: None,
                });
            }
            pending = idx;
        }
        // S12.5 AI DEBUG DOOR: refresh the /budget + /state JSON (cheap strings).
        if let Ok(mut d) = debug.write() {
            d.budget = renderer.debug_budget_json();
            d.state = renderer.debug_state_json();
        }
        deadline += render_interval;
        let now = Instant::now();
        if deadline > now {
            thread::sleep(deadline - now);
        } else {
            deadline = now;
        }

        if hud_enabled {
            let tick_now = Instant::now();
            let frame_ms = tick_now.duration_since(last_tick).as_secs_f64() * 1e3;
            last_tick = tick_now;
            frame_times.push_back(frame_ms);
            if frame_times.len() > hud_window.max(1) {
                frame_times.pop_front();
            }
            // HUD shows DELIVERED cadence (incl. vsync/pacing) — render-cost
            // instruments are GAIA_NATIVE_HUD_LOG + live_loop_audit.
            let mut sorted: Vec<f64> = frame_times.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_ms = sorted[sorted.len() / 2];
            let fps = if median_ms > 0.0 {
                1000.0 / median_ms
            } else {
                0.0
            };
            let payload = format!("window.__gaiaHud && window.__gaiaHud({fps:.1},{median_ms:.2})");
            let _ = hud_overlay.eval(payload.clone());
            hud_frame += 1;
            if hud_logged < 5 {
                eprintln!("[hud] {payload}");
                hud_logged += 1;
            } else if hud_log_every > 0 && hud_frame % hud_log_every == 0 {
                eprintln!("[hud] frame {hud_frame} {payload}");
            }
        }
    }
    // Drain the last in-flight frame before the render thread returns.
    if let Some(prev) = pending.take() {
        let _ = renderer.device.poll(wgpu::PollType::Wait {
            submission_index: Some(prev),
            timeout: None,
        });
    }
}
