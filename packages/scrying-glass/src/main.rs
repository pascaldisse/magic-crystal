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

use crystal::{Core, GaiaPackage, load_world_dir};
use glam::Vec3;
use scrying_glass::ScryingGlassPackage;
use scrying_glass::bvh::{Bvh, BvhParams, DynamicSplice, RefitParams};
use scrying_glass::integrator::{Integrator, IntegratorParams, IntegratorUniform};
use scrying_glass::scene::{
    Camera, RenderScene, SceneParameters, SunDefaults, SunLight, WalkerPose,
};
use tauri::{Manager, PhysicalPosition, PhysicalSize, WebviewUrl};

const DEFAULT_NATIVE_PORT: u16 = 8430;
const BYTES_PER_PIXEL: u32 = 4;
const CAPTURE_SLOT_COUNT: usize = 3;

#[derive(Clone)]
struct ScryingGlassConfig {
    window_width: f64,
    window_height: f64,
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
        let world_path = std::env::var_os("GAIA_WORLD")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko"));
        let config = Self {
            window_width: number("GAIA_NATIVE_WIDTH", 960.0)?,
            window_height: number("GAIA_NATIVE_HEIGHT", 640.0)?,
            panel_width: number("SPIKE_PANEL_WIDTH", 300.0)?,
            panel_height: number("SPIKE_PANEL_HEIGHT", 154.0)?,
            panel_margin: number("SPIKE_PANEL_MARGIN", 24.0)?,
            fps: number("GAIA_NATIVE_FPS", 60.0)?,
            native_port,
            title: std::env::var("GAIA_NATIVE_TITLE")
                .unwrap_or_else(|_| "GAIA — Scrying Glass".into()),
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
                cluster_error_threshold: number("GAIA_NATIVE_CLUSTER_ERROR", 1.0)? as f32,
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
                degrade_ratio: number("GAIA_NATIVE_BVH_REFIT_DEGRADE", 10.0964)? as f32,
                max_refits: integer("GAIA_NATIVE_BVH_REFIT_MAX", 0)?,
            },
            capture_frames: integer("GAIA_NATIVE_CAPTURE_FRAMES", 48)?,
        };
        if config.window_width <= 0.0
            || config.window_height <= 0.0
            || config.panel_width <= 0.0
            || config.panel_height <= 0.0
            || config.panel_margin < 0.0
            || config.fps <= 0.0
            || config.native_port == 0
        {
            return Err(
                "window dimensions, FPS, and GAIA_NATIVE_PORT must be positive (margin may be zero)"
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
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n{extra_headers}\r\n",
        body.len()
    )?;
    stream.write_all(body)
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

/// Shared state the HTTP organs read: the live surface frame, the moving-eye
/// render channel, and — for the Embodiment — the walking body + its floor.
struct HttpContext {
    latest: LatestFrame,
    scry: mpsc::Sender<ScryRequest>,
    player: Arc<Mutex<Player>>,
    ground: Arc<Ground>,
    tick_dt: f32,
}

/// Read a full HTTP request (headers + any body) honouring Content-Length.
fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
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
        if buffer.len() > 1 << 20 {
            return None; // 1 MiB header guard
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
    let Some((headers, body)) = read_request(&mut stream) else {
        return;
    };
    let first_line = headers.lines().next().unwrap_or_default().to_owned();
    let mut tokens = first_line.split_whitespace();
    let method = tokens.next().unwrap_or_default();
    let target = tokens.next().unwrap_or_default();
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

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

    // No query = exactly the prior behaviour: serve the latest live surface frame.
    if query.is_empty() {
        match latest.read().ok().and_then(|frame| frame.clone()) {
            Some(frame) => respond_frame(&mut stream, &frame),
            None => {
                let _ = write_response(
                    &mut stream,
                    "503 Service Unavailable",
                    "text/plain; charset=utf-8",
                    b"framebuffer not ready\n",
                    "Retry-After: 1\r\n",
                );
            }
        }
        return;
    }

    // Moving eye: parse the pose overrides and ask the render thread for a fresh frame.
    let params = match parse_scry_query(query) {
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
    let (reply_tx, reply_rx) = mpsc::channel();
    if scry
        .send(ScryRequest {
            params,
            reply: reply_tx,
        })
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
        "[scry] GET http://127.0.0.1:{port}/scry (alias: /screenshot; optional pos/yaw/pitch/fov/w/h)"
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

struct Renderer {
    // Safety: created from the native Tauri Window's raw handles; the app owns that Window
    // until shutdown, and the render worker stops before process exit.
    surface: wgpu::Surface<'static>,
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
    /// The persistent two-level splice (LEVER 1): refits the dynamic partition
    /// per tick when the set is unchanged, rebuilds only on set change / bound
    /// degradation. Its `merged` tree is what gets uploaded.
    splice: DynamicSplice,
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
    /// Persistent window accumulation: progressive while the eye is still,
    /// reset the instant it moves or the surface resizes.
    surface_accum: wgpu::Buffer,
    surface_compute_bg: wgpu::BindGroup,
    surface_blit_bg: wgpu::BindGroup,
    samples_before: u32,
    /// (eye, yaw, pitch, width, height) the current accumulation belongs to.
    last_view: Option<([f32; 3], f32, f32, u32, u32)>,
    offscreen: OffscreenTarget,
    pixel_order: PixelOrder,
    capture_sender: mpsc::Sender<CaptureReady>,
}

impl Renderer {
    fn new(
        window: &tauri::Window,
        capture_sender: mpsc::Sender<CaptureReady>,
        scene: RenderScene,
        int_params: IntegratorParams,
        bvh_params: &BvhParams,
        refit_params: RefitParams,
        capture_frames: u32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let target = unsafe {
            wgpu::SurfaceTargetUnsafe::from_display_and_window(window, window)
                .map_err(|error| format!("raw-window-handle target: {error}"))?
        };
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|error| format!("wgpu surface: {error}"))?
        };
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|error| format!("wgpu adapter: {error}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|error| format!("wgpu device: {error}"))?;
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // The acceleration: a STATIC BVH over the Great Chain's EXACT non-behavior
        // leaf triangles (built once, cached), with the living layer's dynamic
        // partition spliced on top. Load budget printed, never gated (RENDER:
        // cost ∝ pixels, not FLOPs).
        let build_start = Instant::now();
        let static_bvh = Bvh::build(&scene.leaf_triangles(), bvh_params);
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

        let surface_accum = integrator.make_accum(&device, config.width, config.height);
        let surface_compute_bg = integrator.compute_bind_group(&device, &surface_accum);
        let surface_blit_bg = integrator.blit_bind_group(&device, &surface_accum);
        let offscreen = OffscreenTarget::new(&device, format, config.width, config.height);
        eprintln!(
            "[wgpu] traced surface + offscreen framebuffer: {format:?} {}x{}",
            config.width, config.height
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            integrator,
            scene,
            static_bvh,
            splice,
            last_models,
            camera,
            sun,
            sky_top,
            sky_horizon,
            int_params,
            capture_frames,
            surface_accum,
            surface_compute_bg,
            surface_blit_bg,
            samples_before: 0,
            last_view: None,
            offscreen,
            pixel_order,
            capture_sender,
        })
    }

    /// Rebuild the window accumulation buffer (zeroed) for the current surface
    /// size and drop the accumulated samples — the reset gesture on move/resize.
    fn reset_surface_accum(&mut self) {
        let accum = self
            .integrator
            .make_accum(&self.device, self.config.width, self.config.height);
        self.surface_compute_bg = self.integrator.compute_bind_group(&self.device, &accum);
        self.surface_blit_bg = self.integrator.blit_bind_group(&self.device, &accum);
        self.surface_accum = accum;
        self.samples_before = 0;
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
    fn advance_world(&mut self, body_speed: f32, walker: Option<WalkerPose>) {
        let has_bodies = !self.scene.bodies.is_empty();
        if self.scene.dynamics.entities().is_empty() && !has_bodies {
            return; // a still realm never pays the living-layer cost
        }
        // RITE V·V1 — drive the embodied bodies from the walker's velocity: the
        // commanded speed feeds each body's SAMA state machine, its pose re-skins
        // the body per tick. A walking body changes the dynamic partition every
        // tick even when the living models are still, so it forces a re-splice.
        // RITE V FINAL WELD — `walker` (its world pose) drives walker-ATTACHED
        // bodies (`follows: "walker"`): they TRACK the walker, gait derived from
        // displacement, instead of gaiting in place off the broadcast.
        let bodies_animating = self.scene.command_bodies_walked(body_speed, walker);
        self.scene.tick();
        let models = self.scene.dynamics.model_matrices();
        if models == self.last_models && !bodies_animating {
            return; // nothing moved — keep accumulating
        }
        self.splice
            .update(&self.static_bvh, &self.scene.dynamic_leaf_triangles());
        self.integrator
            .update_bvh(&self.device, &self.splice.merged);
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
            self.surface.configure(&self.device, &self.config);
            self.offscreen = OffscreenTarget::new(
                &self.device,
                self.config.format,
                self.config.width,
                self.config.height,
            );
            self.reset_surface_accum();
            self.last_view = None;
        }
    }

    /// Point the windowed camera at the embodied player's eye. Movement resets
    /// the accumulation on the next frame (detected by `last_view`).
    fn set_view_pose(&mut self, eye: Vec3, yaw: f32, pitch: f32) {
        self.camera.eye = eye;
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;
    }

    fn view_key(&self) -> ([f32; 3], f32, f32, u32, u32) {
        (
            self.camera.eye.to_array(),
            self.camera.yaw,
            self.camera.pitch,
            self.config.width,
            self.config.height,
        )
    }

    fn render(&mut self, size: PhysicalSize<u32>) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.resize(size);

        // Reset accumulation the instant the eye moves (progressive while still).
        let key = self.view_key();
        if self.last_view != Some(key) {
            self.reset_surface_accum();
            self.last_view = Some(key);
        }

        let (width, height) = (self.config.width, self.config.height);
        let uniform = IntegratorUniform::build(
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

        let surface_frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                None
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => None,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("traced frame + capture"),
            });
        // One accumulation frame, then present the running mean to both targets.
        self.integrator.dispatch(
            &self.queue,
            &mut encoder,
            &uniform,
            &self.surface_compute_bg,
            width,
            height,
        );
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
        self.queue.submit(Some(encoder.finish()));
        self.samples_before += self.int_params.spp;
        if let Some(frame) = surface_frame {
            self.queue.present(frame);
        }
    }

    /// The moving eye: integrate `capture_frames` accumulation frames from an
    /// arbitrary pose to a per-request offscreen target and read it back. Runs on
    /// the render thread; the surface loop's own accumulation is untouched.
    fn capture_pose(&mut self, params: &ScryParams) -> Result<CapturedFrame, String> {
        let width = params.width.unwrap_or(self.config.width).max(1);
        let height = params.height.unwrap_or(self.config.height).max(1);
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

        // Present the converged mean to a fresh sRGB target, then read it back.
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
}

/// Optional moving-eye overrides parsed from `GET /scry?...`.
/// All absent = exactly the default spawn-pose capture.
#[derive(Clone, Debug, Default)]
struct ScryParams {
    pos: Option<[f32; 3]>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    fov: Option<f32>,
    width: Option<u32>,
    height: Option<u32>,
}

struct ScryRequest {
    params: ScryParams,
    reply: mpsc::Sender<Result<CapturedFrame, String>>,
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
    let loaded = load_world_dir(&config.world_path, &mut core.world)
        .unwrap_or_else(|error| panic!("load GAIA_WORLD {}: {error}", config.world_path.display()));
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
        loaded.path.display(),
        loaded.scenes,
        loaded.entity_count,
        render_scene.chains.len(),
        cluster_count,
    );

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![panel_pressed])
        .setup(move |app| {
            let window = tauri::window::WindowBuilder::new(app, "wgpu-surface")
                .title(config.title.clone())
                .inner_size(config.window_width, config.window_height)
                .build()?;
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
                if let tauri::WindowEvent::Resized(size) = event {
                    let (position, panel_size) = resize_config.panel_layout(*size);
                    let _ = resize_overlay.set_position(position);
                    let _ = resize_overlay.set_size(panel_size);
                }
            });
            install_passthrough_monitor(window.clone(), config.clone())
                .map_err(std::io::Error::other)?;

            let latest = Arc::new(RwLock::new(None));
            let capture_sender = spawn_capture_worker(latest.clone());

            // The Embodiment: the world's own leaf triangles become the floor
            // (exact geometry, view-independent — never a camera's coarse cut),
            // and the world spawn pose becomes a walking body.
            let ground = Arc::new(Ground::from_positions(&render_scene.leaf_positions()));
            let spawn_eye = render_scene.camera.eye;
            let spawn_yaw = render_scene.camera.yaw;
            let player_params = PlayerParams::from_env().map_err(std::io::Error::other)?;
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
                &window,
                capture_sender,
                render_scene,
                config.integrator,
                &config.bvh,
                config.refit,
                config.capture_frames,
            )
            .map_err(std::io::Error::other)?;
            let (scry_tx, scry_rx) = mpsc::channel::<ScryRequest>();
            start_screenshot_server(
                native_port,
                HttpContext {
                    latest,
                    scry: scry_tx,
                    player: player.clone(),
                    ground: ground.clone(),
                    tick_dt,
                },
            )
            .map_err(std::io::Error::other)?;
            let running = Arc::new(AtomicBool::new(true));
            app.manage(RuntimeState {
                running: running.clone(),
            });
            let render_player = player.clone();
            let render_ground = ground.clone();
            thread::Builder::new()
                .name("gaia-render".into())
                .spawn(move || {
                    let mut renderer = renderer;
                    let mut deadline = Instant::now();
                    while running.load(Ordering::Acquire) {
                        // Service moving-eye requests off the frame loop's hot path.
                        while let Ok(request) = scry_rx.try_recv() {
                            let frame = renderer.capture_pose(&request.params);
                            let _ = request.reply.send(frame);
                        }
                        // Step the body one fixed tick and aim the window camera
                        // at its eye.
                        let mut body_speed = 0.0f32;
                        let mut walker_pose = None;
                        if let Ok(mut body) = render_player.lock() {
                            body.step(tick_dt, &render_ground);
                            let pose = body.pose();
                            // The walker's horizontal velocity drives sama.
                            body_speed = body.velocity.length();
                            // RITE V FINAL WELD — the walker's world pose drives
                            // walker-ATTACHED bodies (they track the player).
                            walker_pose = Some(WalkerPose {
                                position: pose.position,
                                yaw: pose.yaw,
                            });
                            drop(body);
                            renderer.set_view_pose(pose.position, pose.yaw, pose.pitch);
                        }
                        // Tick the world clock, drive the embodied bodies from
                        // the walker velocity + pose, and re-splice the living layer.
                        renderer.advance_world(body_speed, walker_pose);
                        if let Ok(size) = window.inner_size() {
                            renderer.render(size);
                        }
                        deadline += render_interval;
                        let now = Instant::now();
                        if deadline > now {
                            thread::sleep(deadline - now);
                        } else {
                            deadline = now;
                        }
                    }
                })
                .map_err(std::io::Error::other)?;
            eprintln!(
                "[scrying-glass] child webview overlay created; render, capture, PNG, and HTTP run off the main thread"
            );
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Tauri app build failed")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                app.state::<RuntimeState>()
                    .running
                    .store(false, Ordering::Release);
            }
        });
}
