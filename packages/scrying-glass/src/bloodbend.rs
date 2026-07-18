//! DAS BLUTBÄNDIGEN — B0, the DATA DOOR (docs/proposals/BLOODBEND.md).
//!
//! Live alteration of the RUNNING glass: file-watch on the loaded scene JSON +
//! the WGSL shader source, validate every change through die Zauberpolizei
//! BEFORE it touches living tissue (reject-before-apply, never partial), snapshot
//! the previous state into a bend-journal FIRST (Traumdeuter-Vorritt), then apply
//! the SMALLEST safe tier. Standing laws honoured here:
//!
//! - FULL-MOON RULE (law 2): a scene bend is validated by re-running the whole
//!   loader + render-scene materialization into a THROWAWAY world; only a clean
//!   result is swapped in. A bad edit leaves the world byte-identical. A shader
//!   bend is validated by a wgpu error scope around module + pipeline creation;
//!   a bad shader keeps the old pipeline rendering.
//! - TRAUMDEUTER-VORRITT (law 3-adjacent): the PREVIOUS good scene bytes are
//!   copied into `<journal>/scene-<stamp>/` before the new ones are applied;
//!   undo = copy that snapshot back over the scene files, the watch re-applies.
//! - BLAST-RADIUS LADDER (law 4): the diff (added/removed/changed entity ids)
//!   is reported; the scene tier rebuilds the render scene + BVH live, the
//!   window/device/surface/pipelines all persist untouched.
//!
//! IRON: every value is a param with a default. Master switch
//! `GAIA_NATIVE_BLOODBEND` (default ON — validation makes it safe).

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// One detected change the watcher hands the render thread. The render thread
/// owns the device + scene, so ALL validation and application happen there;
/// the watcher only reports WHICH surface moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bend {
    /// A watched scene JSON file (world.json or scenes/*.json) changed.
    Scene,
    /// The watched WGSL shader source changed.
    Shader,
}

/// Params for the data door — all env, all defaulted (IRON law).
#[derive(Clone, Debug)]
pub struct BloodbendParams {
    /// Master switch `GAIA_NATIVE_BLOODBEND` (default true).
    pub enabled: bool,
    /// mtime poll interval `GAIA_NATIVE_BLOODBEND_POLL` ms (default 500).
    pub poll: Duration,
    /// Journal root `GAIA_NATIVE_BLOODBEND_JOURNAL` (default `debug/bloodbend-journal`).
    pub journal_dir: PathBuf,
    /// Watched WGSL source `GAIA_NATIVE_BLOODBEND_SHADER` (default the compiled-in
    /// integrator.wgsl source path).
    pub shader_path: PathBuf,
    /// The scene JSON files watched: world.json (if present) + every scenes/*.json.
    pub scene_paths: Vec<PathBuf>,
}

impl BloodbendParams {
    pub fn from_env(world_path: &Path) -> Result<Self, String> {
        let enabled = match std::env::var("GAIA_NATIVE_BLOODBEND") {
            Ok(v) => v
                .parse::<bool>()
                .map_err(|_| format!("GAIA_NATIVE_BLOODBEND must be true or false, got {v:?}"))?,
            Err(_) => true,
        };
        let poll_ms = match std::env::var("GAIA_NATIVE_BLOODBEND_POLL") {
            Ok(v) => v
                .parse::<u64>()
                .map_err(|_| format!("GAIA_NATIVE_BLOODBEND_POLL must be ms, got {v:?}"))?,
            Err(_) => 500,
        };
        let journal_dir = std::env::var_os("GAIA_NATIVE_BLOODBEND_JOURNAL")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("debug/bloodbend-journal"));
        let shader_path = std::env::var_os("GAIA_NATIVE_BLOODBEND_SHADER")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src/integrator.wgsl"))
            });
        Ok(Self {
            enabled,
            poll: Duration::from_millis(poll_ms.max(1)),
            journal_dir,
            shader_path,
            scene_paths: watched_scene_files(world_path),
        })
    }
}

/// The live bend state the render thread carries: the params, the world path +
/// scene params needed to re-materialize a bent scene, and the LAST GOOD scene
/// bytes (the snapshot the journal preserves before the next apply).
#[derive(Clone, Debug)]
pub struct Bloodbend {
    pub params: BloodbendParams,
    pub world_path: PathBuf,
    pub scene_params: crate::scene::SceneParameters,
    pub last_good: BTreeMap<PathBuf, String>,
}

impl Bloodbend {
    /// Seed from the boot state: the currently-loaded scene bytes ARE the first
    /// "last good" — the first bend journals these before applying its change.
    pub fn seed(
        params: BloodbendParams,
        world_path: PathBuf,
        scene_params: crate::scene::SceneParameters,
    ) -> Self {
        let last_good = read_scene_bytes(&params.scene_paths);
        Self {
            params,
            world_path,
            scene_params,
            last_good,
        }
    }
}

/// The scene JSON files a world exposes to the watcher: world.json (if present)
/// plus every scenes/*.json. Order is stable (BTree/sorted) so the state hash
/// and journal are deterministic.
pub fn watched_scene_files(world_path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let meta = world_path.join("world.json");
    if meta.is_file() {
        files.push(meta);
    }
    let scenes = world_path.join("scenes");
    if let Ok(read) = fs::read_dir(&scenes) {
        let mut scene_files: Vec<PathBuf> = read
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
            .collect();
        scene_files.sort();
        files.extend(scene_files);
    }
    files
}

/// Read every watched scene file into an id→bytes map keyed by path. Missing
/// files are simply absent (a scene deleted on disk is a change the loader will
/// reject on its own). This is the "last good" snapshot the journal preserves.
pub fn read_scene_bytes(scene_paths: &[PathBuf]) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    for path in scene_paths {
        if let Ok(text) = fs::read_to_string(path) {
            out.insert(path.clone(), text);
        }
    }
    out
}

/// A millisecond stamp since the epoch — the journal folder name. No chrono dep.
fn stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// TRAUMDEUTER-VORRITT: copy the PREVIOUS good scene bytes into
/// `<journal>/scene-<stamp>/` before the new state is applied. Returns the
/// snapshot dir. Undo = copy these files back over the scene paths; the watch
/// re-applies them. Files are stored by their base name (scene ids are unique
/// across scenes so collisions do not occur for the naruko-shaped worlds; a
/// numeric suffix disambiguates if two watched files share a name).
pub fn journal_previous(
    journal_dir: &Path,
    previous: &BTreeMap<PathBuf, String>,
) -> Result<PathBuf, String> {
    let dir = journal_dir.join(format!("scene-{}", stamp()));
    fs::create_dir_all(&dir).map_err(|e| format!("create journal dir {}: {e}", dir.display()))?;
    let mut used: BTreeMap<String, u32> = BTreeMap::new();
    for (path, text) in previous {
        let base = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unnamed.json".to_string());
        let name = match used.get_mut(&base) {
            Some(n) => {
                *n += 1;
                format!("{n}-{base}")
            }
            None => {
                used.insert(base.clone(), 0);
                base
            }
        };
        let dest = dir.join(&name);
        fs::write(&dest, text).map_err(|e| format!("write journal {}: {e}", dest.display()))?;
    }
    Ok(dir)
}

/// Diff two scene-file maps at the ENTITY level (law 4 blast-radius report).
/// Each file is a JSON object of id→entity-doc; we union all ids across files
/// and classify each as added / removed / changed (serialized value differs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
}

impl SceneDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "+{} added {:?} · ~{} changed {:?} · -{} removed {:?}",
            self.added.len(),
            self.added,
            self.changed.len(),
            self.changed,
            self.removed.len(),
            self.removed,
        )
    }
}

/// Flatten a scene-file map into one id→doc map (last file wins on a duplicate
/// id, matching the loader's own duplicate handling is not needed here — the
/// loader rejects duplicates, so this only runs on already-valid inputs).
fn entities_of(files: &BTreeMap<PathBuf, String>) -> BTreeMap<String, serde_json::Value> {
    let mut out = BTreeMap::new();
    for text in files.values() {
        if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(text)
        {
            for (id, doc) in map {
                out.insert(id, doc);
            }
        }
    }
    out
}

pub fn diff_scenes(
    previous: &BTreeMap<PathBuf, String>,
    next: &BTreeMap<PathBuf, String>,
) -> SceneDiff {
    let old = entities_of(previous);
    let new = entities_of(next);
    let mut diff = SceneDiff::default();
    for (id, doc) in &new {
        match old.get(id) {
            None => diff.added.push(id.clone()),
            Some(prev) if prev != doc => diff.changed.push(id.clone()),
            Some(_) => {}
        }
    }
    for id in old.keys() {
        if !new.contains_key(id) {
            diff.removed.push(id.clone());
        }
    }
    diff
}

/// A deterministic FNV-1a hash of the render scene's STATIC leaf triangles
/// (position + albedo + emission + metallic + roughness). "World state hash" for
/// the ordeals: unchanged across a REJECTED bend (world byte-identical), changed
/// across an APPLIED edit (the crate's new colour). View-independent, error 0.
pub fn scene_state_hash(scene: &crate::scene::RenderScene) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |bits: u32| {
        for b in bits.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for tri in scene.leaf_triangles() {
        for p in tri.positions {
            for c in p {
                eat(c.to_bits());
            }
        }
        for c in tri.albedo {
            eat(c.to_bits());
        }
        for c in tri.emission {
            eat(c.to_bits());
        }
        eat(tri.metallic.to_bits());
        eat(tri.roughness.to_bits());
    }
    h
}

/// Emit a Zauberpolizei POLICE REPORT — a bend was rejected, the living world
/// is untouched. One line, telegraphic: which surface, which law, where.
pub fn police_report(surface: &str, detail: &str) {
    eprintln!("[bloodbend] ⛔ ZAUBERPOLIZEI REJECT · {surface} · {detail}");
}

/// Emit a bend-applied notice (a bend passed inspection and touched tissue).
pub fn bend_applied(surface: &str, detail: &str) {
    eprintln!("[bloodbend] ✅ BEND APPLIED · {surface} · {detail}");
}

/// Spawn the file-watch helper thread (law 1: mtime polling, NO new crate).
/// Sends a [`Bend`] the instant a watched file's mtime advances (or it appears /
/// disappears). Returns the receiver the render loop drains. The thread lives
/// for the process; it exits when the receiver is dropped (send fails).
pub fn spawn_watcher(params: &BloodbendParams) -> mpsc::Receiver<Bend> {
    let (tx, rx) = mpsc::channel();
    let poll = params.poll;
    let shader_path = params.shader_path.clone();
    let scene_paths = params.scene_paths.clone();
    thread::Builder::new()
        .name("bloodbend-watch".into())
        .spawn(move || {
            let mtime = |p: &Path| fs::metadata(p).and_then(|m| m.modified()).ok();
            let mut scene_mtimes: BTreeMap<PathBuf, Option<SystemTime>> =
                scene_paths.iter().map(|p| (p.clone(), mtime(p))).collect();
            let mut shader_mtime = mtime(&shader_path);
            eprintln!(
                "[bloodbend] 🩸 watching {} scene file(s) + {} @ {}ms poll",
                scene_paths.len(),
                shader_path.display(),
                poll.as_millis(),
            );
            loop {
                thread::sleep(poll);
                let mut scene_dirty = false;
                for (path, last) in scene_mtimes.iter_mut() {
                    let now = mtime(path);
                    if now != *last {
                        *last = now;
                        scene_dirty = true;
                    }
                }
                if scene_dirty && tx.send(Bend::Scene).is_err() {
                    return;
                }
                let now_shader = mtime(&shader_path);
                if now_shader != shader_mtime {
                    shader_mtime = now_shader;
                    if tx.send(Bend::Shader).is_err() {
                        return;
                    }
                }
            }
        })
        .expect("spawn bloodbend watcher");
    rx
}
