//! oracle-cli — read a GAIA world through the RAIN senses. PULL-ONLY.
//!
//! Usage:
//!   oracle-cli [--world DIR] \[look\] [--fov F] [--grid N] [--nearest N]
//!             [--pos x,y,z] [--yaw R] [--pitch R] [--layer ids|depth|both|none]
//!   oracle-cli [--world DIR] proprio `ENTITY-ID`
//!
//! Defaults: world = $GAIA_WORLD or client-rs/worlds/naruko; pose = world spawn.

use oracle::{look, proprio, EyePose, Glance, Layers, LookParams, World};
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("oracle-cli: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut world_arg: Option<String> = None;
    let mut mode = Mode::Look;
    let mut params = LookParams::default();
    let mut pos: Option<[f32; 3]> = None;
    let mut yaw: Option<f32> = None;
    let mut pitch: Option<f32> = None;
    let mut layer = Layer::Ids;
    let mut proprio_id: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        match a.as_str() {
            "look" => mode = Mode::Look,
            "proprio" => mode = Mode::Proprio,
            "--world" => world_arg = Some(value(&args, &mut i, &a)?),
            "--fov" => params.fov_deg = parse(&value(&args, &mut i, &a)?)?,
            "--grid" => params.grid = parse(&value(&args, &mut i, &a)?)?,
            "--nearest" => params.nearest_n = parse(&value(&args, &mut i, &a)?)?,
            "--near" => params.near = parse(&value(&args, &mut i, &a)?)?,
            "--far" => params.far = parse(&value(&args, &mut i, &a)?)?,
            "--yaw" => yaw = Some(parse(&value(&args, &mut i, &a)?)?),
            "--pitch" => pitch = Some(parse(&value(&args, &mut i, &a)?)?),
            "--pos" => pos = Some(parse_vec3(&value(&args, &mut i, &a)?)?),
            "--max-grid" => params.max_grid = parse(&value(&args, &mut i, &a)?)?,
            "--support-ratio" => params.support_ratio = parse(&value(&args, &mut i, &a)?)?,
            "--layer" => layer = Layer::parse(&value(&args, &mut i, &a)?)?,
            other if !other.starts_with("--") && mode == Mode::Proprio && proprio_id.is_none() => {
                proprio_id = Some(other.to_string());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let default_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let dir = world_arg
        .map(PathBuf::from)
        .unwrap_or_else(|| World::resolve_dir(&default_dir));
    let world = World::load(&dir)?;

    println!("# world: {}", world.world_dir.display());
    println!("# scenes: {}", world.scene_files.join(", "));
    println!("# entities: {}", world.entities.len());
    for w in &world.schema_warnings {
        println!("# schema-warning: {w}");
    }

    match mode {
        Mode::Proprio => {
            let id = proprio_id.ok_or("proprio needs an entity id")?;
            let p = proprio(&world, &id).ok_or_else(|| format!("no entity '{id}'"))?;
            print!("{}", render_proprio(&p));
        }
        Mode::Look => {
            let eye = resolve_eye(&world, pos, yaw, pitch)?;
            // The CLI --layer selects which grid layers are COMPUTED (not just
            // shown): unrequested layers cost nothing (RAIN context diet).
            params.layers = layer.as_layers();
            let glance = look(&world, eye, params).map_err(|e| e.to_string())?;
            print!("{}", render_glance(&glance, layer));
        }
    }
    Ok(())
}

#[derive(PartialEq)]
enum Mode {
    Look,
    Proprio,
}

#[derive(Clone, Copy)]
enum Layer {
    Ids,
    Depth,
    Both,
    None,
}
impl Layer {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "ids" => Ok(Layer::Ids),
            "depth" => Ok(Layer::Depth),
            "both" => Ok(Layer::Both),
            "none" => Ok(Layer::None),
            other => Err(format!("unknown layer: {other}")),
        }
    }
    fn as_layers(self) -> Layers {
        match self {
            Layer::Ids => Layers::IDS,
            Layer::Depth => Layers::DEPTH,
            Layer::Both => Layers::BOTH,
            Layer::None => Layers::NONE,
        }
    }
}

fn resolve_eye(
    world: &World,
    pos: Option<[f32; 3]>,
    yaw: Option<f32>,
    pitch: Option<f32>,
) -> Result<EyePose, String> {
    let spawn = world.spawn_pose();
    let base = spawn.unwrap_or(EyePose {
        position: [0.0, 2.0, 0.0],
        yaw: 0.0,
        pitch: 0.0,
    });
    Ok(EyePose {
        position: pos.unwrap_or(base.position),
        yaw: yaw.unwrap_or(base.yaw),
        pitch: pitch.unwrap_or(base.pitch),
    })
}

fn render_glance(g: &Glance, layer: Layer) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\nGLANCE  eye=({:.1},{:.1},{:.1}) yaw={:.3} pitch={:.3} fov={:.0}° grid={}×{}\n",
        g.eye.position[0],
        g.eye.position[1],
        g.eye.position[2],
        g.eye.yaw,
        g.eye.pitch,
        g.fov_deg,
        g.grid,
        g.grid
    ));
    out.push_str(&format!("captions: {} entities in view", g.entity_count));
    if let Some(env) = &g.environment {
        out.push_str(&format!("  |  env: {env}"));
    }
    out.push('\n');
    out.push_str(&format!("nearest {}:\n", g.nearest.len()));
    for (i, n) in g.nearest.iter().enumerate() {
        let side = if n.bearing_deg.abs() < 0.5 {
            "ahead".to_string()
        } else if n.bearing_deg < 0.0 {
            format!("{:.0}° L", -n.bearing_deg)
        } else {
            format!("{:.0}° R", n.bearing_deg)
        };
        out.push_str(&format!(
            "  {}. {:<16} {:>7}  el {:+.0}°  {:.0}m  size {:.1}m{}\n",
            i + 1,
            n.id,
            side,
            n.elevation_deg,
            n.range,
            n.size,
            match &n.emissive {
                Some(color) => format!("  ✦ {color}"),
                None => String::new(),
            }
        ));
    }

    // Legend: one glyph per distinct entity in the grid.
    let mut legend: Vec<String> = Vec::new();
    let glyph_of = |id: &str, legend: &mut Vec<String>| -> char {
        if let Some(k) = legend.iter().position(|x| x == id) {
            glyph(k)
        } else {
            legend.push(id.to_string());
            glyph(legend.len() - 1)
        }
    };

    if matches!(layer, Layer::Ids | Layer::Both) {
        out.push_str(&format!(
            "\nglance grid — ids ({}×{}, row 0 = top):\n",
            g.grid, g.grid
        ));
        let horizon = g.horizon_row();
        for row in 0..g.grid {
            out.push_str("  ");
            for col in 0..g.grid {
                match g.cell_id(row * g.grid + col) {
                    Some(id) => out.push(glyph_of(id, &mut legend)),
                    None => out.push('.'),
                }
                out.push(' ');
            }
            if row + 1 == horizon {
                out.push_str(" ── horizon");
            }
            out.push('\n');
        }
        out.push_str("  legend: ");
        for (k, id) in legend.iter().enumerate() {
            out.push_str(&format!("{}={}  ", glyph(k), id));
        }
        out.push('\n');
    }

    if matches!(layer, Layer::Depth | Layer::Both) {
        out.push_str(&format!(
            "\nglance grid — depth m ({}×{}, row 0 = top):\n",
            g.grid, g.grid
        ));
        for row in 0..g.grid {
            out.push_str("  ");
            for col in 0..g.grid {
                let d = g.cell_depth(row * g.grid + col);
                if d.is_finite() {
                    out.push_str(&format!("{:>5.0} ", d));
                } else {
                    out.push_str("    . ");
                }
            }
            out.push('\n');
        }
    }
    out
}

fn render_proprio(p: &oracle::Proprio) -> String {
    let mut out = String::new();
    out.push_str(&format!("\nPROPRIO {}\n", p.id));
    out.push_str(&format!(
        "  position: ({:.2}, {:.2}, {:.2})\n",
        p.position[0], p.position[1], p.position[2]
    ));
    if let Some(c) = p.bounds_center {
        out.push_str(&format!(
            "  bounds center: ({:.2}, {:.2}, {:.2})\n",
            c[0], c[1], c[2]
        ));
    }
    if let Some(s) = p.size {
        out.push_str(&format!(
            "  size: {:.2} × {:.2} × {:.2}\n",
            s[0], s[1], s[2]
        ));
    }
    if let Some(yaw) = p.yaw {
        out.push_str(&format!("  yaw: {yaw:.4}\n"));
    }
    out.push_str(&format!(
        "  emissive: {}\n",
        p.emissive.as_deref().unwrap_or("none")
    ));
    out.push_str("  components:\n");
    for c in &p.components {
        out.push_str(&format!("    {:<16} {}\n", c.name, c.preview));
    }
    out
}

fn glyph(k: usize) -> char {
    const G: &[u8] = b"#@O*+=%&XYZWABCDEFHIJKLMNPQRSTUV";
    G.get(k).copied().unwrap_or(b'?') as char
}

fn value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse::<T>().map_err(|_| format!("bad value: {s}"))
}

fn parse_vec3(s: &str) -> Result<[f32; 3], String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<f32>()
                .map_err(|_| format!("bad vec3: {s}"))
        })
        .collect::<Result<_, _>>()?;
    if v.len() != 3 {
        return Err(format!("--pos needs x,y,z (got {s})"));
    }
    Ok([v[0], v[1], v[2]])
}
