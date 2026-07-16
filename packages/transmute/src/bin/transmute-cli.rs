//! transmute-cli — transmute a test mesh and print Great Chain stats.
//!
//! Usage:
//!   transmute-cli [sphere|cube|cylinder|cone] [--res N] [--out FILE.cbdg]
//! Defaults to a high-res sphere. Prints levels, per-level cluster counts and
//! triangle counts, partitioner backend, group count, and (if --out) the
//! serialized size.

use transmutation::{
    subdivided_cube, transmute_default, uv_sphere, Dag, GaiaPrimitive, Mesh, TransmuteParams,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut kind = "sphere".to_string();
    let mut res: usize = 128;
    let mut out: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--res" => {
                i += 1;
                res = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(res);
            }
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            other => kind = other.to_string(),
        }
        i += 1;
    }

    let mesh: Mesh = match kind.as_str() {
        "cube" => subdivided_cube(2.0, res),
        "cylinder" => GaiaPrimitive::Cylinder {
            radius_top: 0.5,
            radius_bottom: 1.0,
            height: 3.0,
            radial_segments: res,
            height_segments: res / 2 + 1,
        }
        .tessellate(),
        "cone" => GaiaPrimitive::Cone {
            radius: 1.0,
            height: 2.0,
            radial_segments: res,
            height_segments: res / 2 + 1,
        }
        .tessellate(),
        _ => uv_sphere(1.0, res, (res * 3) / 4),
    };

    let params = TransmuteParams::default();
    let dag = transmute_default(&mesh, &params).expect("transmute");
    print_stats(&kind, res, &mesh, &dag);

    if let Some(path) = out {
        let bytes = transmutation::serialize(&dag).expect("serialize");
        std::fs::write(&path, &bytes).expect("write out");
        println!("wrote {} ({} bytes)", path, bytes.len());
    }
}

fn print_stats(kind: &str, res: usize, mesh: &Mesh, dag: &Dag) {
    println!("== transmutation Great Chain stats ==");
    println!("mesh:        {kind} @ res {res}");
    println!("input verts: {}", mesh.vertices.len());
    println!("input tris:  {}", mesh.tri_count());
    println!("partitioner: {}", dag.partitioner);
    println!("levels:      {}", dag.level_count());
    println!("clusters:    {} total", dag.clusters.len());
    println!("groups:      {} total", dag.groups.len());
    println!(
        "leaf tri sum:{} (== input: {})",
        dag.leaf_tri_sum(),
        dag.leaf_tri_sum() == mesh.tri_count()
    );
    println!();
    println!(
        "{:<6} {:>10} {:>12} {:>12}",
        "level", "clusters", "tris", "avg tris/cl"
    );
    for (lvl, ids) in dag.levels.iter().enumerate() {
        let tris: usize = ids.iter().map(|&id| dag.cluster(id).tri_count()).sum();
        let avg = if ids.is_empty() {
            0.0
        } else {
            tris as f32 / ids.len() as f32
        };
        println!("{:<6} {:>10} {:>12} {:>12.1}", lvl, ids.len(), tris, avg);
    }
}
