// scratch probe — derived realm values for the A2 binding + clip
fn main() {
    use std::path::Path;
    use crystal::{Core, load_world_dir};
    let world_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
    let mut core = Core::default();
    load_world_dir(&world_path, &mut core.world).unwrap();
    let srcs = scrying_glass::scene::emissive_sources(&core.world).unwrap();
    println!("emissive sources ({}):", srcs.len());
    for s in &srcs { println!("  {:20} pos={:?} color={:?}", s.id, s.position, s.color); }
    let counter = scrying_glass::scene::top_flat_surface_y(&core.world, "naruko_stall_massing").unwrap();
    println!("stall top flat surface y = {:?}", counter);
    // nearest emitter to the plume xz (-1, 25.6)
    let plume = [-1.0f32, 3.0, 25.6];
    let mut best: Option<(&str, f32)> = None;
    for s in &srcs {
        let d = ((s.position[0]-plume[0]).powi(2)+(s.position[2]-plume[2]).powi(2)).sqrt();
        if best.map_or(true,|(_,bd)| d<bd) { best=Some((&s.id,d)); }
    }
    println!("nearest emitter to plume xz: {:?}", best);
}
