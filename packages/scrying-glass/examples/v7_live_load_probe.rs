//! Ghoul probe: does RdirectLive::from_system accept the v7 (39-in split) weights blob?
use scrying_glass::rdirect_live::RdirectLive;

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v7.bin");
    let weights = std::fs::read(&path).expect("read v7 weights");
    match RdirectLive::from_system(&weights) {
        Ok(live) => println!("LOADED OK in_features={} out_channels={}", live.in_features(), live.out_channels()),
        Err(e) => println!("REJECTED: {e}"),
    }
}
