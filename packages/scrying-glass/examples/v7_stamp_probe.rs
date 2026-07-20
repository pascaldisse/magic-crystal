use scrying_glass::rdirect::{stamp_path_for, verify_stamp};
fn main() {
    let w = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rdirect-weights-v7.bin");
    let bytes = std::fs::read(&w).expect("read v7 weights");
    let stamp = stamp_path_for(&w);
    println!("stamp path = {}", stamp.display());
    println!("verify_stamp = {}", verify_stamp(&bytes, &stamp));
}
