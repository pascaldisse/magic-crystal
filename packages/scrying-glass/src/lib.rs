use crystal::{Core, GaiaPackage, PackageManifest};

pub mod bloodbend;
pub mod bvh;
pub mod denoiser;
pub mod denoiser_dataset;
pub mod denoiser_gpu;
pub mod error_metric;
pub mod horizon;
pub mod input;
pub mod integrator;
pub mod physics;
pub mod player;
pub mod presence;
pub mod retina;
pub mod scene;
pub mod upscaler;
pub mod upscaler_dataset;
pub mod upscaler_gpu;

pub const PACKAGE_NAME: &str = "scrying-glass";
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct ScryingGlassPackage;

impl GaiaPackage for ScryingGlassPackage {
    fn register(&self, core: &mut Core) {
        core.register_package(PackageManifest::new(PACKAGE_NAME, PACKAGE_VERSION));
    }
}
