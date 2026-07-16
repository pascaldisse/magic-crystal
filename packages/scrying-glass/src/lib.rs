use crystal::{Core, GaiaPackage, PackageManifest};

pub mod bvh;
pub mod input;
pub mod integrator;
pub mod player;
pub mod scene;

pub const PACKAGE_NAME: &str = "scrying-glass";
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct ScryingGlassPackage;

impl GaiaPackage for ScryingGlassPackage {
    fn register(&self, core: &mut Core) {
        core.register_package(PackageManifest::new(PACKAGE_NAME, PACKAGE_VERSION));
    }
}
