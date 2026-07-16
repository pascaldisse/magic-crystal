use crystal::{Core, GaiaPackage, PackageManifest};

pub const PACKAGE_NAME: &str = "scrying-glass";
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct ScryingGlassPackage;

impl GaiaPackage for ScryingGlassPackage {
    fn register(&self, core: &mut Core) {
        core.register_package(PackageManifest::new(PACKAGE_NAME, PACKAGE_VERSION));
    }
}
