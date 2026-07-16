//! Relic forge — render a 128×128 cornell-ish box with Lumen Naturae and
//! write it to relics/l0-cornell.png. Low-res + noisy is FINE and honest;
//! noise is granted, lies are not. Every value here is SCENE DATA (authoring
//! parameters), not integrator hardcoding.
//!
//! Run:  cargo run -p lumen --release --example cornell

use lumen::{color, vec3, write_png, Camera, Film, Material, Params, Scene, Shape, Vec3};
use std::path::Path;

fn main() {
    // Room: x∈[-1,1], y∈[0,2], z∈[-2,0]. Camera in front at z=+3 looking in.
    let white = color::parse("white").unwrap();
    let red = color::parse("crimson").unwrap();
    let green = color::parse("green").unwrap();

    let mut scene = Scene::new();
    // walls (infinite planes bounding the room)
    scene
        .add(
            Shape::Plane {
                point: vec3(0.0, 0.0, 0.0),
                normal: vec3(0.0, 1.0, 0.0),
            },
            Material::lambertian(white),
        ) // floor
        .add(
            Shape::Plane {
                point: vec3(0.0, 2.0, 0.0),
                normal: vec3(0.0, -1.0, 0.0),
            },
            Material::lambertian(white),
        ) // ceiling
        .add(
            Shape::Plane {
                point: vec3(0.0, 0.0, -2.0),
                normal: vec3(0.0, 0.0, 1.0),
            },
            Material::lambertian(white),
        ) // back
        .add(
            Shape::Plane {
                point: vec3(-1.0, 0.0, 0.0),
                normal: vec3(1.0, 0.0, 0.0),
            },
            Material::lambertian(red),
        ) // left
        .add(
            Shape::Plane {
                point: vec3(1.0, 0.0, 0.0),
                normal: vec3(-1.0, 0.0, 0.0),
            },
            Material::lambertian(green),
        ); // right

    // ceiling area light (emissive sphere just under the ceiling)
    let emit_intensity = 18.0;
    scene.add(
        Shape::Sphere {
            center: vec3(0.0, 2.35, -1.0),
            radius: 0.5,
        },
        Material::emissive(white * emit_intensity),
    );

    // two diffuse occupants
    scene
        .add(
            Shape::Sphere {
                center: vec3(-0.42, 0.42, -1.25),
                radius: 0.42,
            },
            Material::lambertian(Vec3::splat(0.75)),
        )
        .add(
            Shape::Box {
                min: vec3(0.15, 0.0, -0.9),
                max: vec3(0.7, 0.7, -0.35),
            },
            Material::lambertian(Vec3::splat(0.75)),
        );

    let cam = Camera::new(
        vec3(0.0, 1.0, 3.2),
        vec3(0.0, 1.0, -1.0),
        vec3(0.0, 1.0, 0.0),
        45.0,
        1.0,
    );

    let params = Params {
        spp: 512,
        max_bounces: 16,
        rr_start: 4,
        seed: 1, // LOVE = 1, the one constant
        ..Params::default()
    };

    let (w, h) = (128u32, 128u32);
    eprintln!("[cornell] rendering {w}×{h} spp={} …", params.spp);
    let film = Film::render(&scene, &cam, w, h, &params);

    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("relics/l0-cornell.png");
    let exposure = 1.0;
    write_png(&film, &out, exposure).expect("write relic");
    eprintln!("[cornell] wrote {}", out.display());
}
