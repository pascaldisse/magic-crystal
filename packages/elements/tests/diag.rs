use elements::{Collider, ContactMaterial, RigidBody, Solver, SolverConfig, Vec3};

#[test]
fn diag_incline() {
    let g = 9.81;
    let angle = 25.0_f64.to_radians();
    let mu_s = 0.6;
    let mu_d = 0.4;
    let cfg = SolverConfig {
        dt: 1.0 / 240.0,
        substeps: 16,
        iterations: 100,
        gravity: Vec3::new(0.0, -g, 0.0),
        ..Default::default()
    };
    let mut s = Solver::new(cfg);
    let (sin, cos) = (angle.sin(), angle.cos());
    let normal = Vec3::new(0.0, cos, sin);
    let along_x = Vec3::new(1.0, 0.0, 0.0);
    let up_slope = Vec3::new(0.0, sin, -cos);
    let mat = ContactMaterial {
        friction_static: mu_s,
        friction_dynamic: mu_d,
        restitution: 0.0,
        ..ContactMaterial::default()
    };
    s.collider = Some(Collider::incline(angle, 20.0, mat));
    let radius = 0.03;
    let dims = Vec3::new(0.3, 0.3, 0.3);
    let counts = (3usize, 3usize, 3usize);
    let (nx, ny, nz) = counts;
    let pm = 600.0 * dims.x * dims.y * dims.z / (nx * ny * nz) as f64;
    let step = Vec3::new(
        dims.x / (nx - 1) as f64,
        dims.y / (ny - 1) as f64,
        dims.z / (nz - 1) as f64,
    );
    let mut idx = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let lx = -dims.x * 0.5 + step.x * ix as f64;
                let ly = radius + 0.001 + step.y * iy as f64;
                let lz = -dims.z * 0.5 + step.z * iz as f64;
                let pos = along_x.scale(lx) + normal.scale(ly) + up_slope.scale(lz);
                idx.push(s.particles.add_with_radius(pos, 1.0 / pm, radius));
            }
        }
    }
    let body = RigidBody::from_indices(&s.particles, idx.clone(), 1.0, s.config.polar);
    s.rigids.push(body);
    let down = up_slope.scale(-1.0);
    let tri = s.collider.as_ref().unwrap().triangles[0];
    for k in 0..480 {
        s.step();
        if k % 40 == 0 {
            let c = s.rigids[0].centroid;
            let dd = (c - Vec3::ZERO).dot(down);
            let minsd = idx
                .iter()
                .map(|&i| (s.particles.pos[i] - tri.v0).dot(tri.normal))
                .fold(f64::INFINITY, f64::min);
            let maxv = idx
                .iter()
                .map(|&i| s.particles.vel[i].length())
                .fold(0.0f64, f64::max);
            println!(
                "tick {} c=({:.3},{:.3},{:.3}) down={:.4} maxv={:.4} minsd={:.5}",
                k, c.x, c.y, c.z, dd, maxv, minsd
            );
        }
    }
}
