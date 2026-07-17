use elements::fluid::{fill, FluidPoolSpec};
// surface flatness: bin fluid by (x,z) cell, take max-y per column, report spread center vs edge
fn flatness(s:&elements::Solver)->(f64,f64,f64){
    // center columns |x|,|z|<0.15 ; edge columns |x|or|z|>0.45
    let (mut cmax,mut emax,mut gmax)=(f64::NEG_INFINITY,f64::NEG_INFINITY,f64::NEG_INFINITY);
    for &i in &s.fluid_particles { let q=s.particles.pos[i];
        gmax=gmax.max(q.y);
        if q.x.abs()<0.12 && q.z.abs()<0.12 { cmax=cmax.max(q.y); }
        if q.x.abs()>0.45 || q.z.abs()>0.45 { emax=emax.max(q.y); }
    }
    (cmax,emax,gmax)
}
fn maxspd(s:&elements::Solver)->f64{ s.fluid_particles.iter().map(|&i|s.particles.vel[i].length()).fold(0.0,f64::max)}
fn main(){
    for tk in [0.0,0.02,0.05,0.1] {
        let spec=FluidPoolSpec{spacing:0.08,..Default::default()};
        let mut p=fill(spec);
        let mut cfg=p.solver.fluid.unwrap(); cfg.tensile_k=tk; p.solver.fluid=Some(cfg);
        for _ in 0..300 { p.solver.step(); }
        let (c,e,g)=flatness(&p.solver);
        println!("tensile_k={:.2}: center_top={:.3} edge_top={:.3} dome={:.3} globalmax={:.3} maxspd={:.3}",tk,c,e,c-e,g,maxspd(&p.solver));
    }
}
