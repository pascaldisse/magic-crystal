# v7 outlier autopsy (2026-07-19, diagnosis round — NOT training)

Weights: `data/rdirect-weights-v7.bin` sha256=`636c8743faf25fa3311d8752a9e5d6ff0abd4b4ad52fb2466e2cfcaf4bbdf8ce`
(never fully ordealed before this round — full 640x480 ordeal detached this
room, see bottom).

Tool: `examples/rdirect_v7_autopsy.rs` — renders val pose `orbit_-20` at
480x360 and 640x480, settles the recurrent 39-in split net over K=8 frames
(same seed formula as the trainer), finds sparkle outliers with the ORDEAL'S
OWN definition (`err = net_lum - teacher_lum > 0.15`, strict local max over
3x3), and for each reports net/teacher/target-composite plus the target's
channel decomposition (E_full exact, D_full raw, D_full box-blurred r=2).
Raw table: `scratch/v7d-autopsy.log`.

## Verdict: COPIED, not INVENTED (mixed but COPIED-dominant)

- 480x360: 8/8 outliers found (matches sparkle 46.3/Mpx exactly: 46.3 *
  0.1728 Mpx = 8.0). **6 COPIED, 2 INVENTED.**
- 640x480: 16/16 outliers found (52.1/Mpx * 0.3072 Mpx = 16.0). **15 COPIED,
  1 INVENTED.**
- Classification: COPIED = the smoothed TARGET (what the trainer's loss
  actually optimizes, `E_full + blur(D_full, r=2)`) still has a local
  luminance excess > 0.05 over its own 3x3 neighbour mean at that pixel
  (i.e. the target itself is locally bright/structured there). INVENTED =
  target locally flat, net produced brightness with nothing to copy.
- At every single outlier pixel in both resolutions, `D_raw` and `D_blur`
  are both ~0.0003–0.005 — negligible next to `E` (0.02–1.2) and next to the
  net's overshoot itself (err 0.15–0.87). **D contributes essentially
  nothing to any of these top-50 outliers.** The outliers sit on genuine
  bright E-channel structure (specular/direct highlights — e.g. the 480x360
  cluster at y=189, x=239–260 looks like one highlight's edge run; the
  640x480 cluster at y≈247–257 similarly) where the net's output is
  systematically 1.15x–4.1x the teacher's already-bright value there, not
  a lone dark pixel spiking from near-zero.

**This falsifies the room's working hypothesis** ("the TARGET carries
firefly energy on its sharp side [via D]"). In this val pose's top-50
outlier set, D (the ~0.88%-of-frame-energy indirect channel) is not the
source — the net is over-amplifying real, already-bright E-channel detail.
The "sparkle" metric's isolated-local-max definition catches this
over-amplification the same way it would catch a lone D firefly, because it
only checks local peakedness of the *error*, not whether the underlying
pixel is otherwise structured. Two of 8 (480x360) / one of 16 (640x480) ARE
genuinely INVENTED (target locally flat, net bright anyway) — real but a
minority.

## Anonymous-channel routing (task 4)

**There is no third "anonymous" channel in the code.** The split design has
exactly two: `E` (radiance via zero-or-more specular/low-roughness bounces,
`SPEC_CHAIN_MAX_ROUGHNESS=0.25`) and `D` (radiance after a diffuse/rough
scatter — `src/rdirect.rs:495`, `~0.88% of frame energy` per the same
comment, matching the "ANONYMOUS ~0.88%" figure in the task — this IS D,
under its "diffuse-lucky" name, not a separate bucket). `src/integrator.wgsl`
`radiance_ed` assigns EVERY path contribution (BSDF chain, emissive hit, sun,
even the medium in-scatter term) to either `e_acc` or `d_acc` — nothing is
dropped. Target construction (`rdirect_train_v7.rs::render_pose`, lines
~178–202): `d_blurred = box_blur(D_full, radius=2)`, `target =
E_full(exact) + d_blurred`. So the ~0.88%-share D energy is **box-blurred
(r=2) and added back into the target**, not dropped and not left sharp in E.
Given D's raw magnitude at outlier pixels is already ~0.001–0.005 (see
above), r=2 blur is not doing meaningful work at THESE pixels because D is
already near-zero there — the blur radius is not the active variable for
this outlier population.

## Res mismatch (task 3)

Same weights, same val pose, same K/ref_frames/blur settings:

| res | sparkle/Mpx |
|---|---|
| 480x360 (train res) | 46.3 |
| 640x480 (ordeal res) | 52.1 |

Delta: **+5.8/Mpx (+12.5%) at ordeal resolution vs train resolution.** A
real but modest mismatch — not large enough by itself to explain the
100+/Mpx blowups seen in the three failed cures; those happened at the SAME
480x360 train resolution the monitor already measures at (v7d-train.log:
epoch 9 sparkle 214.1 measured at 480x360, well past this ~12% res
sensitivity).

## Recommended single next lever

Not "more D-blur" (D is already negligible at the outliers we can see) and
not another loss-shape change (three tried, banned). The actual failure
signature across all three cures (sparkle climbs monotonically epoch over
epoch while resid creeps down) plus this autopsy (outliers ride real bright
E-structure, net systematically overshoots it) points at **the net learning
to over-scale bright regions as training continues past its best checkpoint
— an optimizer/capacity issue on the E side, not a D/target-noise issue**.
Single next lever: **do NOT resume/fine-tune further past 636c8743** (every
continuation regresses); if another attempt is made, add a bounded
correction ONLY on already-bright pixels (e.g. clip predicted luminance to
some multiple of the corresponding SMOOTHED TARGET's own local value, which
this autopsy shows the net violates by 1.15x-4.1x) rather than touching the
D-blur/target construction which is already doing its job for the sampled
outliers.

## Detached ordeal (task 1)

`nohup nice -n 19 env GAIA_ORDEAL_WEIGHTS=v7 GAIA_ORDEAL_W=640
GAIA_ORDEAL_H=480 cargo run --release --example real_image_ordeal -j2 >
scratch/v7d-ordeal-640-full.log 2>&1 &` — PID **70234** (shell wrapper
70232), started this room, running against current 636c8743 weights (never
fully ordealed before). Was still running (~3 min elapsed of an expected
~13 min) when this autopsy note was written — table not available yet, see
`scratch/v7d-ordeal-640-full.log` for the eventual result; do not kill it.
