# TODO — neural motion generation (animation system) · 2026-07-19, his order

STUDY NEVER IMPORT. Study refs → learned act born INSIDE our one system
(DESIGN LAW). PERFORMANCE RULE applies: motion net vs the procedural it
replaces, equal quality or dies.

## Ref 1 (his screenshot, room chat-mrrlinr6-ic04 07-19)
ARDY — Autoregressive Diffusion with Hybrid Representation for Interactive
Human Motion Generation. NVIDIA + ETH Zürich: Kaifeng Zhao, Mathis
Petrovich, Haotian Zhang, Tingwu Wang, Siyu Tang, Davis Rempe.
→ interactive/streaming human motion: autoregressive diffusion, hybrid
representation, control signal = 2D root path (screenshot: "2D Root @ 274").
→ fit: presence/NPC motion from goals (walk-to, gesture), not canned clips.

## Ref 2 (his recollection: "similar for dogs a while back")
MANN — Mode-Adaptive Neural Networks for Quadruped Motion Control,
SIGGRAPH 2018 (He Zhang, Sebastian Starke, Taku Komura, Jun Saito) —
THE dog paper; Starke's AI4Animation line around it: PFNN (human, 2017),
DeepPhase (2022). [his memory, ref identity UNVERIFIED — confirm on read]
→ fit: homunculus already ships quadruped morphology (cat, 30 bones).

## Ref 3 (his order, room chat-mruhqnjx-c9mj 07-23)
Cascadeur — https://cascadeur.com/ (Nekki). Physics-assisted keyframe
animation tool: AI autoposing (pose from few joint pins), ballistic
trajectory solve for jumps/falls, secondary-motion/overlap generation,
physics-correctness pass over hand-keys. [site claims, UNVERIFIED — study
on read]
→ fit: not a runtime net — an AUTHORING-side pattern; study the
autoposing + physics-cleanup ops for our editor-side animation tooling
(pin a few joints → solver completes the pose; keys → physically-plausible
trajectory). STUDY NEVER IMPORT applies.

## Our surface today
packages/homunculus: skeleton.rs (humanoid/quadruped/lerp) · walk.rs =
phase-driven procedural walk (splitmix64 jitter) · pose.rs FK/blend.
packages/vessel: SDF mesh + LBS skin + regions/palette.
→ phase-driven procedural = exactly the input family PFNN/MANN learned
from; upgrade path is native.

## The atom (when called)
Learned motion act inside the one system: net(control: root path/goal,
phase, morphology params) → pose stream; teacher = procedural walk +
mocap-free self-play?? (data question OPEN); gates: determinism ·
cost-vs-procedural · foot-slip metric vs walk.rs baseline · works BOTH
morphologies (humanoid + quadruped from one conditioning — homunculus
lerp already interpolates bodies).
Silicon note: small autoregressive net, per-agent, off frame path →
neural-core candidate tier (afm/CoreML door), NOT GPU (Pleroma's). [placement plan, UNVERIFIED]
