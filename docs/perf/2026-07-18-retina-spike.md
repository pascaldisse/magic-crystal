# RETINA — Matrix-vision spike · 2026-07-18

## Scope

- `GET /retina` → native-window HTTP organ; render-thread request/reply; no net state, screenshot, PNG, radiance, secondary rays.
- Source → post-transmute leaf triangles; packed native BVH; primary hit only.
- Base defaults → `64×64`; `depth,normal,entity-id,material-id,world-pos`.
- Pose params → `pos=x,y,z&yaw=&pitch=&fov=&w=&h=`; same eye convention as `/pose`/`/scry`.
- Layers → `layers=depth,normal,entity-id,material-id,world-pos`; omitted channel absent from JSON.
- IDs → per-image `u32` arrays + `entity_table`/`material_table`; `4294967295` = miss. Depth miss = `-1`; normal/world-pos miss = `[0,0,0]`.
- Fovea → `fovea=center_x,center_y,radius,scale`; `;` separates levels. Each level traces its own smaller image-plane window at `base_res×scale`; no base-grid upscale.
- `motion` → HTTP 400: `UNVERIFIED: no previous-frame plumbing`.

## Wire shape

```sh
curl 'http://127.0.0.1:8440/retina?pos=0,1.7000003,44&yaw=0&pitch=-0.03&fov=20&w=8&h=8&layers=depth,normal,entity-id,material-id,world-pos&fovea=0.5,0.5,0.25,2'
```

```json
{"base":{"resolution":[8,8],"sample_window":[0.5,0.5,1.0],"ray_model":"native-bvh-primary","miss_depth":-1.0},"fovea":[{"center":[0.5,0.5],"radius":0.25,"scale":2,"image":{"resolution":[16,16],"sample_window":[0.5,0.5,0.25]}}]}
```

## Live proof · :8440

- `/pose` → eye `[0,1.7000003,44]`; yaw `0`; pitch overridden `-0.03`.
- Scene truth → `naruko_seawall`: front face `z=19`; authored wall center `z=18`, depth `2` → front face `z=19`.
- Retina cell 38 → entity `naruko_seawall`; depth `25.185371`; normal `[0,0,1]`; world-pos `[2.758174,0.397892,19.0]`.
- Depth check → `sqrt(2.758174² + (0.397892-1.7000003)² + (19-44)²) = 25.18537`; retina `25.185371`.
- Normal check → scene's north box face = `[0,0,1]`; retina exact `[0,0,1]`.
- Fovea proof → requested `8×8`, `scale=2` → returned `16×16`, window `[0.5,0.5,0.25]`.
- Motion request → HTTP `400`; body `motion is UNVERIFIED: no previous-frame plumbing`.

## Gaps

- Motion vectors, cluster-id, anim-phase, attention fetch, diff/watch → absent.
- CPU primary traversal/rebuilt per request → Act-1 correctness spike; not a GPU vis-buffer performance claim.
- Material tokens = renderer linear-material recipes; not authored material asset names.
