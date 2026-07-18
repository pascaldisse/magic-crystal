#!/usr/bin/env bash
# S15 bench: offscreen, release, player-shaped walk, >=600 frames, curl /budget
# (outside + world_stages sub-table + wall-fps from stderr). NO WINDOWS.
# Usage: s15-bench.sh <label> <port> [extra env assignments...]
set -u
cd "$(dirname "$0")/../../../.."   # repo root
LABEL="${1:?label}"; PORT="${2:?port}"; shift 2
PROOF=packages/scrying-glass/proof/neural-live
BIN=target/release/scrying-glass
LOG="$PROOF/s15-$LABEL.log"

env GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true GAIA_NATIVE_NET_PRESENT=true \
    GAIA_NATIVE_HUD=false GAIA_NATIVE_PORT="$PORT" \
    GAIA_WORLD="$PWD/worlds/naruko" "$@" \
    "$BIN" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

# wait for server up
for i in $(seq 1 60); do
  curl -s "http://127.0.0.1:$PORT/pose" >/dev/null 2>&1 && break
  sleep 0.5
done

# player-shaped: hold W (walk forward) in bursts across the run
for i in $(seq 1 40); do
  curl -s -X POST "http://127.0.0.1:$PORT/walk" \
       -d '{"keys":["KeyW"],"ticks":16}' >/dev/null 2>&1
  sleep 0.4
done

# let it accumulate >=600 frames total, then snapshot /budget
sleep 4
BUDGET=$(curl -s "http://127.0.0.1:$PORT/budget")
echo "=== /budget ($LABEL) ===" | tee -a "$LOG"
echo "$BUDGET" | tee -a "$LOG"
echo "$BUDGET" > "$PROOF/s15-$LABEL.budget.json"
echo "=== wall-fps tail ($LABEL) ===" | tee -a "$LOG"
grep "WALL-FPS" "$LOG" | tail -3
