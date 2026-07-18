#!/usr/bin/env bash
# S20 bench (THE PURGE): offscreen, release, player-shaped walk, >=600 frames,
# sleep ON, DEFAULT weights = v2. Pleroma-or-black only; NO belief/pose eyes.
# On-demand /scry?eye=presented (the only eye). NO WINDOWS.
# Usage: s20-bench.sh <label> <port> [extra env VAR=val ...]
set -u
cd "$(dirname "$0")/../../../.."   # repo root
LABEL="${1:?label}"; PORT="${2:?port}"; shift 2
PROOF=packages/scrying-glass/proof/neural-live
BIN=target/release/scrying-glass
LOG="$PROOF/s20-$LABEL.log"

env GAIA_NEURAL_LIVE=1 GAIA_NATIVE_OFFSCREEN=true \
    GAIA_NATIVE_HUD=false GAIA_NATIVE_SLEEP=1 GAIA_NATIVE_PORT="$PORT" \
    GAIA_WORLD="$PWD/worlds/naruko" "$@" \
    "$BIN" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

for i in $(seq 1 60); do
  curl -s "http://127.0.0.1:$PORT/pose" >/dev/null 2>&1 && break
  sleep 0.5
done

# player-shaped: hold W in bursts, >=600 frames at ~50fps (~15s + margin).
for i in $(seq 1 60); do
  curl -s -X POST "http://127.0.0.1:$PORT/walk" \
       -d '{"keys":["KeyW"],"ticks":16}' >/dev/null 2>&1
  sleep 0.35
done

sleep 4
BUDGET=$(curl -s "http://127.0.0.1:$PORT/budget")
STATE=$(curl -s "http://127.0.0.1:$PORT/state")
echo "=== /budget ($LABEL) ===" | tee -a "$LOG"
echo "$BUDGET" | tee -a "$LOG"
echo "$BUDGET" > "$PROOF/s20-$LABEL.budget.json"
echo "$STATE"  > "$PROOF/s20-$LABEL.state.json"
echo "=== /state ($LABEL) ===" | tee -a "$LOG"
echo "$STATE" | tee -a "$LOG"

curl -s "http://127.0.0.1:$PORT/scry?eye=presented" -o "$PROOF/s20-$LABEL-presented.png"

echo "=== wall-fps tail ($LABEL) ===" | tee -a "$LOG"
grep "WALL-FPS" "$LOG" | tail -3
