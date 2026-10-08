#!/usr/bin/env bash
# Run tools/infer_navsim_dump.py for one seed, sharded over $GPUS; idempotent (fills missing tokens).
# Usage: bash scripts/_dump_sharded.sh <merged DCP> <gear split> <out dir> <seed> [episodes file]
set -u
MODEL=$1; GEAR=$2; OUT=$3; SEED=$4; EPS=${5:-}
cd "$REPO"
mkdir -p "$OUT/test" "$OUT/logs"
for ROUND in 1 2 3; do
    MISS=$(python - "$OUT/test" "$GEAR" "$EPS" <<'PY'
import json, os, sys
out, gear, eps_f = sys.argv[1], sys.argv[2], sys.argv[3]
ep2tok = {json.loads(l)["episode_index"]: json.loads(l)["navsim_token"]
          for l in open(os.path.join(gear, "meta", "episodes.jsonl")) if l.strip()}
want = [int(x) for x in open(eps_f).read().split()] if eps_f else sorted(ep2tok)
have = {f[:-4] for f in os.listdir(out)}
print(" ".join(str(e) for e in want if ep2tok[e] not in have))
PY
)
    NM=$(echo "$MISS" | wc -w)
    [ "$NM" -eq 0 ] && { echo "seed$SEED complete: $(ls "$OUT/test" | wc -l) npy"; exit 0; }
    IFS=',' read -ra G <<< "$GPUS"
    NG=${#G[@]}; PER=$(( (NM + NG - 1) / NG ))
    read -ra MARR <<< "$MISS"
    echo "seed$SEED round $ROUND: $NM episodes over $NG GPUs"
    PIDS=()
    for i in $(seq 0 $((NG-1))); do
        CHUNK="${MARR[@]:$((i*PER)):$PER}"
        [ -z "$CHUNK" ] && continue
        CUDA_VISIBLE_DEVICES=${G[$i]} python tools/infer_navsim_dump.py \
            --checkpoint-path "$MODEL" --gear-root "$GEAR" --out-dir "$OUT" --split test \
            --seed "$SEED" --history-frames 4 --episodes $CHUNK > "$OUT/logs/dump_s${SEED}_g$i.log" 2>&1 &
        PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || true; done
done
echo "seed$SEED: $(ls "$OUT/test" | wc -l) npy (some episodes may still be missing, check $OUT/logs)"
