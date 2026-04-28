#!/usr/bin/env bash
# Vary network size (gradient buffer size) at a fixed rank count and algorithm.
# Shows how each all-reduce strategy scales with message size.
#
#   ./scripts/netsize_sweep.sh
#   P=8 ALGO=hd ./scripts/netsize_sweep.sh
#
# Output: netsize_p<P>_<algo>.csv

set -euo pipefail

P=${P:-4}
ALGO=${ALGO:-ring}
EPOCHS=${EPOCHS:-3}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
OUT="netsize_p${P}_${ALGO}.csv"

# Each entry is a label and a --layers string.
# Approx param counts:  small~50K  medium~235K  large~920K  xlarge~2.1M
CONFIGS=(
    "small:784,64,10"
    "medium:784,256,128,10"
    "large:784,512,256,128,10"
    "xlarge:784,1024,512,256,10"
)

if [[ ! -x build/train_mpi ]]; then
    echo "build/train_mpi missing. Run 'make mpi' first." >&2
    exit 1
fi

echo "config,params_approx,algo,compute_ms,allreduce_ms,iter_ms,allreduce_pct" > "$OUT"

for entry in "${CONFIGS[@]}"; do
    label="${entry%%:*}"
    layers="${entry##*:}"
    echo "=== config=$label  layers=$layers  algo=$ALGO  P=$P ==="

    LOG=$(mktemp)
    srun -n "$P" ./build/train_mpi \
        --algo "$ALGO" --layers "$layers" \
        --epochs "$EPOCHS" --global-batch "$GLOBAL_BATCH" \
        --warmup 5 | tee "$LOG"

    # Pull param count from the "Total params:" line
    params=$(awk '/Total params:/ { print $3 }' "$LOG")

    comp=$(awk '/avg compute/    { print $5 }' "$LOG")
    comm=$(awk '/avg all-reduce/ { print $3 }' "$LOG")
    iter=$(awk '/avg total/      { print $5 }' "$LOG")
    pct=$( awk '/avg all-reduce/ { gsub(/[()%]/,"",$5); print $5 }' "$LOG")

    echo "$label,$params,$ALGO,$comp,$comm,$iter,$pct" >> "$OUT"
    rm "$LOG"
    echo
done

echo "wrote $OUT"
