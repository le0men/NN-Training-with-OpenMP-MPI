#!/usr/bin/env bash
# Run allreduce_bench at power-of-2 and non-power-of-2 rank counts and
# collect only the HD rows into a single CSV for the fold-overhead plot.
#
#   ./scripts/hd_nonpower2_bench.sh
#   RANKS="4 6 8" SIZES="1024,65536,1048576,4194304" ./scripts/hd_nonpower2_bench.sh

set -euo pipefail

RANKS=${RANKS:-"4 6 8"}
SIZES=${SIZES:-"1024,8192,65536,262144,1048576,4194304"}
REPS=${REPS:-30}
OUT="hd_nonpower2_comparison.csv"

if [[ ! -x build/allreduce_bench ]]; then
    echo "build/allreduce_bench missing. Run 'make bench' first." >&2
    exit 1
fi

echo "ranks,size,median_ms,min_ms,gbs_eff" > "$OUT"

for P in $RANKS; do
    echo "=== P=$P ==="
    LOG=$(mktemp)
    srun -n "$P" ./build/allreduce_bench --sizes "$SIZES" --reps "$REPS" | tee "$LOG"

    awk -v p="$P" '
        /^[[:space:]]+[0-9]/ && $2 == "hd" {
            printf "%d,%s,%s,%s,%s\n", p, $1, $3, $4, $5
        }' "$LOG" >> "$OUT"

    rm "$LOG"
    echo
done

echo "wrote $OUT"
