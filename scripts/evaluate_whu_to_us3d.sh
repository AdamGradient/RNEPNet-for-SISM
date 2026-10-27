#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 DATA_ROOT [evaluate.py options]" >&2
    exit 2
fi
DATA_ROOT="$1"
shift
exec python "$ROOT/evaluate.py" --dataset us3d --data-root "$DATA_ROOT" --checkpoint "$ROOT/weights/whu.pth" "$@"
