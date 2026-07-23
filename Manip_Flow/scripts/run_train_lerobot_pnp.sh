#!/usr/bin/env bash
# Launch Manip_Flow LeRobot-UMI pnp training with the DINOv3 backbone served
# from the pre-populated NAS HF cache (no download at startup).
#
# Usage:
#   bash scripts/run_train_lerobot_pnp.sh /path/to/lerobot_dataset_root [extra hydra overrides...]
#
set -euo pipefail

# --- HF cache: DINOv3 (vit_base_patch16_dinov3.lvd1689m) is already downloaded here ---
export HF_HOME=/data/nas_ray/home/eason.er/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
# Force fully-offline so a cache miss FAILS LOUDLY instead of silently
# re-downloading to ~/.cache. Drop this line if you ever need to fetch a new model.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# repo root that must be on sys.path (holds the top-level ``Manip_Flow`` package)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/opt/miniforge/envs/starvla/bin/python

DATASET_PATH="${1:-/data/nas_ray/home/eason.er/datasets/umi-pnp-table}"
shift || true

exec "$PY" "$REPO_ROOT/Manip_Flow/scripts/train_flow_umi.py" \
    --config-name train_flow_lerobot_umi_pnp \
    task.dataset_path="$DATASET_PATH" \
    "$@"
