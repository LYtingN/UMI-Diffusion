#!/usr/bin/env bash
# Submit the Manip_Flow UMI-PnP training job (git_workspace mode) WITHOUT ever
# writing the GitHub PAT into the tracked yaml.
#
# Every committed job yaml has a placeholder GH_PAT: "REPLACE_WITH_GITHUB_PAT".
# This script copies it to a temp file, injects the real token from $GH_PAT,
# submits that temp copy, then deletes it. The token never touches git.
#
#   export GH_PAT=github_pat_xxxxxxxx      # your GitHub token (Contents:read on the repo)
#   bash Manip_Flow/scripts/submit_flow_umi_pnp.sh                        # 默认 drawer
#   bash Manip_Flow/scripts/submit_flow_umi_pnp.sh job_flow_umi_shelf0730_h100.yaml
#   bash Manip_Flow/scripts/submit_flow_umi_pnp.sh /abs/path/to/job.yaml
#
# 第一个参数可以是 config/ 下的文件名,也可以是任意路径;省略则用 DEFAULT_JOB。
#
# Notes:
#   * git_workspace pulls code from the REMOTE, so push first: git push origin main
#   * no_proxy is set so md_ai_kit can reach the central server (local proxy is broken).
set -euo pipefail

DEFAULT_JOB="job_flow_umi_drawer_h100.yaml"

# --- reach the central server despite the broken local proxy (127.0.0.1:17897) ---
export no_proxy="127.0.0.1,localhost,.aliyuncsslb.com"
export NO_PROXY="$no_proxy"

if [[ -z "${GH_PAT:-}" ]]; then
  echo "ERROR: GH_PAT is not set. Run:  export GH_PAT=github_pat_xxxx" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/Manip_Flow/config"

# 裸文件名 -> 在 config/ 下解析;带 / 的按给定路径用
ARG="${1:-$DEFAULT_JOB}"
if [[ "$ARG" == */* ]]; then
  TEMPLATE="$ARG"
else
  TEMPLATE="$CONFIG_DIR/$ARG"
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: job config not found: $TEMPLATE" >&2
  echo "可用的 job yaml:" >&2
  ls -1 "$CONFIG_DIR"/job_*.yaml 2>/dev/null | sed 's#.*/#  #' >&2
  exit 1
fi

# temp copy in a private dir; guaranteed cleanup even on error/Ctrl-C
TMP="$(mktemp -t job_flow_umi_pnp.XXXXXX.yaml)"
chmod 600 "$TMP"
cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

# inject the real token in place of the placeholder (only in the temp copy)
sed "s#REPLACE_WITH_GITHUB_PAT#${GH_PAT}#" "$TEMPLATE" > "$TMP"

if grep -q "REPLACE_WITH_GITHUB_PAT" "$TMP"; then
  echo "ERROR: placeholder not replaced — check the template still contains REPLACE_WITH_GITHUB_PAT" >&2
  exit 1
fi

echo "[submit] $(basename "$TEMPLATE") — git_workspace job, token injected into temp copy (not committed)"
md_ai_kit submit "$TMP"
