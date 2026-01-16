#!/usr/bin/env bash
set -euo pipefail

# reproduce_main_results.sh
# One-command "main results" reproduction script (GUARANTEED TO RUN)
#
# It will run:
#   1) Euclidean baselines (euclidean latent) on toy_hierarchy + toy_periodic + toy_pose(optional)
#   2) Manifold models (hyperbolic/circle/product) on the same tasks
#   3) For each run: train -> rollout_eval -> ood_eval
#
# GUARANTEE:
# - If real_video / vlm_binding are included, real_wrapper falls back to FakeData pseudo-video
# - If toy_pose.yaml is missing, it is skipped
# - If a geometry-specific config is missing, it is skipped (no hard failure)
#
# Expected configs directory layout:
#   configs/
#     toy_hierarchy.yaml
#     toy_periodic.yaml
#     toy_pose.yaml (optional)
#     real_video.yaml (optional)
#     vlm_binding.yaml (optional)
#
# Important:
# - This script does NOT attempt to edit YAML on the fly (robust & portable).
# - To reproduce "Euclid vs Manifold", you should have separate YAMLs per geometry, e.g.:
#     toy_hierarchy_euclid.yaml
#     toy_hierarchy_hyperbolic.yaml
#     toy_periodic_euclid.yaml
#     toy_periodic_circle.yaml
#     toy_pose_euclid.yaml
#     toy_pose_circle.yaml
#     toy_*_product.yaml (optional)
#
# If you only have the base configs (toy_hierarchy.yaml, toy_periodic.yaml),
# this script still runs them and produces results (pipeline sanity).
#
# Usage:
#   bash reproduce_main_results.sh
#
# Optional env vars:
#   USE_VENV=0
#   PY_BIN=python3
#   VENV_DIR=./.venv_gnwm
#   HORIZON=25
#   BATCH_SIZE=64
#   MAX_EVAL_BATCHES=200
#
# Output:
#   runs/main_results/<run_name>/*
#
# This script is "hard to break": it checks file existence and skips missing configs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
RUNS_DIR="${ROOT_DIR}/runs/main_results"

PY_BIN="${PY_BIN:-}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_gnwm}"
HORIZON="${HORIZON:-25}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-200}"

detect_python() {
  if [[ -n "${PY_BIN}" ]]; then
    echo "${PY_BIN}"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo "ERROR: python not found" >&2
  exit 1
}

setup_venv() {
  local py="$1"
  if [[ "${USE_VENV}" == "0" ]]; then
    echo "[reproduce] USE_VENV=0, skipping venv setup."
    return
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[reproduce] Creating venv at ${VENV_DIR}"
    "${py}" -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  echo "[reproduce] Upgrading pip"
  python -m pip install --upgrade pip wheel setuptools

  if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found at ${ROOT_DIR}/requirements.txt" >&2
    exit 1
  fi

  echo "[reproduce] Installing requirements.txt"
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
}

ensure_paths() {
  if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "ERROR: configs/ directory not found at ${CONFIG_DIR}" >&2
    exit 1
  fi
  for f in train.py rollout_eval.py ood_eval.py; do
    if [[ ! -f "${ROOT_DIR}/${f}" ]]; then
      echo "ERROR: ${f} not found at ${ROOT_DIR}/${f}" >&2
      exit 1
    fi
  done
  mkdir -p "${RUNS_DIR}"
}

pick_ckpt() {
  local out_dir="$1"
  local best="${out_dir}/ckpts/best.pt"
  if [[ -f "${best}" ]]; then
    echo "${best}"
    return
  fi
  local last
  last="$(ls -1 "${out_dir}/ckpts"/ckpt_epoch_*.pt 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${last}" && -f "${last}" ]]; then
    echo "${last}"
    return
  fi
  echo ""
}

run_cfg() {
  local cfg_path="$1"
  local run_name="$2"

  if [[ ! -f "${cfg_path}" ]]; then
    echo "SKIP: missing config ${cfg_path}"
    return
  fi

  local out_dir="${RUNS_DIR}/${run_name}"
  mkdir -p "${out_dir}"

  echo ""
  echo "============================================================"
  echo "[reproduce] RUN: ${run_name}"
  echo "           CFG: ${cfg_path}"
  echo "============================================================"

  python "${ROOT_DIR}/train.py" --config "${cfg_path}" --override_out_dir "${out_dir}"

  local ckpt
  ckpt="$(pick_ckpt "${out_dir}")"

  echo "[reproduce] rollout_eval (H=${HORIZON})"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    python "${ROOT_DIR}/rollout_eval.py" \
      --config "${cfg_path}" \
      --ckpt "${ckpt}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --horizon "${HORIZON}" \
      --num_batches 30
  else
    echo "WARN: no checkpoint found, rollout_eval with random model."
    python "${ROOT_DIR}/rollout_eval.py" \
      --config "${cfg_path}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --horizon "${HORIZON}" \
      --num_batches 10
  fi

  echo "[reproduce] ood_eval"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    python "${ROOT_DIR}/ood_eval.py" \
      --config "${cfg_path}" \
      --ckpt "${ckpt}" \
      --batch_size "${BATCH_SIZE}" \
      --max_batches "${MAX_EVAL_BATCHES}"
  else
    echo "WARN: no checkpoint found, ood_eval with random model."
    python "${ROOT_DIR}/ood_eval.py" \
      --config "${cfg_path}" \
      --batch_size "${BATCH_SIZE}" \
      --max_batches 50
  fi
}

main() {
  local py
  py="$(detect_python)"
  echo "[reproduce] Using python: ${py}"

  ensure_paths
  setup_venv "${py}"

  # ------------------------------------------------------------
  # Main results grid (robust): tries multiple common filenames.
  # You can add/remove configs freely; missing ones are skipped.
  # ------------------------------------------------------------

  # Toy Hierarchy
  run_cfg "${CONFIG_DIR}/toy_hierarchy_euclid.yaml"     "toy_hierarchy_euclid"
  run_cfg "${CONFIG_DIR}/toy_hierarchy_hyperbolic.yaml" "toy_hierarchy_hyperbolic"
  run_cfg "${CONFIG_DIR}/toy_hierarchy_product.yaml"    "toy_hierarchy_product"
  # Fallback: if you only have a single toy_hierarchy.yaml, run it once
  run_cfg "${CONFIG_DIR}/toy_hierarchy.yaml"            "toy_hierarchy_default"

  # Toy Periodic
  run_cfg "${CONFIG_DIR}/toy_periodic_euclid.yaml"      "toy_periodic_euclid"
  run_cfg "${CONFIG_DIR}/toy_periodic_circle.yaml"      "toy_periodic_circle"
  run_cfg "${CONFIG_DIR}/toy_periodic_product.yaml"     "toy_periodic_product"
  run_cfg "${CONFIG_DIR}/toy_periodic.yaml"             "toy_periodic_default"

  # Toy Pose (optional)
  run_cfg "${CONFIG_DIR}/toy_pose_euclid.yaml"          "toy_pose_euclid"
  run_cfg "${CONFIG_DIR}/toy_pose_circle.yaml"          "toy_pose_circle"
  run_cfg "${CONFIG_DIR}/toy_pose_product.yaml"         "toy_pose_product"
  run_cfg "${CONFIG_DIR}/toy_pose.yaml"                 "toy_pose_default"

  # Real video + VLM binding (optional; guaranteed to run via fallback)
  run_cfg "${CONFIG_DIR}/real_video_euclid.yaml"        "real_video_euclid"
  run_cfg "${CONFIG_DIR}/real_video_product.yaml"       "real_video_product"
  run_cfg "${CONFIG_DIR}/real_video.yaml"               "real_video_default"

  run_cfg "${CONFIG_DIR}/vlm_binding_euclid.yaml"       "vlm_binding_euclid"
  run_cfg "${CONFIG_DIR}/vlm_binding_product.yaml"      "vlm_binding_product"
  run_cfg "${CONFIG_DIR}/vlm_binding.yaml"              "vlm_binding_default"

  echo ""
  echo "[reproduce] Done."
  echo "[reproduce] All outputs are in: ${RUNS_DIR}"
}

main "$@"

