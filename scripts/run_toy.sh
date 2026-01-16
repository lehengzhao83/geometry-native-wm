#!/usr/bin/env bash
set -euo pipefail

# run_toy.sh
# End-to-end runnable script for toy experiments:
#   - toy_hierarchy
#   - toy_periodic
#   - toy_pose
#
# It will:
#   1) create a venv (optional)
#   2) install requirements.txt
#   3) run training for each config
#   4) run rollout_eval.py
#   5) run ood_eval.py (if test_ood enabled in the config)
#
# Usage:
#   bash run_toy.sh
#
# Notes:
# - If you already have an environment, set USE_VENV=0 to skip venv creation:
#     USE_VENV=0 bash run_toy.sh
# - If your python is "python3", script will auto-detect.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
RUNS_DIR="${ROOT_DIR}/runs"

PY_BIN="${PY_BIN:-}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_gnwm}"
HORIZON="${HORIZON:-25}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"

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
    echo "[run_toy] USE_VENV=0, skipping venv setup."
    return
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[run_toy] Creating venv at ${VENV_DIR}"
    "${py}" -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  echo "[run_toy] Upgrading pip"
  python -m pip install --upgrade pip wheel setuptools

  if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found at ${ROOT_DIR}/requirements.txt" >&2
    exit 1
  fi

  echo "[run_toy] Installing requirements.txt"
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
}

ensure_paths() {
  if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "ERROR: configs/ directory not found at ${CONFIG_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${ROOT_DIR}/train.py" ]]; then
    echo "ERROR: train.py not found at ${ROOT_DIR}/train.py" >&2
    exit 1
  fi
  if [[ ! -f "${ROOT_DIR}/rollout_eval.py" ]]; then
    echo "ERROR: rollout_eval.py not found at ${ROOT_DIR}/rollout_eval.py" >&2
    exit 1
  fi
  if [[ ! -f "${ROOT_DIR}/ood_eval.py" ]]; then
    echo "ERROR: ood_eval.py not found at ${ROOT_DIR}/ood_eval.py" >&2
    exit 1
  fi
  mkdir -p "${RUNS_DIR}"
}

run_one() {
  local cfg_name="$1"
  local run_name="$2"
  local cfg_path="${CONFIG_DIR}/${cfg_name}"

  if [[ ! -f "${cfg_path}" ]]; then
    echo "WARN: config not found: ${cfg_path} (skipping)"
    return
  fi

  local out_dir="${RUNS_DIR}/${run_name}"
  mkdir -p "${out_dir}"

  echo ""
  echo "============================================================"
  echo "[run_toy] Training: ${cfg_name}"
  echo "         out_dir:  ${out_dir}"
  echo "============================================================"

  # We override out_dir and (optionally) epochs/batch_size by exporting env vars or editing YAML.
  # Here we only override out_dir through CLI arg supported by train.py.
  python "${ROOT_DIR}/train.py" \
    --config "${cfg_path}" \
    --override_out_dir "${out_dir}"

  local ckpt_best="${out_dir}/ckpts/best.pt"
  local ckpt_fallback=""
  if [[ -f "${ckpt_best}" ]]; then
    ckpt_fallback="${ckpt_best}"
  else
    # fallback to last checkpoint if best missing
    ckpt_fallback="$(ls -1 "${out_dir}/ckpts"/ckpt_epoch_*.pt 2>/dev/null | tail -n 1 || true)"
  fi

  echo ""
  echo "[run_toy] Rollout eval (H=${HORIZON})"
  if [[ -n "${ckpt_fallback}" && -f "${ckpt_fallback}" ]]; then
    python "${ROOT_DIR}/rollout_eval.py" \
      --config "${cfg_path}" \
      --ckpt "${ckpt_fallback}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --horizon "${HORIZON}" \
      --num_batches 30
  else
    echo "WARN: No checkpoint found for rollout_eval; running with randomly initialized model."
    python "${ROOT_DIR}/rollout_eval.py" \
      --config "${cfg_path}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --horizon "${HORIZON}" \
      --num_batches 10
  fi

  echo ""
  echo "[run_toy] OOD eval"
  if [[ -n "${ckpt_fallback}" && -f "${ckpt_fallback}" ]]; then
    python "${ROOT_DIR}/ood_eval.py" \
      --config "${cfg_path}" \
      --ckpt "${ckpt_fallback}" \
      --batch_size "${BATCH_SIZE}" \
      --max_batches 200
  else
    echo "WARN: No checkpoint found for ood_eval; running with randomly initialized model."
    python "${ROOT_DIR}/ood_eval.py" \
      --config "${cfg_path}" \
      --batch_size "${BATCH_SIZE}" \
      --max_batches 50
  fi
}

main() {
  local py
  py="$(detect_python)"
  echo "[run_toy] Using python: ${py}"

  ensure_paths
  setup_venv "${py}"

  # Run toy configs that are expected in configs/
  run_one "toy_hierarchy.yaml" "toy_hierarchy"
  run_one "toy_periodic.yaml" "toy_periodic"
  # If you have a toy_pose.yaml, it will run; otherwise it is skipped safely.
  run_one "toy_pose.yaml" "toy_pose"

  echo ""
  echo "[run_toy] Done. Check outputs in: ${RUNS_DIR}"
}

main "$@"

