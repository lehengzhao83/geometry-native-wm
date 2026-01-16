#!/usr/bin/env bash
set -euo pipefail

# run_real.sh
# End-to-end runnable script for "real" experiments:
#   - real_video.yaml
#   - vlm_binding.yaml
#
# GUARANTEE TO WORK:
# - If you don't have any real video frames, datasets/real_wrapper.py will
#   automatically fall back to FakeData pseudo-video, so training/eval still runs.
#
# Usage:
#   bash run_real.sh
#
# Optional env vars:
#   USE_VENV=0                # skip venv creation
#   PY_BIN=python3            # choose python binary
#   VENV_DIR=./.venv_gnwm
#   HORIZON=25
#   BATCH_SIZE=32
#
# Notes:
# - train.py supports --override_out_dir only; for epochs/batch_size, edit YAML.
# - This script still passes BATCH_SIZE to eval scripts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
RUNS_DIR="${ROOT_DIR}/runs"

PY_BIN="${PY_BIN:-}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_gnwm}"
HORIZON="${HORIZON:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"

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
    echo "[run_real] USE_VENV=0, skipping venv setup."
    return
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[run_real] Creating venv at ${VENV_DIR}"
    "${py}" -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  echo "[run_real] Upgrading pip"
  python -m pip install --upgrade pip wheel setuptools

  if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found at ${ROOT_DIR}/requirements.txt" >&2
    exit 1
  fi

  echo "[run_real] Installing requirements.txt"
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
  echo "[run_real] Training: ${cfg_name}"
  echo "          out_dir:  ${out_dir}"
  echo "============================================================"

  # Training (guaranteed to run; real_wrapper will fallback if no real data)
  python "${ROOT_DIR}/train.py" \
    --config "${cfg_path}" \
    --override_out_dir "${out_dir}"

  local ckpt
  ckpt="$(pick_ckpt "${out_dir}")"

  echo ""
  echo "[run_real] Rollout eval (H=${HORIZON})"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    python "${ROOT_DIR}/rollout_eval.py" \
      --config "${cfg_path}" \
      --ckpt "${ckpt}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --horizon "${HORIZON}" \
      --num_batches 20
  else
    echo "WARN: No checkpoint found for rollout_eval; running with randomly initialized model."
    python "${ROOT_DIR}/rollout_eval.py" \
      --config "${cfg_path}" \
      --split test \
      --batch_size "${BATCH_SIZE}" \
      --horizon "${HORIZON}" \
      --num_batches 5
  fi

  echo ""
  echo "[run_real] OOD eval"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    python "${ROOT_DIR}/ood_eval.py" \
      --config "${cfg_path}" \
      --ckpt "${ckpt}" \
      --batch_size "${BATCH_SIZE}" \
      --max_batches 150
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
  echo "[run_real] Using python: ${py}"

  ensure_paths
  setup_venv "${py}"

  # Expected configs in configs/
  run_one "real_video.yaml" "real_video"
  run_one "vlm_binding.yaml" "vlm_binding"

  echo ""
  echo "[run_real] Done. Check outputs in: ${RUNS_DIR}"
  echo "[run_real] Note: If you didn't provide real frames, results are on FakeData pseudo-video (still valid for pipeline sanity)."
}

main "$@"

