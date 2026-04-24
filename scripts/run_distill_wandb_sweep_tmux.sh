#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION_NAME:-distill_wandb_sweep}"
SWEEP_YAML="${SWEEP_YAML:-sweeps/distill_wandb_sweep.yaml}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5}"
AGENTS_PER_GPU="${AGENTS_PER_GPU:-1}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${SESSION}.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install tmux first."
  exit 1
fi
if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI not found. Install with: python3 -m pip install --user wandb"
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

launch_agent_window() {
  local window_name="$1"
  local gpu="$2"
  local log_file="$3"
  local sweep_target="$4"
  local tmux_cmd="cd \"${ROOT_DIR}\" && CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1} wandb agent \"${sweep_target}\" 2>&1 | tee -a \"${log_file}\""

  if [[ "${window_name}" == "__first__" ]]; then
    tmux new-session -d -s "${SESSION}" -n "${5}" "${tmux_cmd}"
  else
    tmux new-window -t "${SESSION}" -n "${window_name}" "${tmux_cmd}"
  fi
}

create_sweep() {
  local output
  output="$(cd "${ROOT_DIR}" && wandb sweep "${SWEEP_YAML}")"
  echo "${output}" | tee -a "${MASTER_LOG}" >/dev/null
  local sweep_target
  sweep_target="$(echo "${output}" | awk '/Run sweep agent/ {print $NF}' | tail -n 1)"
  if [[ -z "${sweep_target}" ]]; then
    echo "Failed to parse sweep target from wandb output."
    exit 1
  fi
  echo "${sweep_target}"
}

start() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists."
    echo "Attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  local sweep_target="${SWEEP_TARGET:-}"
  if [[ -z "${sweep_target}" ]]; then
    echo "Creating W&B sweep from ${SWEEP_YAML}..."
    sweep_target="$(create_sweep)"
  fi
  echo "Using sweep target: ${sweep_target}"
  echo "Sweep target: ${sweep_target}" >> "${MASTER_LOG}"

  local launched=0
  local i
  for ((i=0; i<${#GPU_IDS[@]}; i++)); do
    local gpu="${GPU_IDS[$i]}"
    local slot
    for ((slot=0; slot<AGENTS_PER_GPU; slot++)); do
      local suffix=""
      local window_name="gpu${gpu}"
      local log_file="${LOG_DIR}/${SESSION}_gpu${gpu}.log"
      if (( AGENTS_PER_GPU > 1 )); then
        suffix="_slot$((slot + 1))"
        window_name="gpu${gpu}-${slot}"
        log_file="${LOG_DIR}/${SESSION}_gpu${gpu}${suffix}.log"
      fi

      if (( launched == 0 )); then
        launch_agent_window "__first__" "${gpu}" "${log_file}" "${sweep_target}" "${window_name}"
      else
        launch_agent_window "${window_name}" "${gpu}" "${log_file}" "${sweep_target}"
      fi
      launched=$((launched + 1))
    done
  done

  echo "Started tmux session '${SESSION}' with ${launched} W&B agents."
  echo "Attach: tmux attach -t ${SESSION}"
}

status() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' is running."
    tmux list-windows -t "${SESSION}" -F "window=#{window_name} active=#{window_active}"
  else
    echo "Session '${SESSION}' is not running."
  fi
}

attach() {
  tmux attach -t "${SESSION}"
}

logs() {
  local gpu="${LOG_GPU:-${GPU_IDS[0]}}"
  local slot="${LOG_SLOT:-1}"
  local log_file="${LOG_DIR}/${SESSION}_gpu${gpu}.log"
  if (( AGENTS_PER_GPU > 1 )); then
    log_file="${LOG_DIR}/${SESSION}_gpu${gpu}_slot${slot}.log"
  fi
  if [[ ! -f "${log_file}" ]]; then
    echo "Log file not found yet: ${log_file}"
    exit 1
  fi
  tail -f "${log_file}"
}

stop() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
    echo "Stopped session '${SESSION}'."
  else
    echo "Session '${SESSION}' is not running."
  fi
}

case "${ACTION}" in
  start) start ;;
  status) status ;;
  attach) attach ;;
  logs) logs ;;
  stop) stop ;;
  *)
    echo "Usage: $0 {start|status|attach|logs|stop}"
    echo "Optional env vars:"
    echo "  SESSION_NAME=<tmux session name>"
    echo "  SWEEP_YAML=<path/to/sweep.yaml>"
    echo "  SWEEP_TARGET=<entity/project/sweep_id>  # skip sweep creation"
    echo "  GPU_IDS=0,1,2,3,4,5"
    echo "  AGENTS_PER_GPU=1"
    echo "  LOG_GPU=1                                # for logs action"
    echo "  LOG_SLOT=2                               # for logs when AGENTS_PER_GPU>1"
    exit 1
    ;;
esac
