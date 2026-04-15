#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
SESSION="${SESSION_NAME:-model_zoo_main}"
CONFIG="${CONFIG_PATH:-configs/cifar10_default.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install tmux first."
  exit 1
fi

mkdir -p "${LOG_DIR}"

start() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already exists."
    echo "Attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  local cmd
  cmd="cd \"${ROOT_DIR}\" && PYTHONPATH=./graph_metanetworks-main python3 -m model_zoo --config \"${CONFIG}\" 2>&1 | tee -a \"${LOG_FILE}\""
  tmux new-session -d -s "${SESSION}" "${cmd}"

  echo "Started session '${SESSION}'."
  echo "Config: ${CONFIG}"
  echo "Logs:   ${LOG_FILE}"
  echo "Attach: tmux attach -t ${SESSION}"
}

status() {
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' is running."
    echo "Use 'tmux attach -t ${SESSION}' to view live output."
    tmux list-panes -t "${SESSION}" -F "pane_active=#{pane_active} pid=#{pane_pid} cmd=#{pane_current_command}"
  else
    echo "Session '${SESSION}' is not running."
  fi
}

logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "Log file not found yet: ${LOG_FILE}"
    exit 1
  fi
  tail -f "${LOG_FILE}"
}

attach() {
  tmux attach -t "${SESSION}"
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
  logs) logs ;;
  attach) attach ;;
  stop) stop ;;
  *)
    echo "Usage: $0 {start|status|logs|attach|stop}"
    echo "Optional env vars:"
    echo "  SESSION_NAME=<tmux-session-name>"
    echo "  CONFIG_PATH=<path-to-config-yaml>"
    exit 1
    ;;
esac
