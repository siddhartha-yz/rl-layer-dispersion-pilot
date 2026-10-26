#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
case "${1:-}" in
  ""|--rerun) ;;
  *) echo "Usage: ./reproduce.sh [--rerun]" >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv and install dependencies as described in README.md first." >&2
  exit 1
fi
if [[ "${1:-}" != "--rerun" && -f runs/summary.json && ! -f runs/all/seed_11/adapter.pt ]]; then
  echo "Restore the checkpoint Release asset first, or use ./reproduce.sh --rerun to retrain." >&2
  exit 1
fi
if [[ "${1:-}" == "--rerun" && -d runs ]]; then
  task_run_stamp=$(date +%Y%m%d-%H%M%S)
  mkdir -p archive/reproductions
  mv runs "archive/reproductions/runs-${task_run_stamp}"
fi
.venv/bin/python download_model.py
.venv/bin/python audit_data.py
.venv/bin/python verify_mechanism.py
.venv/bin/python -u rl_layer_pilot.py --mode sweep --steps 200
.venv/bin/python -u rl_layer_pilot.py --mode pilot --steps 300
.venv/bin/python verify_results.py
.venv/bin/python diagnose_drop_scale.py
.venv/bin/python analyze_pilot.py
