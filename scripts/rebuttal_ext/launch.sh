#!/usr/bin/env bash
# Start one rebuttal experiment: one tmux session per dataset, running in the experiment's own directory.
#
#   scripts/rebuttal_ext/launch.sh <experiment> [gpu_id]
#
# Results go to rebuttal_ext_runs/<experiment>/results/test_results.csv and
# rebuttal_ext_runs/<experiment>/checkpoints/. Run setup_experiments.py first.
set -euo pipefail

experiment="${1:?usage: launch.sh <experiment> [gpu_id]}"
gpu="${2:-0}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$repo_root/rebuttal_ext_runs/$experiment"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/myenv01/bin/python}"

[[ -f "$work/run.py" ]] || { echo "[ERROR] $work is not set up; run setup_experiments.py" >&2; exit 1; }
if [[ -f "$work/results/test_results.csv" ]]; then
  echo "[ERROR] $work already has results; move them before launching again" >&2
  exit 1
fi

mapfile -t scripts < <(find "$work/scripts" -name 'run_*.sh' | sort)
sessions=()
for script in "${scripts[@]}"; do
  stem="$(basename -- "$script" .sh)"
  session="ext_${experiment}_${stem#run_}"
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "[ERROR] tmux session $session already exists" >&2
    exit 1
  fi
  sessions+=("$session")
done

mkdir -p "$work/logs" "$work/.mpl"
for i in "${!scripts[@]}"; do
  session="${sessions[$i]}"
  tmux new-session -d -s "$session" \
    "bash -lc \"cd '$work' && PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR='$work/.mpl' PYTHON_BIN='$python_bin' GPU_ID='$gpu' bash '${scripts[$i]}' 2>&1 | tee '$work/logs/$session.log'; echo '[$session] finished'; exec bash\""
  echo "started $session"
done
