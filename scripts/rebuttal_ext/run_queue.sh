#!/usr/bin/env bash
# Run several rebuttal experiments from one queue with a fixed number of parallel workers.
#
#   scripts/rebuttal_ext/run_queue.sh <gpu_id> <workers> <experiment> [<experiment> ...]
#
# One job is one (experiment, dataset) pair. Jobs start in the given experiment order and each job runs its
# five horizons in sequence. A failed job is reported and the queue continues. Each job logs to
# rebuttal_ext_runs/<experiment>/logs/, and the queue log is rebuttal_ext_runs/queue.log.
set -uo pipefail

gpu="${1:?usage: run_queue.sh <gpu_id> <workers> <experiment> ...}"
workers="${2:?usage: run_queue.sh <gpu_id> <workers> <experiment> ...}"
shift 2
(( $# > 0 )) || { echo "usage: run_queue.sh <gpu_id> <workers> <experiment> ..." >&2; exit 1; }
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
root="$repo_root/rebuttal_ext_runs"
export REIN_EXT_GPU="$gpu"
export REIN_EXT_PYTHON="${PYTHON_BIN:-$HOME/miniconda3/envs/myenv01/bin/python}"

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT
for experiment in "$@"; do
  work="$root/$experiment"
  [[ -f "$work/run.py" ]] || { echo "[ERROR] $work is not set up" >&2; exit 1; }
  [[ ! -e "$work/results/test_results.csv" ]] || { echo "[ERROR] $work already has results" >&2; exit 1; }
  while IFS= read -r script; do
    printf '%s\n%s\n' "$work" "$script" >> "$jobs_file"
  done < <(find "$work/scripts" -name 'run_*.sh' | sort)
done

run_job() {
  local work="$1" script="$2" name status
  name="$(basename -- "$work")_$(basename -- "$script" .sh)"
  mkdir -p "$work/logs" "$work/.mpl"
  echo "$(date '+%F %T') START $name"
  (
    set -o pipefail
    cd "$work" || exit 1
    PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$work/.mpl" PYTHON_BIN="$REIN_EXT_PYTHON" GPU_ID="$REIN_EXT_GPU" \
      bash "$script" 2>&1 | tee "$work/logs/$name.log" >/dev/null
  )
  status=$?
  if (( status == 0 )); then
    echo "$(date '+%F %T') DONE  $name"
  else
    echo "$(date '+%F %T') FAIL  $name status=$status"
  fi
  return 0
}
export -f run_job

echo "$(date '+%F %T') QUEUE start gpu=$gpu workers=$workers jobs=$(( $(wc -l < "$jobs_file") / 2 ))"
xargs -d '\n' -n 2 -P "$workers" bash -c 'run_job "$0" "$1"' < "$jobs_file"
echo "$(date '+%F %T') QUEUE finished"
