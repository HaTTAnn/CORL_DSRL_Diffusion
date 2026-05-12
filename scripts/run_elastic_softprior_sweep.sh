#!/usr/bin/env bash
set -euo pipefail

PLAN="${1:-8}"
GPUS_CSV="${2:-${GPUS:-6,7}}"
CONDA_ENV="${3:-${CONDA_ENV:-dsrl_diffusion}}"
REPO="/root/storage/CODE/txy/dsrl_fv"
RUN_TAG="${RUN_TAG:-$(date +%Y_%m_%d_%H_%M_%S)}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "no GPU provided" >&2
  exit 2
fi

case "$PLAN" in
  8)
    TASKS=(can square)
    SEEDS=(0 1)
    VARIANTS=(soft_h2_bal soft_h3_safe)
    ;;
  16)
    TASKS=(can square)
    SEEDS=(0 1)
    VARIANTS=(soft_h2_bal soft_h2_res soft_h3_safe soft_h2_edge)
    ;;
  *)
    echo "usage: $0 [8|16] [gpu_csv] [conda_env]" >&2
    echo "example: DRY_RUN=1 $0 16 6,7 dsrl_diffusion" >&2
    exit 2
    ;;
esac

mkdir -p "$REPO/run_logs"

JOBS=()
for variant in "${VARIANTS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("$task $seed $variant")
    done
  done
done

expected="$PLAN"
if [[ "${#JOBS[@]}" -ne "$expected" ]]; then
  echo "internal error: plan $PLAN produced ${#JOBS[@]} jobs" >&2
  exit 3
fi

echo "elastic soft-prior sweep plan=$PLAN gpus=${GPUS_CSV} conda=${CONDA_ENV} tag=${RUN_TAG}"
echo "tasks: ${TASKS[*]}"
echo "seeds: ${SEEDS[*]}"
echo "variants: ${VARIANTS[*]}"
echo
for i in "${!JOBS[@]}"; do
  read -r task seed variant <<< "${JOBS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  printf '%02d gpu=%s task=%s seed=%s variant=%s\n' "$i" "$gpu" "$task" "$seed" "$variant"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "dry run only; no tmux sessions started"
  exit 0
fi

for worker_idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker_idx]}"
  session="dsrl_diff_elastic_p${PLAN}_g${gpu}_${RUN_TAG}"
  cmd="set -euo pipefail; cd '$REPO'; export RUN_TAG='$RUN_TAG'; export CONDA_ENV='$CONDA_ENV'; export WANDB_GROUP_PREFIX='elastic_softprior_p${PLAN}';"
  assigned=0
  for i in "${!JOBS[@]}"; do
    if [[ $((i % ${#GPUS[@]})) -ne "$worker_idx" ]]; then
      continue
    fi
    read -r task seed variant <<< "${JOBS[$i]}"
    log="$REPO/run_logs/${RUN_TAG}_${task}_elastic_${variant}_seed${seed}_gpu${gpu}.log"
    cmd+=" echo '=== start job index=${i} task=${task} seed=${seed} variant=${variant} gpu=${gpu} ===';"
    cmd+=" bash scripts/launch_elastic_softprior.sh '${task}' '${gpu}' '${seed}' '${CONDA_ENV}' '${variant}' 2>&1 | tee '${log}';"
    cmd+=" echo '=== finished job index=${i} task=${task} seed=${seed} variant=${variant} gpu=${gpu} ===';"
    assigned=$((assigned + 1))
  done
  if [[ "$assigned" -eq 0 ]]; then
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 4
  fi
  tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$cmd")"
  echo "started $session with $assigned queued jobs on gpu $gpu"
done

echo
echo "monitor: tmux ls | grep dsrl_diff_elastic"
echo "logs: $REPO/run_logs/${RUN_TAG}_*.log"

