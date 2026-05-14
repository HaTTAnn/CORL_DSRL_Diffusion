#!/usr/bin/env bash
set -euo pipefail

PLAN="${1:-16}"
GPUS_CSV="${2:-${GPUS:-0,1,2,3,4,5,6,7}}"
ENV_LABEL="${3:-${ENV_LABEL:-uv}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
RUN_TAG="${RUN_TAG:-$(date +%Y_%m_%d_%H_%M_%S)}"
DRY_RUN="${DRY_RUN:-0}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-step_difficulty_h20_p${PLAN}}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/run_logs}"

append_export_if_set() {
  local name="$1"
  local value="${!name:-}"
  if [[ -n "$value" ]]; then
    cmd+=" export ${name}=$(printf '%q' "$value");"
  fi
}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "no GPU provided" >&2
  exit 2
fi

case "$PLAN" in
  16)
    TASKS=(can square)
    SEEDS=(0 1)
    VARIANTS=(rank_current advz_current advz_strong qstdz_strong)
    EXPECTED=16
    ;;
  smoke)
    TASKS=(can)
    SEEDS=(0)
    VARIANTS=(rank_current)
    EXPECTED=1
    ;;
  *)
    echo "usage: $0 [16|smoke] [gpu_csv] [env_label]" >&2
    echo "example preview: DRY_RUN=1 $0 16 0,1,2,3,4,5,6,7 uv" >&2
    echo "example launch:  RUN_TAG=stepdiff_v1 $0 16 0,1,2,3,4,5,6,7 uv" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"

JOBS=()
for variant in "${VARIANTS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("$task $seed $variant")
    done
  done
done

if [[ "${#JOBS[@]}" -ne "$EXPECTED" ]]; then
  echo "internal error: plan $PLAN produced ${#JOBS[@]} jobs" >&2
  exit 3
fi

echo "step-difficulty H20 sweep plan=$PLAN gpus=${GPUS_CSV} env=${ENV_LABEL} tag=${RUN_TAG}"
echo "project root: $PROJECT_ROOT"
echo "logs: $LOG_DIR"
echo "wandb mode: ${WANDB_MODE:-online}"
echo "wandb project: ${WANDB_PROJECT:-DSRL_diffusion_FV}"
echo "wandb group prefix: $WANDB_GROUP_PREFIX"
echo "tasks: ${TASKS[*]}"
echo "seeds: ${SEEDS[*]}"
echo "variants: ${VARIANTS[*]}"
echo

GPU_COUNTS=()
SESSIONS=()
for i in "${!JOBS[@]}"; do
  read -r task seed variant <<< "${JOBS[$i]}"
  gpu_idx=$((i % ${#GPUS[@]}))
  gpu="${GPUS[$gpu_idx]}"
  GPU_COUNTS[$gpu_idx]=$(( ${GPU_COUNTS[$gpu_idx]:-0} + 1 ))
  session="dsrl_stepdiff_p${PLAN}_j$(printf '%02d' "$i")_g${gpu}_${task}_s${seed}_${variant}_${RUN_TAG}"
  log="$LOG_DIR/${RUN_TAG}_j$(printf '%02d' "$i")_${task}_elastic_${variant}_seed${seed}_gpu${gpu}.log"
  SESSIONS+=("$session")
  printf '%02d gpu=%s task=%s seed=%s variant=%s session=%s log=%s\n' "$i" "$gpu" "$task" "$seed" "$variant" "$session" "$log"
done

echo
for gpu_idx in "${!GPUS[@]}"; do
  echo "gpu=${GPUS[$gpu_idx]} jobs=${GPU_COUNTS[$gpu_idx]:-0}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "dry run only; no tmux sessions started"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH" >&2
  exit 4
fi

for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 5
  fi
done

for i in "${!JOBS[@]}"; do
  read -r task seed variant <<< "${JOBS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  session="${SESSIONS[$i]}"
  log="$LOG_DIR/${RUN_TAG}_j$(printf '%02d' "$i")_${task}_elastic_${variant}_seed${seed}_gpu${gpu}.log"

  cmd="set -euo pipefail;"
  cmd+=" cd '$PROJECT_ROOT';"
  cmd+=" export RUN_TAG='$RUN_TAG';"
  cmd+=" export WANDB_GROUP_PREFIX='$WANDB_GROUP_PREFIX';"
  cmd+=" export WANDB_MODE='${WANDB_MODE:-online}';"
  cmd+=" export WANDB_PROJECT='${WANDB_PROJECT:-DSRL_diffusion_FV}';"
  append_export_if_set WANDB_API_KEY
  append_export_if_set WANDB_ENTITY
  append_export_if_set WANDB_BASE_URL
  append_export_if_set WANDB_DIR
  append_export_if_set WANDB_CACHE_DIR
  append_export_if_set WANDB_CONFIG_DIR
  append_export_if_set TARGET_ENV_TIMESTEPS
  append_export_if_set N_EVAL_ENVS
  append_export_if_set NUM_EVALS
  append_export_if_set EVAL_VIDEO
  cmd+=" export OMP_NUM_THREADS='${OMP_NUM_THREADS:-1}';"
  cmd+=" export MKL_NUM_THREADS='${MKL_NUM_THREADS:-1}';"
  cmd+=" export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS:-1}';"
  cmd+=" export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS:-1}';"
  cmd+=" echo '=== start job index=${i} task=${task} seed=${seed} variant=${variant} gpu=${gpu} ===';"
  cmd+=" bash scripts/launch_step_difficulty_h20.sh '${task}' '${gpu}' '${seed}' '${ENV_LABEL}' '${variant}' 2>&1 | tee '${log}';"
  cmd+=" echo '=== finished job index=${i} task=${task} seed=${seed} variant=${variant} gpu=${gpu} ===';"

  tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$cmd")"
  echo "started $session on gpu $gpu"
done

echo
echo "started ${#JOBS[@]} tmux sessions"
echo "monitor:"
echo "  tmux ls | grep dsrl_stepdiff_p${PLAN}"
echo "attach one:"
echo "  tmux attach -t ${SESSIONS[0]}"
echo "logs:"
echo "  tail -f $LOG_DIR/${RUN_TAG}_*.log"

