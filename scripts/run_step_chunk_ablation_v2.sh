#!/usr/bin/env bash
set -euo pipefail

# Step/chunk ablation for natural step elasticity:
#   weak_chunk:    steps 3..8, chunk elastic but weak (4 -> 3.75)
#   both_explore:  steps 3..8, chunk fully elastic (4 -> 3)
#   step_only:     steps 3..8, chunk fixed at 4
#   dsrl_baseline: fixed DSRL baseline in this repo (8 denoise steps / chunk 4)

PLAN="${1:-12}"
GPUS_CSV="${2:-${GPUS:-0,1,2,3,4,5,6,7}}"
ENV_LABEL="${3:-${ENV_LABEL:-uv}}"
SIGNAL_VARIANT="${SIGNAL_VARIANT:-advz_strong}"
BASELINE_USE_TARGET_ENV_STOP="${BASELINE_USE_TARGET_ENV_STOP:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
BASE_RUN_TAG="${RUN_TAG:-stepchunk_v2}"
DRY_RUN="${DRY_RUN:-0}"
WANDB_GROUP_PREFIX_BASE="${WANDB_GROUP_PREFIX:-stepchunk_ablation_v2_p${PLAN}}"
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
    ARMS=(weak_chunk both_explore step_only dsrl_baseline)
    EXPECTED=16
    ;;
  12)
    TASKS=(can square)
    SEEDS=(0 1)
    ARMS=(weak_chunk both_explore step_only)
    EXPECTED=12
    ;;
  square8)
    TASKS=(square)
    SEEDS=(0 1)
    ARMS=(weak_chunk both_explore step_only dsrl_baseline)
    EXPECTED=8
    ;;
  square6)
    TASKS=(square)
    SEEDS=(0 1)
    ARMS=(weak_chunk both_explore step_only)
    EXPECTED=6
    ;;
  baseline4)
    TASKS=(can square)
    SEEDS=(0 1)
    ARMS=(dsrl_baseline)
    EXPECTED=4
    ;;
  smoke)
    TASKS=(square)
    SEEDS=(0)
    ARMS=(weak_chunk)
    EXPECTED=1
    ;;
  smoke_baseline)
    TASKS=(square)
    SEEDS=(0)
    ARMS=(dsrl_baseline)
    EXPECTED=1
    ;;
  *)
    echo "usage: $0 [16|12|square8|square6|baseline4|smoke|smoke_baseline] [gpu_csv] [env_label]" >&2
    echo "examples:" >&2
    echo "  DRY_RUN=1 SIGNAL_VARIANT=advz_strong $0 square8 0,1,3 uv" >&2
    echo "  RUN_TAG=stepchunk_v2_advz SIGNAL_VARIANT=advz_strong $0 16 0,1,2,3,4,5,6,7 uv" >&2
    echo "  RUN_TAG=stepchunk_v2_advz $0 baseline4 0,1,3,7 uv" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"

JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("$arm $task $seed")
    done
  done
done

if [[ "${#JOBS[@]}" -ne "$EXPECTED" ]]; then
  echo "internal error: plan $PLAN produced ${#JOBS[@]} jobs" >&2
  exit 3
fi

echo "step/chunk ablation v2 plan=$PLAN signal=${SIGNAL_VARIANT} gpus=${GPUS_CSV} env=${ENV_LABEL} tag=${BASE_RUN_TAG}"
echo "project root: $PROJECT_ROOT"
echo "logs: $LOG_DIR"
echo "wandb project: ${WANDB_PROJECT:-DSRL_robomimic}"
echo "wandb group prefix base: $WANDB_GROUP_PREFIX_BASE"
echo "arms: ${ARMS[*]}"
echo "tasks: ${TASKS[*]}"
echo "seeds: ${SEEDS[*]}"
echo

GPU_COUNTS=()
SESSIONS=()
for i in "${!JOBS[@]}"; do
  read -r arm task seed <<< "${JOBS[$i]}"
  gpu_idx=$((i % ${#GPUS[@]}))
  gpu="${GPUS[$gpu_idx]}"
  GPU_COUNTS[$gpu_idx]=$(( ${GPU_COUNTS[$gpu_idx]:-0} + 1 ))
  signal_label="$SIGNAL_VARIANT"
  if [[ "$arm" == "dsrl_baseline" ]]; then
    signal_label="official"
  fi
  session="dsrl_stepchunk_v2_j$(printf '%02d' "$i")_g${gpu}_${arm}_${task}_s${seed}_${signal_label}_${BASE_RUN_TAG}"
  log="$LOG_DIR/${BASE_RUN_TAG}_j$(printf '%02d' "$i")_${arm}_${task}_${signal_label}_seed${seed}_gpu${gpu}.log"
  SESSIONS+=("$session")
  printf '%02d gpu=%s arm=%s task=%s seed=%s signal=%s session=%s log=%s\n' "$i" "$gpu" "$arm" "$task" "$seed" "$signal_label" "$session" "$log"
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
  read -r arm task seed <<< "${JOBS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  session="${SESSIONS[$i]}"
  signal_label="$SIGNAL_VARIANT"
  if [[ "$arm" == "dsrl_baseline" ]]; then
    signal_label="official"
  fi
  log="$LOG_DIR/${BASE_RUN_TAG}_j$(printf '%02d' "$i")_${arm}_${task}_${signal_label}_seed${seed}_gpu${gpu}.log"

  case "$arm" in
    weak_chunk)
      easy_steps=3
      hard_steps=8
      easy_chunk=4
      hard_chunk=3.75
      chunk_elastic=true
      stochastic_rounding=true
      nfe_upper_can=1.95
      nfe_upper_square=1.95
      ;;
    both_explore)
      easy_steps=3
      hard_steps=8
      easy_chunk=4
      hard_chunk=3
      chunk_elastic=true
      stochastic_rounding=true
      nfe_upper_can=1.95
      nfe_upper_square=1.95
      ;;
    step_only)
      easy_steps=3
      hard_steps=8
      easy_chunk=4
      hard_chunk=4
      chunk_elastic=false
      stochastic_rounding=true
      nfe_upper_can=1.95
      nfe_upper_square=1.95
      ;;
    dsrl_baseline)
      easy_steps=8
      hard_steps=8
      easy_chunk=4
      hard_chunk=4
      chunk_elastic=false
      stochastic_rounding=false
      nfe_upper_can=2.00
      nfe_upper_square=2.00
      ;;
    *)
      echo "unknown arm: $arm" >&2
      exit 6
      ;;
  esac

  if [[ "$task" == "can" ]]; then
    target_nfe="${TARGET_NFE_CAN:-1.75}"
    nfe_lower="${NFE_TARGET_LOWER_CAN:-1.45}"
    nfe_upper="${NFE_TARGET_UPPER_CAN:-$nfe_upper_can}"
    target_env_steps="${TARGET_ENV_TIMESTEPS_CAN:-1000000}"
  else
    target_nfe="${TARGET_NFE_SQUARE:-1.90}"
    nfe_lower="${NFE_TARGET_LOWER_SQUARE:-1.65}"
    nfe_upper="${NFE_TARGET_UPPER_SQUARE:-$nfe_upper_square}"
    target_env_steps="${TARGET_ENV_TIMESTEPS_SQUARE:-2000000}"
  fi

  run_tag="${BASE_RUN_TAG}_${arm}"
  group_prefix="${WANDB_GROUP_PREFIX_BASE}_${arm}"
  label="${ENV_LABEL}_${arm}"

  if [[ "$arm" == "dsrl_baseline" ]]; then
    run_tag="${BASE_RUN_TAG}_${arm}"
    group_prefix="${WANDB_GROUP_PREFIX_BASE}_${arm}"
    baseline_target_env_steps="$target_env_steps"
    if [[ "$task" == "can" ]]; then
      baseline_target_env_steps="${BASELINE_TARGET_ENV_TIMESTEPS_CAN:-$target_env_steps}"
    else
      baseline_target_env_steps="${BASELINE_TARGET_ENV_TIMESTEPS_SQUARE:-$target_env_steps}"
    fi

    cmd="set -euo pipefail;"
    cmd+=" cd '$PROJECT_ROOT';"
    cmd+=" export RUN_TAG='$run_tag';"
    cmd+=" export WANDB_GROUP_PREFIX='$group_prefix';"
    cmd+=" export WANDB_MODE='${WANDB_MODE:-online}';"
    cmd+=" export WANDB_PROJECT='${WANDB_PROJECT:-DSRL_robomimic}';"
    cmd+=" export TARGET_ENV_TIMESTEPS='$baseline_target_env_steps';"
    cmd+=" export BASELINE_USE_TARGET_ENV_STOP='$BASELINE_USE_TARGET_ENV_STOP';"
    cmd+=" export OMP_NUM_THREADS='${OMP_NUM_THREADS:-1}';"
    cmd+=" export MKL_NUM_THREADS='${MKL_NUM_THREADS:-1}';"
    cmd+=" export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS:-1}';"
    cmd+=" export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS:-1}';"
    append_export_if_set BASELINE_SAVE_CHECKPOINT
    append_export_if_set BASELINE_CAN_UTD
    append_export_if_set BASELINE_CAN_INIT_ROLLOUT_STEPS
    append_export_if_set BASELINE_SQUARE_UTD
    append_export_if_set BASELINE_SQUARE_INIT_ROLLOUT_STEPS
    append_export_if_set WANDB_API_KEY
    append_export_if_set WANDB_ENTITY
    append_export_if_set WANDB_BASE_URL
    append_export_if_set WANDB_DIR
    append_export_if_set WANDB_CACHE_DIR
    append_export_if_set WANDB_CONFIG_DIR
    append_export_if_set N_EVAL_ENVS
    append_export_if_set NUM_EVALS
    append_export_if_set EVAL_VIDEO
    cmd+=" echo '=== start job index=${i} arm=${arm} task=${task} seed=${seed} signal=official gpu=${gpu} ===';"
    cmd+=" echo 'baseline config: fixed 8/4 in current repo, target_env_timesteps=${baseline_target_env_steps}, target_stop=${BASELINE_USE_TARGET_ENV_STOP}';"
    cmd+=" bash scripts/launch_dsrl_baseline.sh '${task}' '${gpu}' '${seed}' '${ENV_LABEL}_${arm}' 2>&1 | tee '${log}';"
    cmd+=" echo '=== finished job index=${i} arm=${arm} task=${task} seed=${seed} ===';"
  else
    cmd="set -euo pipefail;"
    cmd+=" cd '$PROJECT_ROOT';"
    cmd+=" export RUN_TAG='$run_tag';"
    cmd+=" export WANDB_GROUP_PREFIX='$group_prefix';"
    cmd+=" export WANDB_MODE='${WANDB_MODE:-online}';"
    cmd+=" export WANDB_PROJECT='${WANDB_PROJECT:-DSRL_robomimic}';"
    cmd+=" export DIFFICULTY_EASY_STEPS_TARGET='$easy_steps';"
    cmd+=" export DIFFICULTY_HARD_STEPS_TARGET='$hard_steps';"
    cmd+=" export DIFFICULTY_EASY_CHUNK_TARGET='$easy_chunk';"
    cmd+=" export DIFFICULTY_HARD_CHUNK_TARGET='$hard_chunk';"
    cmd+=" export ENABLE_CHUNK_ELASTICITY='$chunk_elastic';"
    cmd+=" export STOCHASTIC_ROUNDING='$stochastic_rounding';"
    cmd+=" export TARGET_NFE='$target_nfe';"
    cmd+=" export NFE_TARGET_LOWER='$nfe_lower';"
    cmd+=" export NFE_TARGET_UPPER='$nfe_upper';"
    cmd+=" export TARGET_ENV_TIMESTEPS='$target_env_steps';"
    cmd+=" export RANGE_OPEN_RATE='${RANGE_OPEN_RATE_OVERRIDE:-0.06}';"
    cmd+=" export RANGE_CLOSE_RATE='0.0';"
    cmd+=" export DIFFICULTY_SUCCESS_OPEN_RATE='${DIFFICULTY_SUCCESS_OPEN_RATE_OVERRIDE:-0.06}';"
    cmd+=" export DIFFICULTY_SUCCESS_CLOSE_RATE='0.0';"
    cmd+=" export OMP_NUM_THREADS='${OMP_NUM_THREADS:-1}';"
    cmd+=" export MKL_NUM_THREADS='${MKL_NUM_THREADS:-1}';"
    cmd+=" export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS:-1}';"
    cmd+=" export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS:-1}';"
    append_export_if_set WANDB_API_KEY
    append_export_if_set WANDB_ENTITY
    append_export_if_set WANDB_BASE_URL
    append_export_if_set WANDB_DIR
    append_export_if_set WANDB_CACHE_DIR
    append_export_if_set WANDB_CONFIG_DIR
    append_export_if_set N_EVAL_ENVS
    append_export_if_set NUM_EVALS
    append_export_if_set EVAL_VIDEO
    cmd+=" echo '=== start job index=${i} arm=${arm} task=${task} seed=${seed} signal=${SIGNAL_VARIANT} gpu=${gpu} ===';"
    cmd+=" echo 'arm config: steps=${easy_steps}->${hard_steps} chunk=${easy_chunk}->${hard_chunk} chunk_elastic=${chunk_elastic} stochastic=${stochastic_rounding} nfe=[${nfe_lower},${nfe_upper}] target=${target_nfe}';"
    cmd+=" bash scripts/launch_step_difficulty_h20.sh '${task}' '${gpu}' '${seed}' '${label}' '${SIGNAL_VARIANT}' 2>&1 | tee '${log}';"
    cmd+=" echo '=== finished job index=${i} arm=${arm} task=${task} seed=${seed} ===';"
  fi

  tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$cmd")"
  echo "started $session on gpu $gpu"
done

echo
echo "started ${#JOBS[@]} tmux sessions"
echo "monitor:"
echo "  tmux ls | grep dsrl_stepchunk_v2"
echo "attach one:"
echo "  tmux attach -t ${SESSIONS[0]}"
echo "logs:"
echo "  tail -f $LOG_DIR/${BASE_RUN_TAG}_*.log"
