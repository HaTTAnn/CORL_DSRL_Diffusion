#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: launch_dsrl_baseline.sh TASK GPU SEED [ENV_LABEL]}"
GPU="${2:-0}"
SEED="${3:-0}"
ENV_LABEL="${4:-${ENV_LABEL:-uv}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
DEVICE="${DEVICE:-cuda:0}"
RUN_START_TS="$(date '+%F %T')"
SECONDS=0

format_seconds() {
  local s="$1"
  printf "%02d:%02d:%02d" "$((s / 3600))" "$(((s % 3600) / 60))" "$((s % 60))"
}

on_exit() {
  local code="$?"
  local elapsed="$SECONDS"
  echo ""
  echo "[timer] started:  $RUN_START_TS"
  echo "[timer] finished: $(date '+%F %T')"
  echo "[timer] elapsed:  $(format_seconds "$elapsed")"
  echo "[timer] exit code: $code"
}
trap on_exit EXIT

case "$TASK" in
  can|square) ;;
  *) echo "unknown task: $TASK (expected can|square)" >&2; exit 2 ;;
esac

WANDB_PROJECT="${WANDB_PROJECT:-DSRL_robomimic}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-stepchunk_baseline}"
RUN_TAG="${RUN_TAG:-$(date +%Y_%m_%d_%H_%M_%S)}"
BASELINE_USE_TARGET_ENV_STOP="${BASELINE_USE_TARGET_ENV_STOP:-1}"
BASELINE_SAVE_CHECKPOINT="${BASELINE_SAVE_CHECKPOINT:-false}"

case "$TASK" in
  can)
    TARGET_ENV_TIMESTEPS="${TARGET_ENV_TIMESTEPS:-1000000}"
    OFFICIAL_UTD="${BASELINE_CAN_UTD:-20}"
    OFFICIAL_INIT_ROLLOUT_STEPS="${BASELINE_CAN_INIT_ROLLOUT_STEPS:-1501}"
    ;;
  square)
    TARGET_ENV_TIMESTEPS="${TARGET_ENV_TIMESTEPS:-2000000}"
    OFFICIAL_UTD="${BASELINE_SQUARE_UTD:-20}"
    OFFICIAL_INIT_ROLLOUT_STEPS="${BASELINE_SQUARE_INIT_ROLLOUT_STEPS:-2001}"
    ;;
esac

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Using active virtualenv: $VIRTUAL_ENV"
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
  echo "Activated uv virtualenv: $PROJECT_ROOT/.venv"
else
  echo "No active virtualenv and no $PROJECT_ROOT/.venv found." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=300
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/dppo:$PROJECT_ROOT/stable-baselines3:${PYTHONPATH:-}"

GROUP="${WANDB_GROUP_PREFIX}_${TASK}_dsrl_baseline_${RUN_TAG}"
RUN_NAME="${TASK}_dsrl_baseline_seed${SEED}_${ENV_LABEL}_${RUN_TAG}"

EXTRA_OVERRIDES=()
if [[ -n "${N_EVAL_ENVS:-}" ]]; then
  EXTRA_OVERRIDES+=(++env.n_eval_envs="$N_EVAL_ENVS")
fi
if [[ -n "${NUM_EVALS:-}" ]]; then
  EXTRA_OVERRIDES+=(++num_evals="$NUM_EVALS")
fi
if [[ -n "${EVAL_VIDEO:-}" ]]; then
  EXTRA_OVERRIDES+=(++env.save_video="$EVAL_VIDEO")
fi
if [[ "$BASELINE_USE_TARGET_ENV_STOP" == "1" && "$TARGET_ENV_TIMESTEPS" != "0" ]]; then
  EXTRA_OVERRIDES+=(++train.target_env_timesteps="$TARGET_ENV_TIMESTEPS")
fi

cd "$PROJECT_ROOT"

echo "project root: $PROJECT_ROOT"
echo "python: $(which python)"
echo "run=${RUN_NAME} gpu=${GPU} task=${TASK} seed=${SEED}"
echo "baseline: fixed_denoising_steps=8 fixed_chunk_size=4 enable_three_head=false"
echo "official train defaults: utd=${OFFICIAL_UTD} init_rollout_steps=${OFFICIAL_INIT_ROLLOUT_STEPS}"
echo "target_env_timesteps=${TARGET_ENV_TIMESTEPS} target_stop=${BASELINE_USE_TARGET_ENV_STOP} save_checkpoint=${BASELINE_SAVE_CHECKPOINT}"

python train_dsrl.py --config-name "dsrl_${TASK}.yaml" \
  seed="$SEED" \
  device="$DEVICE" \
  name="$RUN_NAME" \
  wandb.project="$WANDB_PROJECT" \
  wandb.group="$GROUP" \
  log_dir="./logs_stepchunk_baseline" \
  ++save_checkpoint="$BASELINE_SAVE_CHECKPOINT" \
  ++save_replay_buffer=false \
  ++train.total_timesteps=20000000 \
  ++train.utd="$OFFICIAL_UTD" \
  ++train.init_rollout_steps="$OFFICIAL_INIT_ROLLOUT_STEPS" \
  ++train.enable_three_head=false \
  ++train.min_denoising_steps=3 \
  ++train.max_denoising_steps=8 \
  ++train.min_chunk_size=1 \
  ++train.max_chunk_size=4 \
  ++train.fixed_denoising_steps=8 \
  ++train.fixed_chunk_size=4 \
  ++train.step_cost=0.0 \
  ++train.target_nfe=2.0 \
  ++train.actor_compute_lambda=0.0 \
  ++train.actor_compute_lambda_warmup=0.0 \
  ++train.budget_penalty_location=none \
  ++train.nfe_budget_mode=episode_band \
  ++train.nfe_budget_penalty_scale=0.0 \
  ++train.nfe_saving_weight=0.0 \
  ++train.enable_chunk_elasticity=false \
  ++train.stochastic_rounding=false \
  ++train.schedule_entropy_weight=0.0 \
  ++train.difficulty_prior_scale=0.0 \
  ++train.difficulty_weight=0.0 \
  ++train.difficulty_loss_mode=none \
  "${EXTRA_OVERRIDES[@]}"
