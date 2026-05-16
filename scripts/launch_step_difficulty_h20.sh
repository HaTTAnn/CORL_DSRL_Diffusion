#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: launch_step_difficulty_h20.sh TASK GPU SEED [ENV_LABEL] [VARIANT]}"
GPU="${2:-0}"
SEED="${3:-0}"
ENV_LABEL="${4:-${ENV_LABEL:-uv}}"
VARIANT="${5:-rank_current}"

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

N_EVAL_ENVS="${N_EVAL_ENVS:-5}"
NUM_EVALS="${NUM_EVALS:-10}"
EVAL_VIDEO="${EVAL_VIDEO:-false}"
EVAL_VIDEO_FPS="${EVAL_VIDEO_FPS:-20}"
EVAL_VIDEO_MAX_FRAMES="${EVAL_VIDEO_MAX_FRAMES:-300}"
EVAL_VIDEO_FREQ="${EVAL_VIDEO_FREQ:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-DSRL_diffusion_FV}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-step_difficulty_h20}"
RUN_TAG="${RUN_TAG:-$(date +%Y_%m_%d_%H_%M_%S)}"
INIT_ROLLOUT_STEPS="${INIT_ROLLOUT_STEPS:-}"

# Stop by the same low-level environment timestep counter logged as eval/timesteps.
case "$TASK" in
  can)
    TARGET_ENV_TIMESTEPS="${TARGET_ENV_TIMESTEPS:-1000000}"
    TARGET_NFE="${TARGET_NFE:-1.75}"
    NFE_TARGET_LOWER="${NFE_TARGET_LOWER:-1.55}"
    NFE_TARGET_UPPER="${NFE_TARGET_UPPER:-2.05}"
    ;;
  square)
    TARGET_ENV_TIMESTEPS="${TARGET_ENV_TIMESTEPS:-2000000}"
    TARGET_NFE="${TARGET_NFE:-1.90}"
    NFE_TARGET_LOWER="${NFE_TARGET_LOWER:-1.75}"
    NFE_TARGET_UPPER="${NFE_TARGET_UPPER:-2.15}"
    INIT_ROLLOUT_STEPS="${INIT_ROLLOUT_STEPS:-8000}"
    ;;
esac

ACTOR_COMPUTE_LAMBDA="${ACTOR_COMPUTE_LAMBDA:-0.45}"
ACTOR_COMPUTE_LAMBDA_WARMUP="${ACTOR_COMPUTE_LAMBDA_WARMUP:-0.15}"
NFE_UNDER_WEIGHT="${NFE_UNDER_WEIGHT:-0.15}"
NFE_DEBT_LIMIT="${NFE_DEBT_LIMIT:-4.0}"
NFE_BUDGET_PENALTY_SCALE="${NFE_BUDGET_PENALTY_SCALE:-10.0}"
NFE_SAVING_WEIGHT="${NFE_SAVING_WEIGHT:-0.06}"
EPISODE_SUCCESS_THRESHOLD="${EPISODE_SUCCESS_THRESHOLD:--0.5}"
COST_GATE_MODE="${COST_GATE_MODE:-step_monotonic}"
COST_START_STEP="${COST_START_STEP:-10000}"
COST_WARMUP_STEPS="${COST_WARMUP_STEPS:-90000}"
COST_NO_ROLLBACK="${COST_NO_ROLLBACK:-1}"

RANGE_SUCCESS_THRESH_1="${RANGE_SUCCESS_THRESH_1:-0.45}"
RANGE_SUCCESS_THRESH_2="${RANGE_SUCCESS_THRESH_2:-0.65}"
RANGE_SUCCESS_EMA_BETA="${RANGE_SUCCESS_EMA_BETA:-0.8}"
RANGE_SUCCESS_NO_CLOSE="${RANGE_SUCCESS_NO_CLOSE:-1}"
RANGE_ACTUATOR_FLOOR="${RANGE_ACTUATOR_FLOOR:-0.35}"
USER_DIFFICULTY_SUCCESS_THRESH_1="${DIFFICULTY_SUCCESS_THRESH_1:-}"
USER_DIFFICULTY_SUCCESS_THRESH_2="${DIFFICULTY_SUCCESS_THRESH_2:-}"
USER_DIFFICULTY_SUCCESS_OPEN_RATE="${DIFFICULTY_SUCCESS_OPEN_RATE:-}"

DIFFICULTY_SUCCESS_THRESH_1="${DIFFICULTY_SUCCESS_THRESH_1:-0.50}"
DIFFICULTY_SUCCESS_THRESH_2="${DIFFICULTY_SUCCESS_THRESH_2:-0.70}"
DIFFICULTY_SUCCESS_OPEN_RATE="${DIFFICULTY_SUCCESS_OPEN_RATE:-0.10}"
DIFFICULTY_SUCCESS_CLOSE_RATE="${DIFFICULTY_SUCCESS_CLOSE_RATE:-0.04}"
DIFFICULTY_SUCCESS_EMA_BETA="${DIFFICULTY_SUCCESS_EMA_BETA:-0.8}"
DIFFICULTY_SUCCESS_NO_CLOSE="${DIFFICULTY_SUCCESS_NO_CLOSE:-1}"

SCHEDULE_HEADS_AFTER="${SCHEDULE_HEADS_AFTER:-0}"
SCHEDULE_WARMUP_STEPS="${SCHEDULE_WARMUP_STEPS:-60000}"
SCHEDULE_GATE_FLOOR="${SCHEDULE_GATE_FLOOR:-0.3}"
RANGE_OPEN_RATE="${RANGE_OPEN_RATE:-0.10}"
RANGE_CLOSE_RATE="${RANGE_CLOSE_RATE:-0.04}"
ENABLE_CHUNK_ELASTICITY="${ENABLE_CHUNK_ELASTICITY:-true}"
STOCHASTIC_ROUNDING="${STOCHASTIC_ROUNDING:-false}"
DIFFICULTY_PRIOR_START_STEP="${DIFFICULTY_PRIOR_START_STEP:-35000}"
DIFFICULTY_PRIOR_WARMUP_STEPS="${DIFFICULTY_PRIOR_WARMUP_STEPS:-50000}"
DIFFICULTY_START_STEP="${DIFFICULTY_START_STEP:-40000}"
DIFFICULTY_WARMUP_STEPS="${DIFFICULTY_WARMUP_STEPS:-50000}"
DIFFICULTY_MARGIN_START_STEP="${DIFFICULTY_MARGIN_START_STEP:-40000}"
DIFFICULTY_MARGIN_WARMUP_STEPS="${DIFFICULTY_MARGIN_WARMUP_STEPS:-50000}"
DIFFICULTY_MARGIN_TARGET="${DIFFICULTY_MARGIN_TARGET:-0.35}"
DIFFICULTY_HARD_STEPS_TARGET="${DIFFICULTY_HARD_STEPS_TARGET:-8}"
DIFFICULTY_EASY_STEPS_TARGET="${DIFFICULTY_EASY_STEPS_TARGET:-4}"
DIFFICULTY_HARD_CHUNK_TARGET="${DIFFICULTY_HARD_CHUNK_TARGET:-3}"
DIFFICULTY_EASY_CHUNK_TARGET="${DIFFICULTY_EASY_CHUNK_TARGET:-4}"
DIFFICULTY_PRIOR_DEADBAND="${DIFFICULTY_PRIOR_DEADBAND:-0.03}"
DIFFICULTY_PRIOR_GATE_FLOOR="${DIFFICULTY_PRIOR_GATE_FLOOR:-0.0}"

# Current-strength default: preserve old soft-prior behavior but use hard_chunk=3.
DIFFICULTY_PRIOR_SIGNAL_MODE=compute_advantage
DIFFICULTY_SIGNAL_MODE=compute_advantage
DIFFICULTY_PRIOR_SIGNAL_SCALE=1.0
DIFFICULTY_SIGNAL_SCALE=0.75
DIFFICULTY_WEIGHT=0.04
DIFFICULTY_MODE_MARGIN_WEIGHT=0.20
DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.0
if [[ "$TASK" == "can" ]]; then
  DIFFICULTY_PRIOR_SCALE=0.75
  PREPRIOR_RESIDUAL_SCALE=0.65
  SCHEDULE_RESIDUAL_SCALE=0.35
else
  DIFFICULTY_PRIOR_SCALE=0.70
  PREPRIOR_RESIDUAL_SCALE=0.80
  SCHEDULE_RESIDUAL_SCALE=0.45
fi

case "$VARIANT" in
  rank_current)
    ;;
  advz_current)
    DIFFICULTY_PRIOR_SIGNAL_MODE=compute_advantage_z
    DIFFICULTY_SIGNAL_MODE=compute_advantage_z
    DIFFICULTY_PRIOR_SIGNAL_SCALE=1.0
    DIFFICULTY_SIGNAL_SCALE=1.0
    ;;
  advz_strong|qstdz_strong)
    if [[ "$VARIANT" == "advz_strong" ]]; then
      DIFFICULTY_PRIOR_SIGNAL_MODE=compute_advantage_z
      DIFFICULTY_SIGNAL_MODE=compute_advantage_z
    else
      DIFFICULTY_PRIOR_SIGNAL_MODE=q_std_z
      DIFFICULTY_SIGNAL_MODE=q_std_z
    fi
    if [[ "$TASK" == "can" ]]; then
      DIFFICULTY_PRIOR_SCALE=0.80
      DIFFICULTY_WEIGHT=0.08
      DIFFICULTY_MODE_MARGIN_WEIGHT=0.50
      DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.20
      PREPRIOR_RESIDUAL_SCALE=0.40
      SCHEDULE_RESIDUAL_SCALE=0.25
      DIFFICULTY_PRIOR_SIGNAL_SCALE=1.15
      DIFFICULTY_SIGNAL_SCALE=1.00
    else
      DIFFICULTY_PRIOR_SCALE=0.95
      DIFFICULTY_WEIGHT=0.10
      DIFFICULTY_MODE_MARGIN_WEIGHT=0.60
      DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.30
      PREPRIOR_RESIDUAL_SCALE=0.35
      SCHEDULE_RESIDUAL_SCALE=0.20
      DIFFICULTY_PRIOR_SIGNAL_SCALE=1.25
      DIFFICULTY_SIGNAL_SCALE=1.00
      DIFFICULTY_SUCCESS_THRESH_1="${USER_DIFFICULTY_SUCCESS_THRESH_1:-0.45}"
      DIFFICULTY_SUCCESS_THRESH_2="${USER_DIFFICULTY_SUCCESS_THRESH_2:-0.65}"
      DIFFICULTY_SUCCESS_OPEN_RATE="${USER_DIFFICULTY_SUCCESS_OPEN_RATE:-0.15}"
    fi
    ;;
  *)
    echo "unknown variant: $VARIANT (expected rank_current|advz_current|advz_strong|qstdz_strong)" >&2
    exit 2
    ;;
esac

EXTRA_OVERRIDES=()
if [[ -n "$INIT_ROLLOUT_STEPS" ]]; then
  EXTRA_OVERRIDES+=(++train.init_rollout_steps="$INIT_ROLLOUT_STEPS")
fi

cd "$PROJECT_ROOT"
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

GROUP="${WANDB_GROUP_PREFIX}_${TASK}_${VARIANT}_${RUN_TAG}"
RUN_NAME="${TASK}_elastic_${VARIANT}_seed${SEED}_${ENV_LABEL}_${RUN_TAG}"

echo "project root: $PROJECT_ROOT"
echo "python: $(which python)"
echo "run=${RUN_NAME} gpu=${GPU} task=${TASK} seed=${SEED} variant=${VARIANT}"
echo "target_env_timesteps=${TARGET_ENV_TIMESTEPS} eval_envs=${N_EVAL_ENVS} eval_rollouts=${NUM_EVALS}"
echo "difficulty: prior_mode=${DIFFICULTY_PRIOR_SIGNAL_MODE}, loss_mode=${DIFFICULTY_SIGNAL_MODE}, prior_scale=${DIFFICULTY_PRIOR_SCALE}, prior_signal_scale=${DIFFICULTY_PRIOR_SIGNAL_SCALE}, loss_signal_scale=${DIFFICULTY_SIGNAL_SCALE}"
echo "step/chunk: easy_steps=${DIFFICULTY_EASY_STEPS_TARGET}, hard_steps=${DIFFICULTY_HARD_STEPS_TARGET}, easy_chunk=${DIFFICULTY_EASY_CHUNK_TARGET}, hard_chunk=${DIFFICULTY_HARD_CHUNK_TARGET}, preprior=${PREPRIOR_RESIDUAL_SCALE}, residual=${SCHEDULE_RESIDUAL_SCALE}"
echo "loss: weight=${DIFFICULTY_WEIGHT}, mode_margin=${DIFFICULTY_MODE_MARGIN_WEIGHT}, quantile=${DIFFICULTY_QUANTILE_HINGE_WEIGHT}"
echo "cost: band=[${NFE_TARGET_LOWER},${NFE_TARGET_UPPER}], target=${TARGET_NFE}, lambda=${ACTOR_COMPUTE_LAMBDA_WARMUP}->${ACTOR_COMPUTE_LAMBDA}"

python train_dsrl.py --config-name "dsrl_${TASK}.yaml" \
  seed="$SEED" \
  device="$DEVICE" \
  name="$RUN_NAME" \
  wandb.project="$WANDB_PROJECT" \
  wandb.group="$GROUP" \
  ++env.n_eval_envs="$N_EVAL_ENVS" \
  ++env.save_video="$EVAL_VIDEO" \
  ++env.eval_video_fps="$EVAL_VIDEO_FPS" \
  ++env.eval_video_max_frames="$EVAL_VIDEO_MAX_FRAMES" \
  ++env.eval_video_freq="$EVAL_VIDEO_FREQ" \
  ++num_evals="$NUM_EVALS" \
  ++save_checkpoint=false \
  ++save_replay_buffer=false \
  ++train.total_timesteps=20000000 \
  ++train.target_env_timesteps="$TARGET_ENV_TIMESTEPS" \
  ++train.enable_three_head=true \
  ++train.schedule_heads_after="$SCHEDULE_HEADS_AFTER" \
  ++train.min_denoising_steps=3 \
  ++train.max_denoising_steps=8 \
  ++train.min_chunk_size=1 \
  ++train.max_chunk_size=4 \
  ++train.fixed_denoising_steps=8 \
  ++train.fixed_chunk_size=4 \
  ++train.step_cost=0.01 \
  ++train.target_nfe="$TARGET_NFE" \
  ++train.actor_compute_lambda="$ACTOR_COMPUTE_LAMBDA" \
  ++train.actor_compute_lambda_warmup="$ACTOR_COMPUTE_LAMBDA_WARMUP" \
  ++train.cost_gate_mode="$COST_GATE_MODE" \
  ++train.cost_start_step="$COST_START_STEP" \
  ++train.cost_warmup_steps="$COST_WARMUP_STEPS" \
  ++train.cost_no_rollback="$COST_NO_ROLLBACK" \
  ++train.budget_penalty_location=rollout_reward \
  ++train.nfe_budget_mode=episode_band \
  ++train.nfe_debt_limit="$NFE_DEBT_LIMIT" \
  ++train.nfe_budget_penalty_scale="$NFE_BUDGET_PENALTY_SCALE" \
  ++train.nfe_target_lower="$NFE_TARGET_LOWER" \
  ++train.nfe_target_upper="$NFE_TARGET_UPPER" \
  ++train.nfe_under_weight="$NFE_UNDER_WEIGHT" \
  ++train.nfe_saving_weight="$NFE_SAVING_WEIGHT" \
  ++train.episode_success_threshold="$EPISODE_SUCCESS_THRESHOLD" \
  ++train.enable_chunk_elasticity="$ENABLE_CHUNK_ELASTICITY" \
  ++train.stochastic_rounding="$STOCHASTIC_ROUNDING" \
  ++train.range_alpha_mode=step_success \
  ++train.range_actuator_floor="$RANGE_ACTUATOR_FLOOR" \
  ++train.range_success_thresh_1="$RANGE_SUCCESS_THRESH_1" \
  ++train.range_success_thresh_2="$RANGE_SUCCESS_THRESH_2" \
  ++train.range_open_rate="$RANGE_OPEN_RATE" \
  ++train.range_close_rate="$RANGE_CLOSE_RATE" \
  ++train.range_success_no_close="$RANGE_SUCCESS_NO_CLOSE" \
  ++train.difficulty_success_open_rate="$DIFFICULTY_SUCCESS_OPEN_RATE" \
  ++train.difficulty_success_close_rate="$DIFFICULTY_SUCCESS_CLOSE_RATE" \
  ++train.schedule_control_mode=prior_residual \
  ++train.schedule_warmup_steps="$SCHEDULE_WARMUP_STEPS" \
  ++train.schedule_gate_floor="$SCHEDULE_GATE_FLOOR" \
  ++train.preprior_residual_scale="$PREPRIOR_RESIDUAL_SCALE" \
  ++train.schedule_residual_scale="$SCHEDULE_RESIDUAL_SCALE" \
  ++train.schedule_entropy_weight=0.0 \
  ++train.difficulty_prior_start_step="$DIFFICULTY_PRIOR_START_STEP" \
  ++train.difficulty_prior_warmup_steps="$DIFFICULTY_PRIOR_WARMUP_STEPS" \
  ++train.difficulty_prior_scale="$DIFFICULTY_PRIOR_SCALE" \
  ++train.difficulty_prior_deadband="$DIFFICULTY_PRIOR_DEADBAND" \
  ++train.difficulty_prior_signal_mode="$DIFFICULTY_PRIOR_SIGNAL_MODE" \
  ++train.difficulty_prior_signal_scale="$DIFFICULTY_PRIOR_SIGNAL_SCALE" \
  ++train.difficulty_prior_gate_floor="$DIFFICULTY_PRIOR_GATE_FLOOR" \
  ++train.difficulty_weight="$DIFFICULTY_WEIGHT" \
  ++train.difficulty_loss_mode=elastic_margin_hinge \
  ++train.difficulty_signal_mode="$DIFFICULTY_SIGNAL_MODE" \
  ++train.difficulty_signal_scale="$DIFFICULTY_SIGNAL_SCALE" \
  ++train.difficulty_start_step="$DIFFICULTY_START_STEP" \
  ++train.difficulty_warmup_steps="$DIFFICULTY_WARMUP_STEPS" \
  ++train.difficulty_margin_target="$DIFFICULTY_MARGIN_TARGET" \
  ++train.difficulty_margin_gate_floor=0.0 \
  ++train.difficulty_margin_start_step="$DIFFICULTY_MARGIN_START_STEP" \
  ++train.difficulty_margin_warmup_steps="$DIFFICULTY_MARGIN_WARMUP_STEPS" \
  ++train.difficulty_mode_margin_weight="$DIFFICULTY_MODE_MARGIN_WEIGHT" \
  ++train.difficulty_quantile_hinge_weight="$DIFFICULTY_QUANTILE_HINGE_WEIGHT" \
  ++train.difficulty_hard_steps_target="$DIFFICULTY_HARD_STEPS_TARGET" \
  ++train.difficulty_hard_chunk_target="$DIFFICULTY_HARD_CHUNK_TARGET" \
  ++train.difficulty_easy_steps_target="$DIFFICULTY_EASY_STEPS_TARGET" \
  ++train.difficulty_easy_chunk_target="$DIFFICULTY_EASY_CHUNK_TARGET" \
  ++train.difficulty_gate_mode=success \
  ++train.difficulty_success_thresh_1="$DIFFICULTY_SUCCESS_THRESH_1" \
  ++train.difficulty_success_thresh_2="$DIFFICULTY_SUCCESS_THRESH_2" \
  ++train.difficulty_success_no_close="$DIFFICULTY_SUCCESS_NO_CLOSE" \
  ++train.eval_success_ema_beta=0.8 \
  ++train.range_success_ema_beta="$RANGE_SUCCESS_EMA_BETA" \
  ++train.difficulty_success_ema_beta="$DIFFICULTY_SUCCESS_EMA_BETA" \
  "${EXTRA_OVERRIDES[@]}"

