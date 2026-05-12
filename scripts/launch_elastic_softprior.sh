#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: launch_elastic_softprior.sh TASK GPU SEED [ENV_LABEL] [VARIANT]}"
GPU="${2:-7}"
SEED="${3:-1}"
ENV_LABEL="${4:-${ENV_LABEL:-uv}}"
VARIANT="${5:-soft_h2_bal}"

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

N_EVAL_ENVS="${N_EVAL_ENVS:-5}"
NUM_EVALS="${NUM_EVALS:-10}"
EVAL_VIDEO="${EVAL_VIDEO:-false}"
EVAL_VIDEO_FPS="${EVAL_VIDEO_FPS:-20}"
EVAL_VIDEO_MAX_FRAMES="${EVAL_VIDEO_MAX_FRAMES:-300}"
EVAL_VIDEO_FREQ="${EVAL_VIDEO_FREQ:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-DSRL_diffusion_FV}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-elastic_softprior}"
RUN_TAG="${RUN_TAG:-$(date +%Y_%m_%d_%H_%M_%S)}"
INIT_ROLLOUT_STEPS="${INIT_ROLLOUT_STEPS:-}"

case "$TASK" in
  can|lift|square) ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

# Shared stable gates/costs. The sweep below varies schedule geometry only.
TARGET_NFE=1.75
NFE_TARGET_LOWER=1.55
NFE_TARGET_UPPER=2.05
ACTOR_COMPUTE_LAMBDA=0.45
ACTOR_COMPUTE_LAMBDA_WARMUP=0.15
NFE_UNDER_WEIGHT=0.15
NFE_DEBT_LIMIT=4.0
NFE_BUDGET_PENALTY_SCALE=10.0
NFE_SAVING_WEIGHT=0.06
EPISODE_SUCCESS_THRESHOLD=-0.5
COST_GATE_MODE=step_monotonic
COST_START_STEP=10000
COST_WARMUP_STEPS=90000
COST_NO_ROLLBACK=1

RANGE_SUCCESS_THRESH_1=0.45
RANGE_SUCCESS_THRESH_2=0.65
RANGE_SUCCESS_EMA_BETA=0.8
RANGE_SUCCESS_NO_CLOSE=1
RANGE_ACTUATOR_FLOOR=0.35

DIFFICULTY_SUCCESS_THRESH_1=0.50
DIFFICULTY_SUCCESS_THRESH_2=0.70
DIFFICULTY_SUCCESS_OPEN_RATE=0.10
DIFFICULTY_SUCCESS_CLOSE_RATE=0.04
DIFFICULTY_SUCCESS_EMA_BETA=0.8
DIFFICULTY_SUCCESS_NO_CLOSE=1

SCHEDULE_HEADS_AFTER=0
SCHEDULE_WARMUP_STEPS=60000
SCHEDULE_GATE_FLOOR=0.3
DIFFICULTY_PRIOR_START_STEP=35000
DIFFICULTY_PRIOR_WARMUP_STEPS=50000
DIFFICULTY_START_STEP=40000
DIFFICULTY_WARMUP_STEPS=50000
DIFFICULTY_MARGIN_START_STEP=40000
DIFFICULTY_MARGIN_WARMUP_STEPS=50000
DIFFICULTY_WEIGHT=0.04
DIFFICULTY_MARGIN_TARGET=0.35
DIFFICULTY_MODE_MARGIN_WEIGHT=0.20
DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.0
DIFFICULTY_HARD_STEPS_TARGET=8
DIFFICULTY_HARD_CHUNK_TARGET=2
DIFFICULTY_EASY_STEPS_TARGET=4
DIFFICULTY_EASY_CHUNK_TARGET=4
DIFFICULTY_PRIOR_SCALE=0.65
PREPRIOR_RESIDUAL_SCALE=0.70
SCHEDULE_RESIDUAL_SCALE=0.40

case "$VARIANT" in
  # Recommended center point: hard chunk target 2, but prior alone rounds hard chunk to 3.
  soft_h2_bal)
    DIFFICULTY_PRIOR_SCALE=0.65
    DIFFICULTY_HARD_CHUNK_TARGET=2
    PREPRIOR_RESIDUAL_SCALE=0.70
    SCHEDULE_RESIDUAL_SCALE=0.40
    DIFFICULTY_WEIGHT=0.04
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.20
    ;;
  # More residual authority with weaker compute-advantage prior.
  soft_h2_res)
    DIFFICULTY_PRIOR_SCALE=0.55
    DIFFICULTY_HARD_CHUNK_TARGET=2
    PREPRIOR_RESIDUAL_SCALE=0.80
    SCHEDULE_RESIDUAL_SCALE=0.45
    DIFFICULTY_WEIGHT=0.035
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.20
    ;;
  # Safer hard-phase geometry: prior target keeps hard chunk around 3; residual must justify chunk 2.
  soft_h3_safe)
    DIFFICULTY_PRIOR_SCALE=0.70
    DIFFICULTY_HARD_CHUNK_TARGET=3
    PREPRIOR_RESIDUAL_SCALE=0.80
    SCHEDULE_RESIDUAL_SCALE=0.45
    DIFFICULTY_WEIGHT=0.04
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.20
    ;;
  # Stronger prior boundary test: hard chunk sits at the 2/3 rounding boundary, residual can push to 2.
  soft_h2_edge)
    DIFFICULTY_PRIOR_SCALE=0.75
    DIFFICULTY_HARD_CHUNK_TARGET=2
    PREPRIOR_RESIDUAL_SCALE=0.65
    SCHEDULE_RESIDUAL_SCALE=0.35
    DIFFICULTY_WEIGHT=0.04
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.20
    ;;
  # SQUARE-only conservative point: keep chunk safe, raise easy-step floor, and relax the NFE target.
  soft_square_safe)
    if [[ "$TASK" != "square" ]]; then
      echo "variant soft_square_safe is only intended for task=square" >&2
      exit 2
    fi
    TARGET_NFE=1.90
    NFE_TARGET_LOWER=1.75
    NFE_TARGET_UPPER=2.15
    DIFFICULTY_PRIOR_SCALE=0.70
    DIFFICULTY_HARD_CHUNK_TARGET=3
    DIFFICULTY_EASY_STEPS_TARGET=5
    DIFFICULTY_EASY_CHUNK_TARGET=4
    PREPRIOR_RESIDUAL_SCALE=0.75
    SCHEDULE_RESIDUAL_SCALE=0.35
    DIFFICULTY_WEIGHT=0.025
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.15
    INIT_ROLLOUT_STEPS="${INIT_ROLLOUT_STEPS:-8000}"
    ;;
  *)
    echo "unknown variant: $VARIANT (expected soft_h2_bal|soft_h2_res|soft_h3_safe|soft_h2_edge|soft_square_safe)" >&2
    exit 2
    ;;
esac

if [[ "$TASK" == "square" ]]; then
  INIT_ROLLOUT_STEPS="${INIT_ROLLOUT_STEPS:-8000}"
fi

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
  echo "Run from project root:" >&2
  echo "  uv venv .venv --python 3.10" >&2
  echo "  source .venv/bin/activate" >&2
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
echo "soft-prior: scale=${DIFFICULTY_PRIOR_SCALE}, hard_steps=${DIFFICULTY_HARD_STEPS_TARGET}, hard_chunk=${DIFFICULTY_HARD_CHUNK_TARGET}, easy_steps=${DIFFICULTY_EASY_STEPS_TARGET}, easy_chunk=${DIFFICULTY_EASY_CHUNK_TARGET}, preprior=${PREPRIOR_RESIDUAL_SCALE}, residual=${SCHEDULE_RESIDUAL_SCALE}"
echo "cost: band=[${NFE_TARGET_LOWER},${NFE_TARGET_UPPER}], lambda=${ACTOR_COMPUTE_LAMBDA_WARMUP}->${ACTOR_COMPUTE_LAMBDA}, debt=${NFE_DEBT_LIMIT}, saving=${NFE_SAVING_WEIGHT}"
if [[ -n "$INIT_ROLLOUT_STEPS" ]]; then
  echo "rollout: init_rollout_steps=${INIT_ROLLOUT_STEPS}"
fi

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
  ++train.enable_chunk_elasticity=true \
  ++train.stochastic_rounding=false \
  ++train.range_alpha_mode=step_success \
  ++train.range_actuator_floor="$RANGE_ACTUATOR_FLOOR" \
  ++train.range_success_thresh_1="$RANGE_SUCCESS_THRESH_1" \
  ++train.range_success_thresh_2="$RANGE_SUCCESS_THRESH_2" \
  ++train.range_open_rate=0.10 \
  ++train.range_close_rate=0.04 \
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
  ++train.difficulty_prior_deadband=0.03 \
  ++train.difficulty_prior_signal_mode=compute_advantage \
  ++train.difficulty_prior_signal_scale=1.0 \
  ++train.difficulty_prior_gate_floor=0.0 \
  ++train.difficulty_weight="$DIFFICULTY_WEIGHT" \
  ++train.difficulty_loss_mode=elastic_margin_hinge \
  ++train.difficulty_signal_mode=compute_advantage \
  ++train.difficulty_signal_scale=0.75 \
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
