#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: launch_budget_strong_chunk_ablation.sh TASK MODE [GPU] [SEED] [CONDA_ENV] [VARIANT]}"
MODE="${2:?usage: launch_budget_strong_chunk_ablation.sh TASK MODE [GPU] [SEED] [CONDA_ENV] [VARIANT]}"
GPU="${3:-7}"
SEED="${4:-1}"
CONDA_ENV="${5:-${CONDA_ENV:-dsrl}}"
VARIANT="${6:-base}"
N_EVAL_ENVS="${N_EVAL_ENVS:-5}"
NUM_EVALS="${NUM_EVALS:-10}"
EVAL_VIDEO="${EVAL_VIDEO:-false}"
EVAL_VIDEO_FPS="${EVAL_VIDEO_FPS:-20}"
EVAL_VIDEO_MAX_FRAMES="${EVAL_VIDEO_MAX_FRAMES:-300}"
EVAL_VIDEO_FREQ="${EVAL_VIDEO_FREQ:-1}"
RANGE_SUCCESS_THRESH_1=0.45
RANGE_SUCCESS_THRESH_2=0.65
SCHEDULE_HEADS_AFTER=30000
SCHEDULE_WARMUP_STEPS=70000
SCHEDULE_GATE_FLOOR=0.0
RANGE_ACTUATOR_FLOOR=0.0
PREPRIOR_RESIDUAL_SCALE=1.0
SCHEDULE_RESIDUAL_SCALE=0.25
NFE_SAVING_WEIGHT=0.0
EPISODE_SUCCESS_THRESHOLD=-0.5
DIFFICULTY_PRIOR_START_STEP=50000
DIFFICULTY_PRIOR_WARMUP_STEPS=100000
DIFFICULTY_START_STEP=70000
DIFFICULTY_WARMUP_STEPS=100000
DIFFICULTY_MARGIN_START_STEP=70000
DIFFICULTY_MARGIN_WARMUP_STEPS=100000
COST_NO_ROLLBACK=0

case "$TASK" in
  can|lift|square) ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

case "$MODE" in
  fixed) ENABLE_CHUNK=false ;;
  elastic) ENABLE_CHUNK=true ;;
  *) echo "unknown mode: $MODE (expected fixed|elastic)" >&2; exit 2 ;;
esac

case "$VARIANT" in
  base)
    TARGET_NFE=1.75
    NFE_TARGET_LOWER=1.55
    NFE_TARGET_UPPER=2.05
    ACTOR_COMPUTE_LAMBDA=0.35
    NFE_UNDER_WEIGHT=0.15
    NFE_DEBT_LIMIT=0.0
    NFE_BUDGET_PENALTY_SCALE=8.0
    DIFFICULTY_PRIOR_SCALE=0.60
    DIFFICULTY_MARGIN_TARGET=0.50
    DIFFICULTY_MODE_MARGIN_WEIGHT=1.0
    DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.25
    DIFFICULTY_SUCCESS_THRESH_1=0.50
    DIFFICULTY_SUCCESS_THRESH_2=0.70
    DIFFICULTY_SUCCESS_OPEN_RATE=0.10
    DIFFICULTY_SUCCESS_CLOSE_RATE=0.04
    RANGE_SUCCESS_EMA_BETA=0.8
    DIFFICULTY_SUCCESS_EMA_BETA=0.8
    ACTOR_COMPUTE_LAMBDA_WARMUP="$ACTOR_COMPUTE_LAMBDA"
    COST_GATE_MODE=fixed
    COST_START_STEP=0
    COST_WARMUP_STEPS=1
    COST_NO_ROLLBACK=0
    RANGE_SUCCESS_NO_CLOSE=0
    DIFFICULTY_SUCCESS_NO_CLOSE=0
    ;;
  tight)
    TARGET_NFE=1.55
    NFE_TARGET_LOWER=1.35
    NFE_TARGET_UPPER=1.85
    ACTOR_COMPUTE_LAMBDA=0.45
    NFE_UNDER_WEIGHT=0.20
    NFE_DEBT_LIMIT=0.0
    NFE_BUDGET_PENALTY_SCALE=8.0
    DIFFICULTY_PRIOR_SCALE=0.70
    DIFFICULTY_MARGIN_TARGET=0.50
    DIFFICULTY_MODE_MARGIN_WEIGHT=1.0
    DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.25
    DIFFICULTY_SUCCESS_THRESH_1=0.50
    DIFFICULTY_SUCCESS_THRESH_2=0.70
    DIFFICULTY_SUCCESS_OPEN_RATE=0.10
    DIFFICULTY_SUCCESS_CLOSE_RATE=0.04
    RANGE_SUCCESS_EMA_BETA=0.8
    DIFFICULTY_SUCCESS_EMA_BETA=0.8
    ACTOR_COMPUTE_LAMBDA_WARMUP="$ACTOR_COMPUTE_LAMBDA"
    COST_GATE_MODE=fixed
    COST_START_STEP=0
    COST_WARMUP_STEPS=1
    COST_NO_ROLLBACK=0
    RANGE_SUCCESS_NO_CLOSE=0
    DIFFICULTY_SUCCESS_NO_CLOSE=0
    ;;
  stable_v1|stable)
    TARGET_NFE=1.65
    NFE_TARGET_LOWER=1.45
    NFE_TARGET_UPPER=2.05
    ACTOR_COMPUTE_LAMBDA=0.45
    ACTOR_COMPUTE_LAMBDA_WARMUP=0.20
    NFE_UNDER_WEIGHT=0.35
    NFE_DEBT_LIMIT=4.0
    NFE_BUDGET_PENALTY_SCALE=10.0
    DIFFICULTY_PRIOR_SCALE=0.60
    DIFFICULTY_MARGIN_TARGET=0.35
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.30
    DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.0
    DIFFICULTY_SUCCESS_THRESH_1=0.55
    DIFFICULTY_SUCCESS_THRESH_2=0.75
    DIFFICULTY_SUCCESS_OPEN_RATE=0.06
    DIFFICULTY_SUCCESS_CLOSE_RATE=0.12
    RANGE_SUCCESS_EMA_BETA=0.8
    DIFFICULTY_SUCCESS_EMA_BETA=0.6
    COST_GATE_MODE=step_monotonic
    COST_START_STEP=30000
    COST_WARMUP_STEPS=70000
    COST_NO_ROLLBACK=1
    RANGE_SUCCESS_NO_CLOSE=1
    DIFFICULTY_SUCCESS_NO_CLOSE=1
    ;;
  stable_v2|best_v1|best)
    TARGET_NFE=1.75
    NFE_TARGET_LOWER=1.55
    NFE_TARGET_UPPER=2.05
    ACTOR_COMPUTE_LAMBDA=0.45
    ACTOR_COMPUTE_LAMBDA_WARMUP=0.15
    NFE_UNDER_WEIGHT=0.15
    NFE_DEBT_LIMIT=4.0
    NFE_BUDGET_PENALTY_SCALE=10.0
    NFE_SAVING_WEIGHT=0.06
    DIFFICULTY_PRIOR_SCALE=0.75
    DIFFICULTY_MARGIN_TARGET=0.35
    DIFFICULTY_MODE_MARGIN_WEIGHT=0.30
    DIFFICULTY_QUANTILE_HINGE_WEIGHT=0.0
    DIFFICULTY_SUCCESS_THRESH_1=0.50
    DIFFICULTY_SUCCESS_THRESH_2=0.70
    RANGE_SUCCESS_THRESH_1=0.45
    RANGE_SUCCESS_THRESH_2=0.65
    DIFFICULTY_SUCCESS_OPEN_RATE=0.10
    DIFFICULTY_SUCCESS_CLOSE_RATE=0.04
    RANGE_SUCCESS_EMA_BETA=0.8
    DIFFICULTY_SUCCESS_EMA_BETA=0.8
    COST_GATE_MODE=step_monotonic
    COST_START_STEP=10000
    COST_WARMUP_STEPS=90000
    COST_NO_ROLLBACK=1
    SCHEDULE_HEADS_AFTER=0
    SCHEDULE_WARMUP_STEPS=60000
    SCHEDULE_GATE_FLOOR=0.3
    RANGE_ACTUATOR_FLOOR=0.35
    PREPRIOR_RESIDUAL_SCALE=0.75
    SCHEDULE_RESIDUAL_SCALE=0.30
    DIFFICULTY_PRIOR_START_STEP=35000
    DIFFICULTY_PRIOR_WARMUP_STEPS=50000
    DIFFICULTY_START_STEP=40000
    DIFFICULTY_WARMUP_STEPS=50000
    DIFFICULTY_MARGIN_START_STEP=40000
    DIFFICULTY_MARGIN_WARMUP_STEPS=50000
    RANGE_SUCCESS_NO_CLOSE=1
    DIFFICULTY_SUCCESS_NO_CLOSE=1
    ;;
  *)
    echo "unknown variant: $VARIANT (expected base|tight|stable_v1|stable_v2)" >&2
    exit 2
    ;;
esac

cd /root/storage/CODE/txy/dsrl_fv
source /root/miniconda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=300
export WANDB_MODE=online

PROJECT="DSRL_diffusion_FV"
GROUP="fv_${TASK}_episode_band_s3_8_chunk_${VARIANT}"
RUN_NAME="fv_${TASK}_episode_band_s3_8_chunk_${MODE}_${VARIANT}_seed${SEED}_${CONDA_ENV}"

echo "gate policy: range_floor=${RANGE_ACTUATOR_FLOOR}, preprior_residual=${PREPRIOR_RESIDUAL_SCALE}, residual=${SCHEDULE_RESIDUAL_SCALE}, range_no_close=${RANGE_SUCCESS_NO_CLOSE}, difficulty_no_close=${DIFFICULTY_SUCCESS_NO_CLOSE}, cost_gate_mode=${COST_GATE_MODE}, cost_start=${COST_START_STEP}, cost_warmup=${COST_WARMUP_STEPS}, lambda=${ACTOR_COMPUTE_LAMBDA_WARMUP}->${ACTOR_COMPUTE_LAMBDA}, saving=${NFE_SAVING_WEIGHT}"

python train_dsrl.py --config-name "dsrl_${TASK}.yaml" \
  seed="$SEED" \
  device=cuda:0 \
  name="$RUN_NAME" \
  wandb.project="$PROJECT" \
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
  ++train.enable_chunk_elasticity="$ENABLE_CHUNK" \
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
  ++train.difficulty_weight=0.04 \
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
  ++train.difficulty_hard_steps_target=8 \
  ++train.difficulty_hard_chunk_target=1 \
  ++train.difficulty_easy_steps_target=4 \
  ++train.difficulty_easy_chunk_target=4 \
  ++train.difficulty_gate_mode=success \
  ++train.difficulty_success_thresh_1="$DIFFICULTY_SUCCESS_THRESH_1" \
  ++train.difficulty_success_thresh_2="$DIFFICULTY_SUCCESS_THRESH_2" \
  ++train.difficulty_success_no_close="$DIFFICULTY_SUCCESS_NO_CLOSE" \
  ++train.eval_success_ema_beta=0.8 \
  ++train.range_success_ema_beta="$RANGE_SUCCESS_EMA_BETA" \
  ++train.difficulty_success_ema_beta="$DIFFICULTY_SUCCESS_EMA_BETA"
