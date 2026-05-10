# CORL_DSRL_Diffusion

Research code for DSRL with diffusion-policy scheduling and elastic compute control.

This repository contains the training entrypoints, Robomimic/FV configs, modified DSRL implementation, diffusion policy components, and launch scripts used for CORL-style experiments on adaptive denoising steps and adaptive action chunk execution.

Large experiment artifacts are intentionally excluded from git: W&B runs, logs, checkpoints, cached files, generated videos, and datasets are not included.

## Core Functionality

This codebase extends the original DSRL + diffusion policy pipeline with elastic compute scheduling. The main functionality is:

- Dynamic denoising-step control: the policy can choose how many diffusion denoising steps to use per decision query.
- Dynamic chunk execution: the policy can choose how many generated actions to execute before replanning.
- Episode-level NFE budget reward: training can penalize high amortized NFE and reward successful low-compute episodes inside a target band.
- Success-gated saving reward: compute saving reward is only active for successful episodes and only when NFE is inside the configured band.
- Prior-residual schedule control: a difficulty prior provides the hard/easy structure, while the actor learns residual schedule corrections.
- Monotonic gate scheduling: cost and difficulty gates are designed to open progressively instead of repeatedly rolling back.
- W&B logging filter: logs focus on metrics needed for experiment analysis and debugging.
- Evaluation support: success rate, query-level NFE, chunk selection, difficulty, and video logging are supported by the training utilities.

## Method Summary

The current elastic scheduling setup uses an episode-level NFE band:

```text
rho = sum(k_steps) / sum(k_chunk_requested)
```

where `k_steps` is the selected diffusion denoising step count and `k_chunk_requested` is the selected executed chunk length from the scheduler.

The reward shaping follows:

```text
successful episode:
  rho < lower: weak under penalty
  lower <= rho <= upper: success-gated saving bonus
  rho > upper: over-budget penalty

failed episode:
  rho < lower: no under penalty
  lower <= rho <= upper: no saving bonus and no budget penalty
  rho > upper: over-budget penalty
```

This keeps the reward aligned with the scheduler's requested compute while avoiding a saving bonus for failed episodes.

## Repository Layout

```text
.
├── train_dsrl.py
├── utils.py
├── env_utils.py
├── cfg/
│   ├── gym/
│   └── robomimic/
├── scripts/
│   └── launch_budget_strong_chunk_ablation.sh
├── stable-baselines3/
│   └── stable_baselines3/dsrl/
└── dppo/
    ├── agent/
    ├── cfg/
    ├── env/
    ├── model/
    └── util/
```

Important files:

- `train_dsrl.py`: main training entrypoint.
- `utils.py`: evaluation, video logging, W&B filtering, and rollout utilities.
- `env_utils.py`: environment construction utilities.
- `cfg/robomimic/dsrl_can.yaml`: default Robomimic/FV DSRL config.
- `scripts/launch_budget_strong_chunk_ablation.sh`: main elastic-compute launch script.
- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`: modified DSRL algorithm with schedule heads, cost gates, difficulty prior, and episode-level NFE reward.
- `dppo/model/diffusion/diffusion.py`: diffusion policy components used by the DSRL pipeline.

## Main Features

### 1. Elastic Denoising Steps

The scheduler can choose a denoising step count between configured bounds:

```yaml
min_denoising_steps: 3
max_denoising_steps: 8
fixed_denoising_steps: 8
```

For difficult states, the learned schedule can allocate more denoising steps. For easier states, it can reduce denoising steps to save compute.

### 2. Elastic Chunk Execution

The scheduler can choose how many generated actions to execute before the next diffusion query:

```yaml
min_chunk_size: 1
max_chunk_size: 4
fixed_chunk_size: 4
enable_chunk_elasticity: True
```

Shorter chunks allow more frequent replanning. Longer chunks reduce query frequency and compute cost.

### 3. Prior-Residual Schedule Control

The current recommended control mode is:

```yaml
schedule_control_mode: prior_residual
```

The schedule is decomposed into:

- a difficulty prior that learns the main hard/easy allocation structure;
- an actor residual that keeps control authority early in training and provides local correction later.

Relevant parameters include:

```yaml
range_actuator_floor: 0.35
preprior_residual_scale: 0.75
schedule_residual_scale: 0.30
difficulty_prior_scale: 0.75
schedule_gate_floor: 0.3
```

### 4. Monotonic Cost Gate

The cost pressure is opened gradually:

```yaml
actor_compute_lambda: 0.45
actor_compute_lambda_warmup: 0.15
cost_gate_mode: step_monotonic
cost_start_step: 10000
cost_warmup_steps: 90000
cost_no_rollback: True
```

This gives the actor some early schedule control while avoiding full compute pressure before the actuator is ready.

### 5. Difficulty Prior Gate

The difficulty prior opens after the actor has had time to learn useful residual control:

```yaml
difficulty_prior_start_step: 35000
difficulty_prior_warmup_steps: 50000
difficulty_start_step: 40000
difficulty_warmup_steps: 50000
```

The intended behavior is:

- early phase: actor residual has enough control authority;
- middle phase: weak cost signal starts shaping compute use;
- later phase: difficulty prior takes over the main hard/easy structure;
- residual remains active for fine correction.

### 6. Episode-Level NFE Band Reward

The current budget mode is:

```yaml
budget_penalty_location: rollout_reward
nfe_budget_mode: episode_band
nfe_target_lower: 1.55
nfe_target_upper: 2.05
nfe_under_weight: 0.15
nfe_saving_weight: 0.06
nfe_debt_limit: 4.0
nfe_budget_penalty_scale: 10.0
```

The saving bonus is normalized and bounded by `nfe_saving_weight`, so its scale stays controlled across different episode lengths.

## Installation Notes

This repository depends on the original DSRL/DPPO/Robomimic stack. Exact package versions may depend on the machine setup used for the experiments.

A typical setup is:

```bash
conda create -n dsrl python=3.10
conda activate dsrl
pip install -e stable-baselines3
pip install -e dppo
pip install -r stable-baselines3/docs/requirements.txt  # optional docs/dev dependencies
```

You also need the Robomimic/FV environment dependencies and task datasets. Datasets and pretrained checkpoints are not included in this repository.

## Running Training

From the repository root:

```bash
bash scripts/launch_budget_strong_chunk_ablation.sh
```

The launch script uses environment variables plus Hydra overrides. The default script is configured for elastic chunk/step experiments. Important variables can be edited at the top of the script or passed through the environment before launching.

Example:

```bash
TASK=can SEED=0 GPU_ID=0 bash scripts/launch_budget_strong_chunk_ablation.sh
```

The default Robomimic training configuration is in:

```text
cfg/robomimic/dsrl_can.yaml
```

## Logging and Analysis

The logging filter keeps metrics useful for paper experiments and debugging. The expected analysis categories are:

- success curve;
- success-NFE Pareto analysis;
- stability metrics;
- query-level allocation visualization;
- successful vs failed NFE analysis;
- value/Q diagnostics;
- schedule gate and difficulty-prior diagnostics.

Representative logged metrics include:

```text
train/actor_loss
train/critic_loss
train/value_loss
train/q_value_mean
train/q_value_std
train/actor_compute_lambda_active
train/rollout_nfe_penalty_coef
train/episode_budget_ratio
train/episode_budget_over
train/episode_budget_under
train/episode_budget_saving_bonus
train/range_alpha
train/difficulty_prior_gate
train/difficulty_loss_gate
train/cost_progress
```

Evaluation logs include success, NFE, chunk, difficulty, and video-related outputs where enabled by the training utilities.

## Reproducibility Notes

- This repository stores code and configs only.
- W&B runs, videos, checkpoints, cached data, and local logs are excluded.
- Submodule contents from `dppo` and `stable-baselines3` are included as normal source directories in this published copy.
- The published copy was prepared from the working experiment directory while excluding local artifacts and machine-specific files.

## Citation / Acknowledgement

This code builds on DSRL, Stable-Baselines3, DPPO, and Robomimic/FV components. Please cite the original projects where appropriate when using this repository for research.

