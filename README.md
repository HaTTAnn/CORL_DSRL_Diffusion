# CORL_DSRL_Diffusion

DSRL + diffusion policy research code for elastic compute scheduling. The code controls both diffusion denoising steps and executed action chunk length, so the policy can trade off success rate and inference cost.

This repository includes code and configs only. Datasets, pretrained checkpoints, W&B runs, logs, and videos are not included.

## Download

Clone the repository:

```bash
git clone https://github.com/HaTTAnn/CORL_DSRL_Diffusion.git
cd CORL_DSRL_Diffusion
```

Or with SSH:

```bash
git clone git@github.com:HaTTAnn/CORL_DSRL_Diffusion.git
cd CORL_DSRL_Diffusion
```

## Environment Setup

Create a conda environment:

```bash
conda create -n dsrl python=3.10 -y
conda activate dsrl
pip install -U pip setuptools wheel
```

Install the local packages:

```bash
pip install -e stable-baselines3
pip install -e "dppo[robomimic]"
```

For headless GPU servers, set:

```bash
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PWD/dppo:$PWD/stable-baselines3:$PYTHONPATH
```

If MuJoCo / robosuite rendering is not available on the machine, check:

```bash
bash scripts/setup_mujoco210.sh
```

## Required Assets

The repository does not include pretrained diffusion checkpoints or Robomimic/FV normalization files. Put them under the paths used by the configs.

For `can`, the default config expects:

```text
dppo/log/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt
dppo/log/robomimic/can/normalization.npz
```

For other tasks, check these config fields:

```text
cfg/robomimic/dsrl_can.yaml:       base_policy_path, normalization_path
cfg/robomimic/dsrl_lift.yaml:      base_policy_path, normalization_path
cfg/robomimic/dsrl_square.yaml:    base_policy_path, normalization_path
cfg/robomimic/dsrl_transport.yaml: base_policy_path, normalization_path
```

Optional offline replay data is also excluded. If `load_offline_data=True`, make sure the corresponding `offline_data_path` exists.

## How To Run

Main step/chunk ablation launcher:

```bash
bash scripts/run_step_chunk_ablation_v2.sh [PLAN] [GPU_CSV] [ENV_LABEL]
```

Arguments:

```text
PLAN:       12 | square6 | smoke, default 12
GPU_CSV:    CUDA device ids, default 0,1,2,3,4,5,6,7
ENV_LABEL:  runtime label, default uv
```

The current three comparison arms are:

```text
weak_chunk:    steps 3..15, chunk 4 -> 3.75
both_explore:  steps 3..15, chunk 4 -> 3
step_only:     steps 3..15, chunk fixed at 4
```

Preview only:

```bash
DRY_RUN=1 RUN_TAG=stepchunk_v2_3to15 SIGNAL_VARIANT=advz_strong bash scripts/run_step_chunk_ablation_v2.sh 12 0,1,2,3 uv
```

Start training:

```bash
RUN_TAG=stepchunk_v2_3to15 SIGNAL_VARIANT=advz_strong bash scripts/run_step_chunk_ablation_v2.sh 12 0,1,2,3 uv
```

The launcher writes W&B logs online by default. To disable W&B sync:

```bash
WANDB_MODE=offline RUN_TAG=stepchunk_v2_3to15 SIGNAL_VARIANT=advz_strong bash scripts/run_step_chunk_ablation_v2.sh 12 0,1,2,3 uv
```

Single-run entrypoint used by the launcher:

```bash
bash scripts/launch_step_difficulty_h20.sh TASK GPU SEED [ENV_LABEL] [VARIANT]
```

Supported single-run tasks and variants:

```text
TASK:     can | square
VARIANT:  rank_current | advz_current | advz_strong | qstdz_strong
```

## Important Files

```text
train_dsrl.py                                      main training entrypoint
utils.py                                          evaluation, logging, video, W&B filtering
env_utils.py                                      Robomimic/FV wrappers and chunk execution
cfg/robomimic/dsrl_can.yaml                       default CAN config
scripts/run_step_chunk_ablation_v2.sh             current 3-arm step/chunk ablation launcher
scripts/launch_step_difficulty_h20.sh             current single-run launcher
scripts/run_step_difficulty_h20_sweep.sh          optional sweep using the current single-run launcher
stable-baselines3/stable_baselines3/dsrl/dsrl.py  modified DSRL algorithm
dppo/model/diffusion/diffusion.py                 diffusion policy implementation
```

## Core Features

- Elastic denoising steps: choose diffusion steps per query.
- Elastic chunk execution: choose how many actions to execute before replanning.
- Episode-level NFE band reward.
- Success-gated saving bonus.
- Difficulty-prior + actor-residual schedule control.
- Target-normalized allocation difficulty loss aligned with the pi0 mode-margin idea.
- Monotonic cost / difficulty gate opening.
- W&B logs for success, NFE, chunk, difficulty, value loss, Q values, query-level allocation, success rate vs step, and success rate vs env timesteps.

## Key Parameters

The current recommended step/chunk ablation uses:

```text
steps range:             3 to 15
fixed denoising teacher: 8
chunk bounds:            1 to 4
weak_chunk target:       chunk 4 -> 3.75
both_explore target:     chunk 4 -> 3
step_only target:        chunk fixed at 4
NFE target:              1.70, about 15% below fixed 8/4 = 2.00
NFE band:                1.55 to 1.90
saving weight:           0.06
cost lambda:             0.05 -> 0.45
cost gate:               time ramp x success gate 0.65 -> 0.80, open rate 0.08
failed episode cost:     0.15x over-budget penalty before success
difficulty prior:        start 35k, warmup 50k
difficulty loss:         elastic_margin_hinge with target-normalized allocation margin
quantile hinge:          off by default
SQUARE strong gate:      success gates 0.35 -> 0.55, range floor 0.50
```

The full values are in:

```text
cfg/robomimic/dsrl_can.yaml
scripts/run_step_chunk_ablation_v2.sh
scripts/launch_step_difficulty_h20.sh
```

## Notes

- Checkpoints, datasets, W&B logs, videos, and local run logs are intentionally ignored by git.
- The published repository stores `dppo` and `stable-baselines3` as normal source directories, not Git submodules.
- The default Robomimic/FV reward wrapper subtracts `reward_offset=1`, so success is detected through the chunk success signal in the wrapper.

