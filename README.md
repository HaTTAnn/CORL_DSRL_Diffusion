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

Main launch script:

```bash
bash scripts/launch_budget_strong_chunk_ablation.sh TASK MODE [GPU] [SEED] [CONDA_ENV] [VARIANT]
```

Arguments:

```text
TASK:       can | lift | square
MODE:       fixed | elastic
GPU:        CUDA device id, default 7
SEED:       random seed, default 1
CONDA_ENV:  conda env name, default dsrl
VARIANT:    base | tight | stable_v1 | stable_v2
```

Recommended elastic run:

```bash
bash scripts/launch_budget_strong_chunk_ablation.sh can elastic 0 0 dsrl stable_v2
```

Fixed baseline example:

```bash
bash scripts/launch_budget_strong_chunk_ablation.sh can fixed 0 0 dsrl stable_v2
```

The script writes W&B logs online by default. To disable W&B sync:

```bash
WANDB_MODE=offline bash scripts/launch_budget_strong_chunk_ablation.sh can elastic 0 0 dsrl stable_v2
```

## Elastic Soft-Prior Sweep

These scripts run elastic-only soft-prior experiments.

Preview without starting training:

```bash
DRY_RUN=1 scripts/run_elastic_softprior_sweep.sh 8 6,7 dsrl
DRY_RUN=1 scripts/run_elastic_softprior_sweep.sh 16 6,7 dsrl
```

Start training:

```bash
RUN_TAG=softprior_v1 scripts/run_elastic_softprior_sweep.sh 8 6,7 dsrl
RUN_TAG=softprior_v1 scripts/run_elastic_softprior_sweep.sh 16 6,7 dsrl
```

Plans:

```text
8 runs:  can/square x seed0/1 x soft_h2_bal/soft_h3_safe
16 runs: can/square x seed0/1 x soft_h2_bal/soft_h2_res/soft_h3_safe/soft_h2_edge
```

Monitor:

```bash
tmux ls | grep dsrl_diff_elastic
tmux attach -t dsrl_diff_elastic_p8_g6_softprior_v1
```

Detach from tmux without stopping training: `Ctrl-b`, then `d`.

Logs:

```bash
ls run_logs/*softprior_v1*
tail -f run_logs/softprior_v1_can_elastic_soft_h2_bal_seed0_gpu6.log
```

Run one job manually:

```bash
scripts/launch_elastic_softprior.sh can 6 0 dsrl soft_h2_bal
```

Format:

```text
scripts/launch_elastic_softprior.sh TASK GPU SEED CONDA_ENV VARIANT
```

Variants: `soft_h2_bal`, `soft_h2_res`, `soft_h3_safe`, `soft_h2_edge`.

## Important Files

```text
train_dsrl.py                                      main training entrypoint
utils.py                                          evaluation, logging, video, W&B filtering
env_utils.py                                      Robomimic/FV wrappers and chunk execution
cfg/robomimic/dsrl_can.yaml                       default CAN config
scripts/launch_budget_strong_chunk_ablation.sh    main launch script
stable-baselines3/stable_baselines3/dsrl/dsrl.py  modified DSRL algorithm
dppo/model/diffusion/diffusion.py                 diffusion policy implementation
```

## Core Features

- Elastic denoising steps: choose diffusion steps per query.
- Elastic chunk execution: choose how many actions to execute before replanning.
- Episode-level NFE band reward.
- Success-gated saving bonus.
- Difficulty-prior + actor-residual schedule control.
- Monotonic cost / difficulty gate opening.
- W&B logs for success, NFE, chunk, difficulty, value loss, Q values, and query-level allocation.

## Key Parameters

The recommended `stable_v2` setting uses:

```text
steps range:          3 to 8
chunk range:          1 to 4
NFE band:             1.55 to 2.05
saving weight:        0.06
cost lambda:          0.15 -> 0.45
cost gate:            start 10k, warmup 90k
difficulty prior:     start 35k, warmup 50k
schedule mode:        prior_residual
residual scale:       0.75 early, 0.30 after prior handoff
```

The full values are in:

```text
cfg/robomimic/dsrl_can.yaml
scripts/launch_budget_strong_chunk_ablation.sh
```

## Notes

- Checkpoints, datasets, W&B logs, videos, and local run logs are intentionally ignored by git.
- The published repository stores `dppo` and `stable-baselines3` as normal source directories, not Git submodules.
- The default Robomimic/FV reward wrapper subtracts `reward_offset=1`, so success is detected through the chunk success signal in the wrapper.

