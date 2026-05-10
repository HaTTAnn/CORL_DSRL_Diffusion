# CORL_DSRL_Diffusion

Research code for DSRL with diffusion-policy scheduling and elastic compute control.

This repository contains the training entrypoints, Robomimic/FV configs, modified DSRL implementation, and launch scripts used for the CORL experiments. Large experiment artifacts are intentionally excluded from git, including W&B runs, logs, checkpoints, cached files, and datasets.

## Main Components

- `train_dsrl.py`: main training entrypoint.
- `utils.py`: evaluation, logging, and environment utilities.
- `cfg/robomimic/`: Robomimic task configs.
- `scripts/launch_budget_strong_chunk_ablation.sh`: launch script for elastic compute experiments.
- `stable-baselines3/stable_baselines3/dsrl/`: modified DSRL algorithm code.
- `dppo/`: diffusion policy components used by the DSRL pipeline.

## Current Method

The current elastic scheduling version uses:

- episode-level NFE band reward shaping;
- requested chunk as the budget denominator;
- success-gated saving reward;
- monotonic cost and difficulty gates;
- prior-residual schedule control for denoising steps and executed chunk.

## Quick Start

Install the required environments and datasets following the original DSRL/DPPO and Robomimic setup. Then launch an experiment from the repository root:

```bash
bash scripts/launch_budget_strong_chunk_ablation.sh
```

The script is configured through environment variables and Hydra overrides. Check `cfg/robomimic/dsrl_can.yaml` for the default training parameters.

## Notes

This repository is for research reproducibility. It does not include pretrained checkpoints, Robomimic datasets, W&B logs, or local machine credentials.

