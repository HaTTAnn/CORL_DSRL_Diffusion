import os


def _setup_mujoco_py_and_robosuite_env():
	"""mujoco_py (pulled in by robomimic/robosuite) needs MuJoCo 2.1 at ~/.mujoco/mujoco210."""
	mj_bin = os.path.join(os.path.expanduser("~"), ".mujoco", "mujoco210", "bin")
	if not os.path.isdir(mj_bin):
		return
	parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
	for p in (mj_bin, "/usr/lib/nvidia"):
		if os.path.isdir(p) and p not in parts:
			parts.insert(0, p)
	os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
	if not os.environ.get("MUJOCO_GL"):
		os.environ["MUJOCO_GL"] = "egl"


_setup_mujoco_py_and_robosuite_env()

import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import hydra
from omegaconf import OmegaConf
import gym
import sys

_repo_root = os.path.dirname(os.path.abspath(__file__))
_sb3_local = os.path.join(_repo_root, "stable-baselines3")
if os.path.isdir(os.path.join(_sb3_local, "stable_baselines3")):
	sys.path.insert(0, _sb3_local)

sys.path.append("./dppo")

from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))

	
def _as_bool(value):
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y", "on"}
	return bool(value)



@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
	OmegaConf.resolve(cfg)

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	if cfg.use_wandb:
		wandb.init(
			project=cfg.wandb.project,
			name=cfg.name,
			group=cfg.wandb.group,
			monitor_gym=True,
			save_code=True,
			config=OmegaConf.to_container(cfg, resolve=True),
		)
		# Keep existing eval curves on W&B _step and add an explicit env-timestep x-axis.
		wandb.define_metric("eval_by_timesteps/timesteps")
		wandb.define_metric(
			"eval_by_timesteps/success_rate",
			step_metric="eval_by_timesteps/timesteps",
		)
		wandb.define_metric(
			"eval_by_timesteps/reward",
			step_metric="eval_by_timesteps/timesteps",
		)

	MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

	num_env = cfg.env.n_envs
	def make_env(render=False):
		if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
			import d4rl  # noqa: F401
			import d4rl.gym_mujoco  # noqa: F401
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
			env = make_robomimic_env(render=render, env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
		env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
		return env

	def make_train_env():
		return make_env(render=False)

	def make_eval_env():
		return make_env(render=False)

	def make_video_env():
		return make_env(render=True)

	base_policy = load_base_policy(cfg)
	env = make_vec_env(make_train_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
	env.seed(cfg.seed + 1)
	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = []
	for _ in range(cfg.train.num_layers):
		net_arch.append(cfg.train.layer_size)
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		log_std_init=0.0,
		post_linear_modules=post_linear_modules,
		n_critics=cfg.train.n_critics,
	)
	if cfg.algorithm == 'dsrl_sac':
		model = SAC(
			"MlpPolicy",
			env,
			device=str(cfg.device),
			learning_rate=cfg.train.actor_lr,
			buffer_size=20000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
		)
	elif cfg.algorithm == 'dsrl_na':
		model = DSRL(
			"MlpPolicy",
			env,
			device=str(cfg.device),
			learning_rate=cfg.train.actor_lr,
			buffer_size=10000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
			enable_three_head=cfg.train.get("enable_three_head", True),
			schedule_heads_after=cfg.train.get("schedule_heads_after", 0),
			min_denoising_steps=cfg.train.get("min_denoising_steps", 5),
			max_denoising_steps=cfg.train.get("max_denoising_steps", 15),
			min_chunk_size=cfg.train.get("min_chunk_size", 1),
			max_chunk_size=cfg.train.get("max_chunk_size", cfg.act_steps),
			fixed_denoising_steps=cfg.train.get("fixed_denoising_steps", 10),
			fixed_chunk_size=cfg.train.get("fixed_chunk_size", cfg.act_steps),
			step_cost=cfg.train.get("step_cost", 0.01),
			target_nfe=cfg.train.get("target_nfe", 2.2),
			actor_compute_lambda=cfg.train.get("actor_compute_lambda", 0.05),
			actor_compute_lambda_warmup=cfg.train.get("actor_compute_lambda_warmup", None),
			cost_gate_mode=cfg.train.get("cost_gate_mode", "fixed"),
			cost_start_step=cfg.train.get("cost_start_step", 0),
			cost_warmup_steps=cfg.train.get("cost_warmup_steps", 1),
			cost_success_thresh_1=cfg.train.get("cost_success_thresh_1", None),
			cost_success_thresh_2=cfg.train.get("cost_success_thresh_2", None),
			cost_open_rate=cfg.train.get("cost_open_rate", 0.08),
			cost_close_rate=cfg.train.get("cost_close_rate", 1.0),
			cost_no_rollback=cfg.train.get("cost_no_rollback", False),
			budget_penalty_location=cfg.train.get("budget_penalty_location", "rollout_reward"),
			nfe_budget_mode=cfg.train.get("nfe_budget_mode", "episode_band"),
			nfe_debt_limit=cfg.train.get("nfe_debt_limit", 8.0),
			nfe_budget_penalty_scale=cfg.train.get("nfe_budget_penalty_scale", None),
			nfe_target_lower=cfg.train.get("nfe_target_lower", None),
			nfe_target_upper=cfg.train.get("nfe_target_upper", None),
			nfe_under_weight=cfg.train.get("nfe_under_weight", 0.3),
			nfe_saving_weight=cfg.train.get("nfe_saving_weight", 0.0),
			episode_success_threshold=cfg.train.get("episode_success_threshold", -0.5),
			failed_episode_cost_weight=cfg.train.get("failed_episode_cost_weight", 1.0),
			enable_chunk_elasticity=cfg.train.get("enable_chunk_elasticity", False),
			range_alpha_mode=cfg.train.get("range_alpha_mode", "step_success"),
			range_actuator_floor=cfg.train.get("range_actuator_floor", 0.0),
			range_success_thresh_1=cfg.train.get("range_success_thresh_1", 0.45),
			range_success_thresh_2=cfg.train.get("range_success_thresh_2", 0.65),
			range_open_rate=cfg.train.get("range_open_rate", 0.10),
			range_close_rate=cfg.train.get("range_close_rate", 0.04),
			range_success_no_close=cfg.train.get("range_success_no_close", False),
			difficulty_success_open_rate=cfg.train.get("difficulty_success_open_rate", None),
			difficulty_success_close_rate=cfg.train.get("difficulty_success_close_rate", None),
			difficulty_success_no_close=cfg.train.get("difficulty_success_no_close", False),
			schedule_control_mode=cfg.train.get("schedule_control_mode", "prior_residual"),
			schedule_warmup_steps=cfg.train.get("schedule_warmup_steps", 0),
			schedule_gate_floor=cfg.train.get("schedule_gate_floor", 1.0),
			schedule_residual_scale=cfg.train.get("schedule_residual_scale", 0.75),
			preprior_residual_scale=cfg.train.get("preprior_residual_scale", 1.0),
			schedule_entropy_weight=cfg.train.get("schedule_entropy_weight", 0.0),
			difficulty_prior_start_step=cfg.train.get("difficulty_prior_start_step", 0),
			difficulty_prior_warmup_steps=cfg.train.get("difficulty_prior_warmup_steps", 20000),
			difficulty_prior_scale=cfg.train.get("difficulty_prior_scale", 0.85),
			difficulty_prior_deadband=cfg.train.get("difficulty_prior_deadband", 0.0),
			difficulty_prior_signal_mode=cfg.train.get("difficulty_prior_signal_mode", "compute_advantage"),
			difficulty_prior_signal_scale=cfg.train.get("difficulty_prior_signal_scale", 2.0),
			difficulty_prior_gate_floor=cfg.train.get("difficulty_prior_gate_floor", 0.3),
			difficulty_weight=cfg.train.get("difficulty_weight", 0.3),
			difficulty_allocation_scale=cfg.train.get("difficulty_allocation_scale", 1.0),
			difficulty_loss_mode=cfg.train.get("difficulty_loss_mode", "elastic_margin_hinge"),
			difficulty_signal_mode=cfg.train.get("difficulty_signal_mode", "compute_advantage"),
			difficulty_signal_scale=cfg.train.get("difficulty_signal_scale", 1.0),
			difficulty_start_step=cfg.train.get("difficulty_start_step", 0),
			difficulty_warmup_steps=cfg.train.get("difficulty_warmup_steps", 20000),
			difficulty_margin_target=cfg.train.get("difficulty_margin_target", 0.5),
			difficulty_margin_gate_floor=cfg.train.get("difficulty_margin_gate_floor", 0.45),
			difficulty_margin_start_step=cfg.train.get("difficulty_margin_start_step", 0),
			difficulty_margin_warmup_steps=cfg.train.get("difficulty_margin_warmup_steps", 20000),
			difficulty_mode_margin_weight=cfg.train.get("difficulty_mode_margin_weight", 1.0),
			difficulty_quantile_hinge_weight=cfg.train.get("difficulty_quantile_hinge_weight", 0.5),
			difficulty_hard_steps_target=cfg.train.get("difficulty_hard_steps_target", None),
			difficulty_hard_chunk_target=cfg.train.get("difficulty_hard_chunk_target", None),
			difficulty_easy_steps_target=cfg.train.get("difficulty_easy_steps_target", None),
			difficulty_easy_chunk_target=cfg.train.get("difficulty_easy_chunk_target", None),
			difficulty_gate_mode=cfg.train.get("difficulty_gate_mode", "success"),
			difficulty_success_thresh_1=cfg.train.get("difficulty_success_thresh_1", 0.50),
			difficulty_success_thresh_2=cfg.train.get("difficulty_success_thresh_2", 0.70),
			eval_success_ema_beta=cfg.train.get("eval_success_ema_beta", 0.8),
			range_success_ema_beta=cfg.train.get("range_success_ema_beta", None),
			difficulty_success_ema_beta=cfg.train.get("difficulty_success_ema_beta", None),
			stochastic_rounding=cfg.train.get("stochastic_rounding", False),
		)

	save_checkpoint = cfg.get("save_checkpoint", True)
	checkpoint_callback = None
	if save_checkpoint:
		checkpoint_callback = CheckpointCallback(
			save_freq=cfg.save_model_interval,
			save_path=cfg.logdir + "/checkpoint/",
			name_prefix="ft_policy",
			save_replay_buffer=cfg.save_replay_buffer,
			save_vecnormalize=True,
		)

	num_env_eval = cfg.env.n_eval_envs
	eval_env = make_vec_env(make_eval_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
	eval_env.seed(cfg.seed + num_env + 1) 
	save_eval_video = _as_bool(cfg.env.get("save_video", False))
	video_env = None
	if save_eval_video:
		video_env = make_vec_env(make_video_env, n_envs=1, vec_env_cls=DummyVecEnv)
		if cfg.algorithm == 'dsrl_sac':
			video_env = DiffusionPolicyEnvWrapper(video_env, cfg, base_policy)
		video_env.seed(cfg.seed + num_env + num_env_eval + 1)

	logging_callback = LoggingCallback(
		action_chunk = cfg.act_steps, 
		eval_episodes = max(1, int(math.ceil(float(cfg.num_evals) / max(1, int(num_env_eval))))), 
		log_freq=MAX_STEPS, 
		use_wandb=cfg.use_wandb, 
		eval_env=eval_env, 
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		algorithm=cfg.algorithm,
		max_steps=MAX_STEPS,
		video_env=video_env,
		save_eval_video=save_eval_video,
		eval_video_fps=cfg.env.get("eval_video_fps", 20),
		eval_video_max_frames=cfg.env.get("eval_video_max_frames", cfg.env.max_episode_steps),
		eval_video_freq=cfg.env.get("eval_video_freq", 1),
		target_env_timesteps=cfg.train.get("target_env_timesteps", None),
	)

	logging_callback.evaluate(model)
	logging_callback.log_count += 1

	if cfg.load_offline_data:
		load_offline_data(model, cfg.offline_data_path, num_env)
	if cfg.train.init_rollout_steps > 0:
		collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)	
		logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

	callbacks = [logging_callback]
	if checkpoint_callback is not None:
		callbacks.insert(0, checkpoint_callback)
	# Train the agent. Keep the paper-scale default, but allow launch scripts to set
	# task-specific horizons such as can=1M and square=2M.
	model.learn(
		total_timesteps=int(cfg.train.get("total_timesteps", 20000000)),
		callback=callbacks,
	)

	if save_checkpoint and len(cfg.name) > 0:
		model.save(cfg.logdir + "/checkpoint/final")

	# Close environment and wandb
	env.close()
	eval_env.close()
	if video_env is not None:
		video_env.close()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
