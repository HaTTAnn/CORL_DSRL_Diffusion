from typing import Any, ClassVar, Optional, TypeVar, Union
from copy import deepcopy

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from torch import nn
from torch.distributions import Normal

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

SelfDSRL = TypeVar("SelfDSRL", bound="DSRL")


class DSRL(OffPolicyAlgorithm):
	"""
	DSRL-NA (noise aliased variant of DSRL)
	Based on the SAC implementation in Stable Baselines3.
	Paper: https://arxiv.org/pdf/2506.15799

	:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
	:param env: The environment to learn from (if registered in Gym, can be str)
	:param learning_rate: learning rate for adam optimizer,
		the same learning rate will be used for all networks (Q-Values, Actor and Value function)
		it can be a function of the current progress remaining (from 1 to 0)
	:param buffer_size: size of the replay buffer
	:param learning_starts: how many steps of the model to collect transitions for before learning starts
	:param batch_size: Minibatch size for each gradient update
	:param tau: the soft update coefficient ("Polyak update", between 0 and 1)
	:param gamma: the discount factor
	:param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
		like ``(5, "step")`` or ``(2, "episode")``.
	:param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
		Set to ``-1`` means to do as many gradient steps as steps done in the environment
		during the rollout.
	:param action_noise: the action noise type (None by default), this can help
		for hard exploration problem. Cf common.noise for the different action noise type.
	:param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
		If ``None``, it will be automatically selected.
	:param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
	:param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
		at a cost of more complexity.
		See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
	:param ent_coef: Entropy regularization coefficient. (Equivalent to
		inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
		Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
	:param target_update_interval: update the target network every ``target_network_update_freq``
		gradient steps.
	:param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
	:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
		instead of action noise exploration (default: False)
	:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
		Default: -1 (only sample at the beginning of the rollout)
	:param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
		during the warm up phase (before learning starts)
	:param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
		the reported success rate, mean episode length, and mean reward over
	:param tensorboard_log: the log location for tensorboard (if None, no logging)
	:param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
	:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
		debug messages
	:param seed: Seed for the pseudo random generators
	:param device: Device (cpu, cuda, ...) on which the code should be run.
		Setting it to auto, the code will be run on the GPU if possible.
	:param _init_setup_model: Whether or not to build the network at the creation of the instance
	:param actor_gradient_steps: Number of gradient steps to take on actor per training update
	:param diffusion_policy: The diffusion policy to use for action generation
	:param diffusion_act_dim: The action dimension for the diffusion policy (tuple of (action chunk length, action_dim))
	:param noise_critic_grad_steps: Number of gradient steps to take on distilled noise critic per training update
	:param critic_backup_combine_type: How to combine the critics for the backup (min or mean)
	:param actor_compute_lambda: Fixed actor-only weight for the budget penalty. Q^A / Bellman stay pure task reward.
	"""
	policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
		"MlpPolicy": MlpPolicy,
		"CnnPolicy": CnnPolicy,
		"MultiInputPolicy": MultiInputPolicy,
	}
	policy: SACPolicy
	actor: Actor
	critic: ContinuousCritic
	critic_target: ContinuousCritic
	critic_noise: ContinuousCritic

	def __init__(
		self,
		policy: Union[str, type[SACPolicy]],
		env: Union[GymEnv, str],
		learning_rate: Union[float, Schedule] = 3e-4,
		buffer_size: int = 1_000_000,  # 1e6
		learning_starts: int = 100,
		batch_size: int = 256,
		tau: float = 0.005,
		gamma: float = 0.99,
		train_freq: Union[int, tuple[int, str]] = 1,
		gradient_steps: int = 1,
		action_noise: Optional[ActionNoise] = None,
		replay_buffer_class: Optional[type[ReplayBuffer]] = None,
		replay_buffer_kwargs: Optional[dict[str, Any]] = None,
		optimize_memory_usage: bool = False,
		ent_coef: Union[str, float] = "auto",
		target_update_interval: int = 1,
		target_entropy: Union[str, float] = "auto",
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		use_sde_at_warmup: bool = False,
		stats_window_size: int = 100,
		tensorboard_log: Optional[str] = None,
		policy_kwargs: Optional[dict[str, Any]] = None,
		verbose: int = 0,
		seed: Optional[int] = None,
		device: Union[th.device, str] = "auto",
		_init_setup_model: bool = True,
		actor_gradient_steps: int = -1,
		diffusion_policy=None,
		diffusion_act_dim=None,
		noise_critic_grad_steps: int = 1,
		critic_backup_combine_type='min',
		enable_three_head: bool = True,
		schedule_heads_after: int = 0,
		min_denoising_steps: int = 3,
		max_denoising_steps: int = 17,
		min_chunk_size: int = 1,
		max_chunk_size: int = 20,
		fixed_denoising_steps: int = 10,
		fixed_chunk_size: int = 20,
		step_cost: float = 0.01,
		target_nfe: float = 2.5,
		actor_compute_lambda: float = 0.0,
		actor_compute_lambda_warmup: Optional[float] = None,
		cost_gate_mode: str = "fixed",
		cost_start_step: float = 0.0,
		cost_warmup_steps: float = 1.0,
		cost_success_thresh_1: Optional[float] = None,
		cost_success_thresh_2: Optional[float] = None,
		cost_open_rate: float = 0.05,
		cost_close_rate: float = 1.0,
		cost_no_rollback: bool = False,
		budget_penalty_location: str = "rollout_reward",
		nfe_budget_mode: str = "episode_band",
		nfe_debt_limit: float = 8.0,
		nfe_budget_penalty_scale: Optional[float] = None,
		nfe_target_lower: Optional[float] = None,
		nfe_target_upper: Optional[float] = None,
		nfe_under_weight: float = 0.3,
		nfe_saving_weight: float = 0.0,
		episode_success_threshold: float = -0.5,
		enable_chunk_elasticity: bool = False,
		range_alpha_mode: str = "step_success",
		range_actuator_floor: float = 0.0,
		range_success_thresh_1: float = 0.45,
		range_success_thresh_2: float = 0.65,
		range_open_rate: float = 0.10,
		range_close_rate: float = 0.04,
		range_success_no_close: bool = False,
		difficulty_success_open_rate: Optional[float] = None,
		difficulty_success_close_rate: Optional[float] = None,
		difficulty_success_no_close: bool = False,
		schedule_control_mode: str = "prior_residual",
		schedule_warmup_steps: float = 0.0,
		schedule_gate_floor: float = 1.0,
		schedule_residual_scale: float = 0.75,
		preprior_residual_scale: float = 1.0,
		schedule_entropy_weight: float = 0.0,
		difficulty_prior_start_step: float = 0.0,
		difficulty_prior_warmup_steps: float = 20000.0,
		difficulty_prior_scale: float = 0.85,
		difficulty_prior_deadband: float = 0.0,
		difficulty_prior_signal_mode: str = "compute_advantage",
		difficulty_prior_signal_scale: float = 2.0,
		difficulty_prior_gate_floor: float = 0.3,
		difficulty_weight: float = 0.3,
		difficulty_allocation_scale: float = 1.0,
		difficulty_loss_mode: str = "elastic_margin_hinge",
		difficulty_signal_mode: str = "compute_advantage",
		difficulty_signal_scale: float = 1.0,
		difficulty_start_step: float = 0.0,
		difficulty_warmup_steps: float = 20000.0,
		difficulty_margin_target: float = 0.5,
		difficulty_margin_gate_floor: float = 0.45,
		difficulty_margin_start_step: float = 0.0,
		difficulty_margin_warmup_steps: float = 20000.0,
		difficulty_mode_margin_weight: float = 1.0,
		difficulty_quantile_hinge_weight: float = 0.5,
		difficulty_hard_steps_target: Optional[float] = None,
		difficulty_hard_chunk_target: Optional[float] = None,
		difficulty_easy_steps_target: Optional[float] = None,
		difficulty_easy_chunk_target: Optional[float] = None,
		difficulty_gate_mode: str = "success",
		difficulty_success_thresh_1: float = 0.50,
		difficulty_success_thresh_2: float = 0.70,
		eval_success_ema_beta: float = 0.8,
		range_success_ema_beta: Optional[float] = None,
		difficulty_success_ema_beta: Optional[float] = None,
		stochastic_rounding: bool = False,
	):
		super().__init__(
			policy,
			env,
			learning_rate,
			buffer_size,
			learning_starts,
			batch_size,
			tau,
			gamma,
			train_freq,
			gradient_steps,
			action_noise,
			replay_buffer_class=replay_buffer_class,
			replay_buffer_kwargs=replay_buffer_kwargs,
			policy_kwargs=policy_kwargs,
			stats_window_size=stats_window_size,
			tensorboard_log=tensorboard_log,
			verbose=verbose,
			device=device,
			seed=seed,
			use_sde=use_sde,
			sde_sample_freq=sde_sample_freq,
			use_sde_at_warmup=use_sde_at_warmup,
			optimize_memory_usage=optimize_memory_usage,
			supported_action_spaces=(spaces.Box,),
			support_multi_env=True,
		)

		self.target_entropy = target_entropy
		self.log_ent_coef = None  # type: Optional[th.Tensor]
		# Entropy coefficient / Entropy temperature
		# Inverse of the reward scale
		self.ent_coef = ent_coef
		self.target_update_interval = target_update_interval
		self.ent_coef_optimizer: Optional[th.optim.Adam] = None
		self.actor_gradient_steps = actor_gradient_steps
		
		self.diffusion_policy = diffusion_policy
		self.diffusion_act_chunk = diffusion_act_dim[0]
		self.diffusion_act_dim = diffusion_act_dim[1]
		self.noise_critic_grad_steps = noise_critic_grad_steps
		self.critic_backup_combine_type = critic_backup_combine_type
		self.enable_three_head = enable_three_head
		self.schedule_heads_after = schedule_heads_after
		self.min_denoising_steps = min_denoising_steps
		self.max_denoising_steps = max_denoising_steps
		self.min_chunk_size = min_chunk_size
		self.max_chunk_size = max_chunk_size
		self.fixed_denoising_steps = fixed_denoising_steps
		self.fixed_chunk_size = fixed_chunk_size
		if self.max_chunk_size > self.diffusion_act_chunk:
			raise ValueError(
				f"max_chunk_size={self.max_chunk_size} exceeds diffusion horizon {self.diffusion_act_chunk}. "
				"The current base policy only predicts that many actions, so larger action chunks are unsupported."
			)
		if self.fixed_chunk_size > self.diffusion_act_chunk:
			raise ValueError(
				f"fixed_chunk_size={self.fixed_chunk_size} exceeds diffusion horizon {self.diffusion_act_chunk}. "
				"Increase the base policy horizon/act_steps first if you want larger executed chunks."
			)
		# step_cost: reserved / config-only; Bellman targets stay pure task reward.
		self.step_cost = step_cost
		self.target_nfe = float(target_nfe)
		self.actor_compute_lambda = float(max(0.0, actor_compute_lambda))
		lambda_warmup = self.actor_compute_lambda if actor_compute_lambda_warmup is None else float(max(0.0, actor_compute_lambda_warmup))
		self.actor_compute_lambda_warmup = float(min(lambda_warmup, self.actor_compute_lambda))
		self.cost_gate_mode = str(cost_gate_mode).lower()
		valid_cost_gate_modes = {"fixed", "success", "open_only_success", "success_mix", "step_monotonic"}
		if self.cost_gate_mode not in valid_cost_gate_modes:
			raise ValueError(f"Unknown cost_gate_mode={cost_gate_mode!r}; expected one of {sorted(valid_cost_gate_modes)}")
		self.cost_start_step = float(max(0.0, cost_start_step))
		self.cost_warmup_steps = float(max(1.0, cost_warmup_steps))
		self.cost_success_thresh_1 = float(range_success_thresh_1 if cost_success_thresh_1 is None else cost_success_thresh_1)
		self.cost_success_thresh_2 = float(range_success_thresh_2 if cost_success_thresh_2 is None else cost_success_thresh_2)
		self.cost_open_rate = float(np.clip(cost_open_rate, 0.0, 1.0))
		self.cost_close_rate = float(np.clip(cost_close_rate, 0.0, 1.0))
		self.cost_no_rollback = bool(cost_no_rollback)
		self.budget_penalty_location = str(budget_penalty_location).lower()
		valid_budget_locations = {"none", "rollout_reward"}
		if self.budget_penalty_location not in valid_budget_locations:
			raise ValueError(f"Unknown budget_penalty_location={budget_penalty_location!r}; expected one of {sorted(valid_budget_locations)}")
		self.nfe_budget_mode = str(nfe_budget_mode).lower()
		valid_rollout_budget_modes = {"episode_band"}
		if self.nfe_budget_mode not in valid_rollout_budget_modes:
			raise ValueError(f"Unknown nfe_budget_mode={nfe_budget_mode!r}; expected one of {sorted(valid_rollout_budget_modes)}")
		self.nfe_debt_limit = float(max(0.0, nfe_debt_limit))
		self.nfe_budget_penalty_scale = float(max(1e-6, self.nfe_debt_limit if nfe_budget_penalty_scale is None else nfe_budget_penalty_scale))
		self.nfe_target_lower = float(self.target_nfe if nfe_target_lower is None else nfe_target_lower)
		self.nfe_target_upper = float(self.target_nfe if nfe_target_upper is None else nfe_target_upper)
		if self.nfe_target_upper < self.nfe_target_lower:
			raise ValueError(f"nfe_target_upper={self.nfe_target_upper} must be >= nfe_target_lower={self.nfe_target_lower}")
		self.nfe_under_weight = float(max(0.0, nfe_under_weight))
		self.nfe_saving_weight = float(max(0.0, nfe_saving_weight))
		self.episode_success_threshold = float(episode_success_threshold)
		self.enable_chunk_elasticity = bool(enable_chunk_elasticity)
		self.range_alpha_mode = str(range_alpha_mode).lower()
		self.range_actuator_floor = float(np.clip(range_actuator_floor, 0.0, 1.0))
		self.range_success_thresh_1 = float(range_success_thresh_1)
		self.range_success_thresh_2 = float(range_success_thresh_2)
		self.range_open_rate = float(np.clip(range_open_rate, 0.0, 1.0))
		self.range_close_rate = float(np.clip(range_close_rate, 0.0, 1.0))
		self.range_success_no_close = bool(range_success_no_close)
		self.difficulty_success_open_rate = float(np.clip(
			self.range_open_rate if difficulty_success_open_rate is None else difficulty_success_open_rate,
			0.0,
			1.0,
		))
		self.difficulty_success_close_rate = float(np.clip(
			self.range_close_rate if difficulty_success_close_rate is None else difficulty_success_close_rate,
			0.0,
			1.0,
		))
		self.difficulty_success_no_close = bool(difficulty_success_no_close)
		self.schedule_control_mode = str(schedule_control_mode).lower()
		valid_schedule_modes = {"learned", "prior_only", "prior_residual"}
		if self.schedule_control_mode not in valid_schedule_modes:
			raise ValueError(f"Unknown schedule_control_mode={schedule_control_mode!r}; expected one of {sorted(valid_schedule_modes)}")
		self.schedule_warmup_steps = float(max(0.0, schedule_warmup_steps))
		self.schedule_gate_floor = float(np.clip(schedule_gate_floor, 0.0, 1.0))
		self.schedule_residual_scale = float(max(0.0, schedule_residual_scale))
		self.preprior_residual_scale = float(max(0.0, preprior_residual_scale))
		self.schedule_entropy_weight = float(max(0.0, schedule_entropy_weight))
		self.difficulty_prior_start_step = float(max(0.0, difficulty_prior_start_step))
		self.difficulty_prior_warmup_steps = float(max(0.0, difficulty_prior_warmup_steps))
		self.difficulty_prior_scale = float(max(0.0, difficulty_prior_scale))
		self.difficulty_prior_deadband = float(np.clip(difficulty_prior_deadband, 0.0, 0.99))
		self.difficulty_prior_signal_mode = str(difficulty_prior_signal_mode).lower()
		if not self._uses_compute_advantage_signal(self.difficulty_prior_signal_mode):
			raise ValueError("Unsupported difficulty signal; use compute_advantage[_<transform>] or q_std[_<transform>].")
		self.difficulty_prior_signal_scale = float(max(0.0, difficulty_prior_signal_scale))
		self.difficulty_prior_gate_floor = float(np.clip(difficulty_prior_gate_floor, 0.0, 1.0))
		self.difficulty_weight = float(max(0.0, difficulty_weight))
		self.difficulty_allocation_scale = float(np.clip(difficulty_allocation_scale, 0.0, 1.0))
		self.difficulty_loss_mode = str(difficulty_loss_mode).lower()
		self.difficulty_signal_mode = str(difficulty_signal_mode).lower()
		if not self._uses_compute_advantage_signal(self.difficulty_signal_mode):
			raise ValueError("Unsupported difficulty signal; use compute_advantage[_<transform>] or q_std[_<transform>].")
		self.difficulty_signal_scale = float(max(0.0, difficulty_signal_scale))
		self.difficulty_start_step = float(max(0.0, difficulty_start_step))
		self.difficulty_warmup_steps = float(max(0.0, difficulty_warmup_steps))
		self.difficulty_margin_target = float(np.clip(difficulty_margin_target, 0.0, 1.0))
		self.difficulty_margin_gate_floor = float(np.clip(difficulty_margin_gate_floor, 0.0, 1.0))
		self.difficulty_margin_start_step = float(max(0.0, difficulty_margin_start_step))
		self.difficulty_margin_warmup_steps = float(max(0.0, difficulty_margin_warmup_steps))
		self.difficulty_mode_margin_weight = float(max(0.0, difficulty_mode_margin_weight))
		self.difficulty_quantile_hinge_weight = float(max(0.0, difficulty_quantile_hinge_weight))
		self.difficulty_hard_steps_target = float(self.max_denoising_steps if difficulty_hard_steps_target is None else difficulty_hard_steps_target)
		self.difficulty_hard_chunk_target = float(self.max_chunk_size if difficulty_hard_chunk_target is None else difficulty_hard_chunk_target)
		self.difficulty_easy_steps_target = float(self.min_denoising_steps if difficulty_easy_steps_target is None else difficulty_easy_steps_target)
		self.difficulty_easy_chunk_target = float(self.max_chunk_size if difficulty_easy_chunk_target is None else difficulty_easy_chunk_target)
		self.difficulty_gate_mode = str(difficulty_gate_mode).lower()
		self.difficulty_success_thresh_1 = float(difficulty_success_thresh_1)
		self.difficulty_success_thresh_2 = float(difficulty_success_thresh_2)
		self.eval_success_ema_beta = float(np.clip(eval_success_ema_beta, 0.0, 0.9999))
		self.range_success_ema_beta = float(np.clip(
			self.eval_success_ema_beta if range_success_ema_beta is None else range_success_ema_beta,
			0.0,
			0.9999,
		))
		self.difficulty_success_ema_beta = float(np.clip(
			self.eval_success_ema_beta if difficulty_success_ema_beta is None else difficulty_success_ema_beta,
			0.0,
			0.9999,
		))
		self.stochastic_rounding = bool(stochastic_rounding)
		self.schedule_head_optimizer: Optional[th.optim.Adam] = None
		self.steps_mu: Optional[nn.Linear] = None
		self.steps_log_std: Optional[nn.Linear] = None
		self.chunk_mu: Optional[nn.Linear] = None
		self.chunk_log_std: Optional[nn.Linear] = None
		self._last_schedule_info: dict[str, Any] = {}
		self._eval_tracking = False
		self._eval_nfes: list[float] = []
		self._eval_steps: list[float] = []
		self._eval_chunks: list[float] = []
		self._eval_steps_target: list[float] = []
		self._eval_chunks_target: list[float] = []
		self._eval_difficulties: list[float] = []
		self._eval_prior_u: list[float] = []
		self._eval_source_scores: list[float] = []
		self._eval_prior_source_scores: list[float] = []
		self._last_eval_steps: Optional[np.ndarray] = None
		self._last_eval_chunks: Optional[np.ndarray] = None
		self._last_eval_steps_target: Optional[np.ndarray] = None
		self._last_eval_chunks_target: Optional[np.ndarray] = None
		self._last_eval_difficulty: Optional[np.ndarray] = None
		self._last_eval_prior_u: Optional[np.ndarray] = None
		self._last_eval_source_score: Optional[np.ndarray] = None
		self._last_eval_prior_source_score: Optional[np.ndarray] = None
		self._eval_success_ema: Optional[float] = None
		self._range_success_ema: Optional[float] = None
		self._difficulty_success_ema: Optional[float] = None
		self._range_success_gate_raw = 0.0
		self._difficulty_success_gate_raw = 0.0
		self._cost_success_gate_raw = 0.0
		self._range_success_mix = 0.0
		self._difficulty_success_mix = 0.0
		self._cost_success_mix = 0.0
		self._gate_monotonic_violation_count = 0
		self._last_monotonic_range_alpha: Optional[float] = None
		self._last_monotonic_difficulty_success_mix: Optional[float] = None
		self._last_monotonic_actor_compute_lambda: Optional[float] = None
		self._last_rollout_steps: Optional[np.ndarray] = None
		self._last_rollout_chunks: Optional[np.ndarray] = None
		self._last_rollout_nfe: Optional[np.ndarray] = None
		self._last_rollout_prior_u: Optional[np.ndarray] = None
		self._last_predict_chunk_exec: Optional[np.ndarray] = None
		self._elapsed_env_steps = 0.0
		self._episode_budget_positions: Optional[list[list[tuple[int, int]]]] = None
		self._episode_budget_steps: Optional[list[list[float]]] = None
		self._episode_budget_chunks: Optional[list[list[float]]] = None
		self._episode_budget_ratios: list[float] = []
		self._episode_budget_unders: list[float] = []
		self._episode_budget_overs: list[float] = []
		self._episode_budget_penalty_totals: list[float] = []
		self._episode_budget_savings: list[float] = []
		self._episode_budget_saving_bonuses: list[float] = []
		self._episode_budget_gross_penalties: list[float] = []
		self._episode_budget_successes: list[float] = []
		self._episode_budget_saving_active: list[float] = []

		if _init_setup_model:
			self._setup_model()

	def _setup_model(self) -> None:
		super()._setup_model()
		self._create_aliases()
		# Running mean and running var
		self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
		self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
		# Target entropy is used when learning the entropy coefficient
		base_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
		if self.target_entropy == "auto":
			# automatically set target entropy if needed
			self.target_entropy = base_entropy - float(self._schedule_entropy_dims())
		else:
			# Force conversion
			# this will also throw an error for unexpected string
			self.target_entropy = float(self.target_entropy)
		# When scheduling is frozen, log π_steps/π_chunk are 0, so α must target w entropy only.
		self._ent_coef_target_w_only = base_entropy
		self._ent_coef_target_joint = base_entropy - float(self._schedule_entropy_dims())

		# The entropy coefficient or entropy can be learned automatically
		# see Automating Entropy Adjustment for Maximum Entropy RL section
		# of https://arxiv.org/abs/1812.05905
		if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
			# Default initial value of ent_coef when learned
			init_value = 1.0
			if "_" in self.ent_coef:
				init_value = float(self.ent_coef.split("_")[1])
				assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

			# Note: we optimize the log of the entropy coeff which is slightly different from the paper
			# as discussed in https://github.com/rail-berkeley/softlearning/issues/37
			self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
			self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
		else:
			# Force conversion to float
			# this will throw an error if a malformed string (different from 'auto')
			# is passed
			self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
		
		policy_noise = self.policy_class(
			self.observation_space,
			self._make_noise_action_space(),
			self.lr_schedule,
			**self.policy_kwargs,
		)
		self.critic_noise = policy_noise.critic
		self.critic_noise = self.critic_noise.to(self.device)
		if self.enable_three_head:
			self._init_schedule_heads()

	def _make_noise_action_space(self) -> spaces.Box:
		if not self.enable_three_head:
			return self.action_space
		assert isinstance(self.action_space, spaces.Box)
		low = np.concatenate([self.action_space.low, np.zeros(2, dtype=self.action_space.low.dtype)], axis=0)
		high = np.concatenate([self.action_space.high, np.ones(2, dtype=self.action_space.high.dtype)], axis=0)
		return spaces.Box(low=low, high=high, dtype=self.action_space.dtype)

	def _init_schedule_heads(self) -> None:
		latent_dim = self.actor.mu.in_features
		self.steps_mu = nn.Linear(latent_dim, 1).to(self.device)
		self.steps_log_std = nn.Linear(latent_dim, 1).to(self.device)
		self.chunk_mu = nn.Linear(latent_dim, 1).to(self.device)
		self.chunk_log_std = nn.Linear(latent_dim, 1).to(self.device)
		nn.init.zeros_(self.steps_mu.weight)
		nn.init.zeros_(self.steps_mu.bias)
		nn.init.zeros_(self.chunk_mu.weight)
		nn.init.zeros_(self.chunk_mu.bias)
		nn.init.zeros_(self.steps_log_std.weight)
		nn.init.constant_(self.steps_log_std.bias, -1.0)
		nn.init.zeros_(self.chunk_log_std.weight)
		nn.init.constant_(self.chunk_log_std.bias, -1.0)
		params = list(self.steps_mu.parameters()) + list(self.steps_log_std.parameters()) + list(self.chunk_mu.parameters()) + list(self.chunk_log_std.parameters())
		self.schedule_head_optimizer = th.optim.Adam(params, lr=self.lr_schedule(1))

	def _schedule_training_active(self) -> bool:
		if not self.enable_three_head:
			return False
		return self._gate_timesteps() >= self.schedule_heads_after

	def _cost_gate_mode_id(self) -> float:
		return {
			"fixed": 0.0,
			"success": 1.0,
			"open_only_success": 2.0,
			"success_mix": 2.0,
			"step_monotonic": 3.0,
		}.get(self.cost_gate_mode, -1.0)

	def _schedule_entropy_dims(self) -> int:
		if not self.enable_three_head:
			return 0
		return 1 + int(self.enable_chunk_elasticity)

	def _cost_progress(self) -> float:
		mode = self.cost_gate_mode
		if mode == "step_monotonic":
			progress = (self._gate_timesteps() - self.cost_start_step) / max(1.0, self.cost_warmup_steps)
			return float(np.clip(progress, 0.0, 1.0))
		if mode in {"open_only_success", "success_mix"}:
			return float(np.clip(self._cost_success_mix, 0.0, 1.0))
		if mode == "success":
			success_for_gate = 0.0 if self._eval_success_ema is None else float(self._eval_success_ema)
			return self._success_gate(success_for_gate, self.cost_success_thresh_1, self.cost_success_thresh_2)
		return 1.0

	def _active_compute_lambda(self) -> float:
		progress = self._cost_progress()
		return float(self.actor_compute_lambda_warmup + (self.actor_compute_lambda - self.actor_compute_lambda_warmup) * progress)

	def _rollout_nfe_penalty_coef(self) -> float:
		if self.budget_penalty_location == "rollout_reward":
			return float(self._active_compute_lambda())
		return 0.0

	def _gate_timesteps(self) -> float:
		return float(max(float(self.num_timesteps), float(self._elapsed_env_steps)))

	def _success_gate(self, success_rate: float, lo: float, hi: float) -> float:
		if hi <= lo:
			return float(success_rate >= hi)
		return float(np.clip((float(success_rate) - lo) / max(1e-6, hi - lo), 0.0, 1.0))

	def _update_gate_mix(
		self,
		current: float,
		target: float,
		open_rate: Optional[float] = None,
		close_rate: Optional[float] = None,
		no_close: bool = False,
	) -> float:
		current = float(np.clip(current, 0.0, 1.0))
		target = float(np.clip(target, 0.0, 1.0))
		open_rate = self.range_open_rate if open_rate is None else float(np.clip(open_rate, 0.0, 1.0))
		close_rate = self.range_close_rate if close_rate is None else float(np.clip(close_rate, 0.0, 1.0))
		if target < current and no_close:
			return current
		rate = open_rate if target > current else close_rate
		return float(np.clip(current + rate * (target - current), 0.0, 1.0))

	def _range_external_gate(self) -> float:
		mode = self.range_alpha_mode
		if mode in {"none", "off", "step", "time"}:
			return 1.0
		if mode in {"success", "step_success"}:
			return float(self._range_success_mix)
		return 1.0

	def _range_actuator_gate(self) -> float:
		external_gate = self._range_external_gate()
		return float(self.range_actuator_floor + (1.0 - self.range_actuator_floor) * external_gate)

	def _difficulty_external_gate(self) -> float:
		mode = self.difficulty_gate_mode
		if mode in {"none", "off", "step", "time"}:
			return 1.0
		if mode in {"range", "range_success"}:
			return float(self._range_success_mix)
		if mode in {"success", "step_success"}:
			return float(self._difficulty_success_mix)
		return 1.0

	def _ramp_gate(self, start_step: float, warmup_steps: float, floor: float = 0.0) -> float:
		timesteps = self._gate_timesteps()
		if timesteps < start_step:
			return 0.0
		if warmup_steps <= 0.0:
			return 1.0
		progress = (timesteps - start_step) / max(1.0, warmup_steps)
		progress = float(np.clip(progress, 0.0, 1.0))
		floor = float(np.clip(floor, 0.0, 1.0))
		return floor + (1.0 - floor) * progress

	def _range_step_progress(self) -> float:
		return self._ramp_gate(float(self.schedule_heads_after), self.schedule_warmup_steps, 0.0)

	def _range_alpha(self) -> float:
		return self._range_step_progress() * self._range_actuator_gate()

	def _range_loss_gate(self) -> float:
		return float(np.clip(self._range_alpha() / 0.1, 0.0, 1.0))

	def _record_monotonic_value(self, attr_name: str, value: float, enabled: bool) -> None:
		previous = getattr(self, attr_name)
		if previous is not None and enabled and value + 1e-8 < float(previous):
			self._gate_monotonic_violation_count += 1
			return
		setattr(self, attr_name, float(value))

	def _gate_snapshot(self) -> dict[str, float]:
		range_alpha = self._range_alpha()
		difficulty_step_gate = self._ramp_gate(self.difficulty_start_step, self.difficulty_warmup_steps, 0.0)
		difficulty_external = self._difficulty_external_gate()
		range_loss_gate = float(np.clip(range_alpha / 0.1, 0.0, 1.0))
		difficulty_combined_gate = difficulty_step_gate * difficulty_external * range_loss_gate
		margin_step_gate = self._ramp_gate(self.difficulty_margin_start_step, self.difficulty_margin_warmup_steps, 0.0)
		margin_floor_gate_raw = self.difficulty_margin_gate_floor * margin_step_gate
		margin_floor_gate = margin_floor_gate_raw * range_loss_gate * difficulty_external
		active_lambda = self._active_compute_lambda()
		self._record_monotonic_value("_last_monotonic_range_alpha", range_alpha, self.range_success_no_close)
		self._record_monotonic_value("_last_monotonic_difficulty_success_mix", self._difficulty_success_mix, self.difficulty_success_no_close)
		self._record_monotonic_value("_last_monotonic_actor_compute_lambda", active_lambda, self.cost_gate_mode == "step_monotonic")
		return {
			"range_alpha": float(range_alpha),
			"range_step_progress": float(self._range_step_progress()),
			"range_success_mix": float(self._range_success_mix),
			"range_success_gate_raw": float(self._range_success_gate_raw),
			"range_success_ema": 0.0 if self._range_success_ema is None else float(self._range_success_ema),
			"range_success_ema_beta": float(self.range_success_ema_beta),
			"range_success_no_close": float(self.range_success_no_close),
			"range_actuator_floor": float(self.range_actuator_floor),
			"range_actuator_gate": float(self._range_actuator_gate()),
			"range_loss_gate": float(range_loss_gate),
			"difficulty_success_gate_raw": float(self._difficulty_success_gate_raw),
			"difficulty_success_mix": float(self._difficulty_success_mix),
			"difficulty_success_ema": 0.0 if self._difficulty_success_ema is None else float(self._difficulty_success_ema),
			"difficulty_success_ema_beta": float(self.difficulty_success_ema_beta),
			"difficulty_success_open_rate": float(self.difficulty_success_open_rate),
			"difficulty_success_close_rate": float(self.difficulty_success_close_rate),
			"difficulty_success_no_close": float(self.difficulty_success_no_close),
			"difficulty_step_gate": float(difficulty_step_gate),
			"difficulty_external_gate": float(difficulty_external),
			"difficulty_combined_gate": float(difficulty_combined_gate),
			"margin_floor_gate_raw": float(margin_floor_gate_raw),
			"margin_floor_gate_after_range_success": float(margin_floor_gate),
			"gate_timesteps": float(self._gate_timesteps()),
			"cost_gate_mode": float(self._cost_gate_mode_id()),
			"cost_progress": float(self._cost_progress()),
			"cost_success_gate_raw": float(self._cost_success_gate_raw),
			"cost_success_mix": float(self._cost_success_mix),
			"actor_compute_lambda_warmup": float(self.actor_compute_lambda_warmup),
			"actor_compute_lambda_target": float(self.actor_compute_lambda),
			"actor_compute_lambda_active": float(active_lambda),
			"gate_monotonic_violation_count": float(self._gate_monotonic_violation_count),
		}

	def _control_value(self, value: float, min_value: int, max_value: int) -> float:
		if max_value <= min_value:
			return 0.5
		ctrl = (float(value) - float(min_value)) / float(max_value - min_value)
		return float(np.clip(ctrl, 0.0, 1.0))

	def _control_from_value(
		self,
		value: float,
		min_value: int,
		max_value: int,
		batch: int,
		device: Optional[th.device] = None,
		dtype: th.dtype = th.float32,
	) -> th.Tensor:
		return th.full((batch, 1), self._control_value(value, min_value, max_value), device=device or self.device, dtype=dtype)

	def _control_from_fixed(self, fixed_value: int, min_value: int, max_value: int, batch: int) -> th.Tensor:
		return self._control_from_value(fixed_value, min_value, max_value, batch, device=self.device)

	def _sample_sigmoid_head(self, mu: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor]:
		log_std = th.clamp(log_std, -3.0, 0.0)
		std = th.exp(log_std)
		dist = Normal(mu, std)
		raw = mu if deterministic else dist.rsample()
		value = th.sigmoid(raw)
		log_prob = dist.log_prob(raw) - th.log(value * (1.0 - value) + 1e-6)
		return value, log_prob.sum(dim=1, keepdim=True)

	def _apply_signal_deadband(self, signal: th.Tensor, deadband: float) -> th.Tensor:
		if deadband <= 0.0:
			return signal
		abs_signal = th.abs(signal)
		shrunk = th.sign(signal) * (abs_signal - deadband) / max(1e-6, 1.0 - deadband)
		return th.where(abs_signal <= deadband, th.zeros_like(signal), th.clamp(shrunk, -1.0, 1.0))

	def _uses_q_std_signal(self, mode: str) -> bool:
		mode = str(mode).lower()
		return mode in {"q_std", "critic_std", "uncertainty"} or mode.startswith("q_std_") or mode.startswith("critic_std_") or mode.startswith("uncertainty_")

	def _uses_compute_advantage_signal(self, mode: str) -> bool:
		mode = str(mode).lower()
		return (
			mode in {"compute_advantage", "advantage", "value_advantage"}
			or mode.startswith("compute_advantage_")
			or mode.startswith("advantage_")
			or mode.startswith("value_advantage_")
			or self._uses_q_std_signal(mode)
		)

	def _difficulty_signal_transform_mode(self, mode: str) -> str:
		mode = str(mode).lower()
		for prefix in ("compute_advantage_", "advantage_", "value_advantage_", "q_std_", "critic_std_", "uncertainty_"):
			if mode.startswith(prefix):
				return mode[len(prefix):]
		if self._uses_compute_advantage_signal(mode):
			return "rank"
		return mode

	def _difficulty_source_score(self, obs: th.Tensor, w_noise: Optional[th.Tensor], mode: str) -> th.Tensor:
		if self._uses_q_std_signal(mode):
			return self._estimate_q_std(obs, w_noise=w_noise)
		return self._estimate_compute_advantage(obs, w_noise=w_noise)

	def _combine_noise_q_values(self, q_values: th.Tensor) -> th.Tensor:
		if self.critic_backup_combine_type == "min":
			combined, _ = th.min(q_values, dim=1, keepdim=True)
		else:
			combined = th.mean(q_values, dim=1, keepdim=True)
		return combined

	def _estimate_compute_advantage(self, obs: th.Tensor, w_noise: Optional[th.Tensor] = None) -> th.Tensor:
		batch = obs.shape[0]
		with th.no_grad():
			if not self.enable_three_head:
				return th.zeros((batch, 1), device=obs.device, dtype=obs.dtype)
			if w_noise is None:
				w_noise, _ = self.actor.action_log_prob(obs)
			w_noise = w_noise.detach().to(device=obs.device)
			dtype = w_noise.dtype
			easy_steps = self._control_from_value(self.difficulty_easy_steps_target, self.min_denoising_steps, self.max_denoising_steps, batch, device=obs.device, dtype=dtype)
			hard_steps = self._control_from_value(self.difficulty_hard_steps_target, self.min_denoising_steps, self.max_denoising_steps, batch, device=obs.device, dtype=dtype)
			easy_chunk = self._control_from_value(self.difficulty_easy_chunk_target, self.min_chunk_size, self.max_chunk_size, batch, device=obs.device, dtype=dtype)
			hard_chunk = self._control_from_value(self.difficulty_hard_chunk_target, self.min_chunk_size, self.max_chunk_size, batch, device=obs.device, dtype=dtype)
			easy_input = th.cat([w_noise, easy_steps, easy_chunk], dim=1)
			hard_input = th.cat([w_noise, hard_steps, hard_chunk], dim=1)
			q_easy = self._combine_noise_q_values(th.cat(self.critic_noise(obs, easy_input), dim=1))
			q_hard = self._combine_noise_q_values(th.cat(self.critic_noise(obs, hard_input), dim=1))
			return (q_hard - q_easy).detach()

	def _estimate_q_std(self, obs: th.Tensor, w_noise: Optional[th.Tensor] = None) -> th.Tensor:
		batch = obs.shape[0]
		with th.no_grad():
			if not self.enable_three_head:
				return th.zeros((batch, 1), device=obs.device, dtype=obs.dtype)
			if w_noise is None:
				w_noise, _ = self.actor.action_log_prob(obs)
			w_noise = w_noise.detach().to(device=obs.device)
			dtype = w_noise.dtype
			base_steps = self._control_from_fixed(
				self.fixed_denoising_steps,
				self.min_denoising_steps,
				self.max_denoising_steps,
				batch,
			).to(device=obs.device, dtype=dtype)
			base_chunk = self._control_from_fixed(
				self.fixed_chunk_size,
				self.min_chunk_size,
				self.max_chunk_size,
				batch,
			).to(device=obs.device, dtype=dtype)
			q_values = th.cat(self.critic_noise(obs, th.cat([w_noise, base_steps, base_chunk], dim=1)), dim=1)
			return q_values.std(dim=1, keepdim=True, unbiased=False).detach()

	def _difficulty_signal_from_score(self, score: th.Tensor, mode: str, scale: float) -> th.Tensor:
		mode = str(mode).lower()
		if mode in {"none", "off", "zero"} or scale <= 0.0:
			return th.zeros_like(score)
		flat = score.reshape(-1)
		if mode.startswith("rank"):
			if flat.numel() <= 1 or (flat.max() - flat.min()).abs().item() <= 1e-8:
				base = th.zeros_like(flat)
			else:
				order = th.argsort(flat)
				base = th.empty_like(flat)
				base[order] = th.linspace(-1.0, 1.0, flat.numel(), device=flat.device, dtype=flat.dtype)
			base = base.reshape_as(score)
		elif mode.startswith("z"):
			mean = score.mean()
			std = score.std(unbiased=False).clamp_min(1e-6)
			base = (score - mean) / std
		else:
			denom = score.detach().abs().mean().clamp_min(1e-6)
			base = score / denom
		if "sigmoid" in mode:
			return th.clamp(2.0 * th.sigmoid(scale * base) - 1.0, -1.0, 1.0)
		return th.clamp(th.tanh(scale * base), -1.0, 1.0)

	def _prior_controls(self, prior_u: th.Tensor, default_steps: th.Tensor, default_chunk: th.Tensor) -> tuple[th.Tensor, th.Tensor, float]:
		batch = prior_u.shape[0]
		dtype = prior_u.dtype
		device = prior_u.device
		mix = 0.5 * (prior_u + 1.0)
		easy_steps = self._control_from_value(self.difficulty_easy_steps_target, self.min_denoising_steps, self.max_denoising_steps, batch, device=device, dtype=dtype)
		hard_steps = self._control_from_value(self.difficulty_hard_steps_target, self.min_denoising_steps, self.max_denoising_steps, batch, device=device, dtype=dtype)
		easy_chunk = self._control_from_value(self.difficulty_easy_chunk_target, self.min_chunk_size, self.max_chunk_size, batch, device=device, dtype=dtype)
		hard_chunk = self._control_from_value(self.difficulty_hard_chunk_target, self.min_chunk_size, self.max_chunk_size, batch, device=device, dtype=dtype)
		prior_steps = easy_steps + mix * (hard_steps - easy_steps)
		prior_chunk = easy_chunk + mix * (hard_chunk - easy_chunk)
		prior_step_gate = self._ramp_gate(self.difficulty_prior_start_step, self.difficulty_prior_warmup_steps, self.difficulty_prior_gate_floor)
		prior_gate = prior_step_gate * self._difficulty_external_gate()
		prior_scale = prior_gate * self.difficulty_prior_scale
		prior_steps = default_steps + prior_scale * (prior_steps - default_steps)
		prior_chunk = default_chunk + prior_scale * (prior_chunk - default_chunk)
		return th.clamp(prior_steps, 0.0, 1.0), th.clamp(prior_chunk, 0.0, 1.0), prior_gate


	def _difficulty_alignment_loss(
		self,
		steps_ctrl: th.Tensor,
		chunk_ctrl: th.Tensor,
		difficulty: th.Tensor,
		mode_steps_ctrl: Optional[th.Tensor] = None,
		mode_chunk_ctrl: Optional[th.Tensor] = None,
	) -> tuple[th.Tensor, dict[str, float]]:
		zero = steps_ctrl.sum() * 0.0
		gate_logs = self._gate_snapshot()
		gate_logs["difficulty_loss_gate"] = 0.0
		gate_logs["difficulty_margin_gate"] = 0.0
		gate_logs["monotonic_inversion_rate"] = 0.0
		if self.difficulty_weight <= 0.0 or self.difficulty_loss_mode in {"none", "off"}:
			return zero, gate_logs

		mode = self.difficulty_loss_mode
		margin_modes = {"elastic_margin", "margin", "difficulty_margin", "elastic_margin_hinge", "margin_hinge", "elastic_margin_quantile"}
		if mode in margin_modes:
			loss_gate = max(gate_logs["difficulty_combined_gate"], gate_logs["margin_floor_gate_after_range_success"])
		else:
			loss_gate = gate_logs["difficulty_combined_gate"]
		gate_logs["difficulty_loss_gate"] = float(loss_gate)
		gate_logs["difficulty_margin_gate"] = float(gate_logs["margin_floor_gate_after_range_success"])
		if loss_gate <= 0.0:
			return zero, gate_logs

		batch = difficulty.shape[0]
		dtype = steps_ctrl.dtype
		device = steps_ctrl.device
		u = th.clamp(difficulty.detach().to(device=device, dtype=dtype), -1.0, 1.0)
		confidence = th.clamp(u.abs(), 0.0, 1.0).detach()
		has_mode_controls = mode_steps_ctrl is not None and mode_chunk_ctrl is not None
		if has_mode_controls:
			mode_steps_ctrl = mode_steps_ctrl.to(device=device, dtype=dtype)
			mode_chunk_ctrl = mode_chunk_ctrl.to(device=device, dtype=dtype)
		allocation_scale = th.as_tensor(self.difficulty_allocation_scale, device=device, dtype=dtype)
		margin_target = th.as_tensor(self.difficulty_margin_target, device=device, dtype=dtype)
		mode_margin_weight = th.as_tensor(max(self.difficulty_mode_margin_weight, 0.0), device=device, dtype=dtype)
		quantile_hinge_weight = th.as_tensor(max(self.difficulty_quantile_hinge_weight, 0.0), device=device, dtype=dtype)

		def confidence_mean(values: th.Tensor) -> th.Tensor:
			return (confidence * values).mean()

		def indexed_mean(values: th.Tensor, indices: th.Tensor) -> th.Tensor:
			if indices.numel() == 0:
				return zero
			return values[indices].mean()

		easy_steps = self._control_from_value(self.difficulty_easy_steps_target, self.min_denoising_steps, self.max_denoising_steps, batch, device=device, dtype=dtype)
		hard_steps = self._control_from_value(self.difficulty_hard_steps_target, self.min_denoising_steps, self.max_denoising_steps, batch, device=device, dtype=dtype)
		easy_chunk = self._control_from_value(self.difficulty_easy_chunk_target, self.min_chunk_size, self.max_chunk_size, batch, device=device, dtype=dtype)
		hard_chunk = self._control_from_value(self.difficulty_hard_chunk_target, self.min_chunk_size, self.max_chunk_size, batch, device=device, dtype=dtype)

		def directional_allocation(step_values: th.Tensor, chunk_values: th.Tensor) -> th.Tensor:
			terms: list[th.Tensor] = []
			step_delta = hard_steps - easy_steps
			if step_delta.detach().abs().max().item() > 1e-6:
				step_score = 2.0 * (step_values - easy_steps) / step_delta - 1.0
				terms.append(th.clamp(step_score, -1.0, 1.0))
			chunk_delta = hard_chunk - easy_chunk
			if self.enable_chunk_elasticity and chunk_delta.detach().abs().max().item() > 1e-6:
				chunk_score = 2.0 * (chunk_values - easy_chunk) / chunk_delta - 1.0
				terms.append(th.clamp(chunk_score, -1.0, 1.0))
			if len(terms) == 0:
				return th.zeros_like(step_values)
			return th.stack(terms, dim=0).mean(dim=0)

		allocation = directional_allocation(steps_ctrl, chunk_ctrl)
		mode_allocation = directional_allocation(mode_steps_ctrl, mode_chunk_ctrl) if has_mode_controls else None

		def quantile_hinge_loss_for(step_values: th.Tensor, chunk_values: th.Tensor) -> th.Tensor:
			if batch < 4:
				return zero
			u_flat = u.reshape(-1)
			_, order = th.sort(u_flat)
			k = max(1, int(np.ceil(0.30 * batch)))
			easy_idx = order[:k]
			hard_idx = order[-k:]
			step_flat = step_values.reshape(-1)
			chunk_flat = chunk_values.reshape(-1)
			easy_steps_flat = easy_steps.reshape(-1)
			hard_steps_flat = hard_steps.reshape(-1)
			easy_chunk_flat = easy_chunk.reshape(-1)
			hard_chunk_flat = hard_chunk.reshape(-1)
			return (
				indexed_mean(F.relu(hard_steps_flat - step_flat).pow(2), hard_idx)
				+ indexed_mean(F.relu(chunk_flat - hard_chunk_flat).pow(2), hard_idx)
				+ indexed_mean(F.relu(step_flat - easy_steps_flat).pow(2), easy_idx)
				+ indexed_mean(F.relu(easy_chunk_flat - chunk_flat).pow(2), easy_idx)
			)

		if mode in {"target_mse", "mse"}:
			mix = 0.5 * (u + 1.0)
			steps_target = easy_steps + mix * (hard_steps - easy_steps)
			chunk_target = easy_chunk + mix * (hard_chunk - easy_chunk)
			loss = F.mse_loss(steps_ctrl, steps_target.detach()) + F.mse_loss(chunk_ctrl, chunk_target.detach())
		elif mode in {"bidir_allocation", "bidir", "allocation"}:
			allocation_target = th.clamp(allocation_scale * u, -1.0, 1.0).detach()
			loss = confidence_mean((allocation - allocation_target).pow(2))
		elif mode in margin_modes:
			effective_margin = allocation_scale * margin_target
			sample_margin_error = F.relu(effective_margin - u * allocation)
			sample_margin_loss = confidence_mean(sample_margin_error.pow(2))
			if mode_allocation is not None:
				mode_margin_error = F.relu(effective_margin - u * mode_allocation)
				mode_margin_loss = confidence_mean(mode_margin_error.pow(2))
			else:
				mode_margin_loss = zero
			quantile_loss = quantile_hinge_loss_for(steps_ctrl, chunk_ctrl)
			if has_mode_controls:
				mode_quantile_loss = quantile_hinge_loss_for(mode_steps_ctrl, mode_chunk_ctrl)
			else:
				mode_quantile_loss = zero
			loss = sample_margin_loss + mode_margin_weight * mode_margin_loss + quantile_hinge_weight * (quantile_loss + mode_margin_weight * mode_quantile_loss)
		elif mode in {"target_hinge", "hard_easy_hinge"}:
			loss = zero
			terms = 0
			u_flat = u.reshape(-1)
			hard_mask = u_flat >= self.difficulty_margin_target
			easy_mask = u_flat <= -self.difficulty_margin_target
			if hard_mask.any():
				loss = loss + F.relu(hard_steps[hard_mask] - steps_ctrl[hard_mask]).pow(2).mean()
				loss = loss + F.relu(chunk_ctrl[hard_mask] - hard_chunk[hard_mask]).pow(2).mean()
				terms += 1
			if easy_mask.any():
				loss = loss + F.relu(steps_ctrl[easy_mask] - easy_steps[easy_mask]).pow(2).mean()
				loss = loss + F.relu(easy_chunk[easy_mask] - chunk_ctrl[easy_mask]).pow(2).mean()
				terms += 1
			if terms > 0:
				loss = loss / float(terms)
		elif mode in {"quantile_hinge", "hinge", "difficulty_hinge"}:
			quantile_loss = quantile_hinge_loss_for(steps_ctrl, chunk_ctrl)
			if has_mode_controls:
				mode_quantile_loss = quantile_hinge_loss_for(mode_steps_ctrl, mode_chunk_ctrl)
			else:
				mode_quantile_loss = zero
			loss = quantile_loss + mode_margin_weight * mode_quantile_loss
		else:
			raise ValueError(f"Unknown difficulty_loss_mode={self.difficulty_loss_mode!r}")

		if batch >= 2:
			with th.no_grad():
				allocation_flat = allocation.reshape(-1)
				u_flat = u.reshape(-1)
				harder = u_flat[:, None] > u_flat[None, :]
				inverted = allocation_flat[:, None] < allocation_flat[None, :]
				valid_pairs = harder.sum().clamp_min(1)
				gate_logs["monotonic_inversion_rate"] = float((harder & inverted).sum().float().div(valid_pairs.float()).item())
		loss = self.difficulty_weight * loss_gate * loss
		return loss, gate_logs

	def _mode_id(self) -> float:
		return {"learned": 0.0, "prior_only": 1.0, "prior_residual": 2.0}.get(self.schedule_control_mode, -1.0)

	def _set_default_schedule_info(self, batch: int, device: th.device, dtype: th.dtype) -> None:
		zeros = th.zeros((batch, 1), device=device, dtype=dtype)
		self._last_schedule_info = {
			"difficulty": zeros,
			"difficulty_prior_u": zeros,
			"difficulty_source_score": zeros,
			"difficulty_prior_source_score": zeros,
			"schedule_gate": 0.0,
			"difficulty_prior_gate": 0.0,
			"difficulty_loss_gate": 0.0,
			"difficulty_margin_gate": 0.0,
			"schedule_control_mode_id": self._mode_id(),
			**self._gate_snapshot(),
		}

	def _record_eval_schedule(self, steps_ctrl: th.Tensor, chunk_ctrl: th.Tensor, deterministic: bool) -> None:
		if not self._eval_tracking:
			return
		step_targets = self._map_control_to_float(
			steps_ctrl,
			self.min_denoising_steps,
			self.max_denoising_steps,
		).to(dtype=th.float32)
		chunk_targets = self._map_control_to_float(
			chunk_ctrl,
			self.min_chunk_size,
			self.max_chunk_size,
		).to(dtype=th.float32)
		step_vals = self._map_control_to_int(
			steps_ctrl,
			self.min_denoising_steps,
			self.max_denoising_steps,
			stochastic=self.stochastic_rounding and (not deterministic),
		).to(dtype=th.float32)
		chunk_vals = self._map_control_to_int(
			chunk_ctrl,
			self.min_chunk_size,
			self.max_chunk_size,
			stochastic=self.stochastic_rounding and (not deterministic),
		).to(dtype=th.float32)
		nfe_vals = step_vals / th.clamp(chunk_vals, min=1.0)
		info = self._last_schedule_info
		self._eval_steps.extend(step_vals.detach().cpu().numpy().tolist())
		self._eval_chunks.extend(chunk_vals.detach().cpu().numpy().tolist())
		self._eval_steps_target.extend(step_targets.detach().cpu().numpy().tolist())
		self._eval_chunks_target.extend(chunk_targets.detach().cpu().numpy().tolist())
		self._eval_nfes.extend(nfe_vals.detach().cpu().numpy().tolist())
		if "difficulty" in info:
			self._eval_difficulties.extend(info["difficulty"].reshape(-1).detach().cpu().numpy().tolist())
		if "difficulty_prior_u" in info:
			self._eval_prior_u.extend(info["difficulty_prior_u"].reshape(-1).detach().cpu().numpy().tolist())
		if "difficulty_source_score" in info:
			self._eval_source_scores.extend(info["difficulty_source_score"].reshape(-1).detach().cpu().numpy().tolist())
		if "difficulty_prior_source_score" in info:
			self._eval_prior_source_scores.extend(info["difficulty_prior_source_score"].reshape(-1).detach().cpu().numpy().tolist())
		self._last_eval_steps = step_vals.detach().cpu().numpy().astype(np.float32).reshape(-1)
		self._last_eval_chunks = chunk_vals.detach().cpu().numpy().astype(np.float32).reshape(-1)
		self._last_eval_steps_target = step_targets.detach().cpu().numpy().astype(np.float32).reshape(-1)
		self._last_eval_chunks_target = chunk_targets.detach().cpu().numpy().astype(np.float32).reshape(-1)
		difficulty = info.get("difficulty")
		if difficulty is not None:
			self._last_eval_difficulty = difficulty.reshape(-1).detach().cpu().numpy().astype(np.float32)
		else:
			self._last_eval_difficulty = np.zeros_like(self._last_eval_steps, dtype=np.float32)
		prior_u = info.get("difficulty_prior_u")
		if prior_u is not None:
			self._last_eval_prior_u = prior_u.reshape(-1).detach().cpu().numpy().astype(np.float32)
		else:
			self._last_eval_prior_u = np.zeros_like(self._last_eval_steps, dtype=np.float32)
		source_score = info.get("difficulty_source_score")
		if source_score is not None:
			self._last_eval_source_score = source_score.reshape(-1).detach().cpu().numpy().astype(np.float32)
		else:
			self._last_eval_source_score = np.zeros_like(self._last_eval_steps, dtype=np.float32)
		prior_source_score = info.get("difficulty_prior_source_score")
		if prior_source_score is not None:
			self._last_eval_prior_source_score = prior_source_score.reshape(-1).detach().cpu().numpy().astype(np.float32)
		else:
			self._last_eval_prior_source_score = np.zeros_like(self._last_eval_steps, dtype=np.float32)

	def _sample_schedule_controls(
		self,
		obs: th.Tensor,
		deterministic: bool = False,
		train_scheduling: Optional[bool] = None,
		w_noise: Optional[th.Tensor] = None,
	) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
		batch = obs.shape[0]
		default_steps = self._control_from_fixed(
			self.fixed_denoising_steps,
			self.min_denoising_steps,
			self.max_denoising_steps,
			batch,
		).to(device=obs.device, dtype=obs.dtype)
		default_chunk = self._control_from_fixed(
			self.fixed_chunk_size,
			self.min_chunk_size,
			self.max_chunk_size,
			batch,
		).to(device=obs.device, dtype=obs.dtype)
		default_log_prob = th.zeros((batch, 1), device=obs.device, dtype=obs.dtype)
		self._set_default_schedule_info(batch, obs.device, obs.dtype)
		if not self.enable_three_head:
			self._record_eval_schedule(default_steps, default_chunk, deterministic)
			return default_steps, default_log_prob, default_chunk, default_log_prob
		if train_scheduling is None:
			train_scheduling = self._schedule_training_active()
		if not train_scheduling:
			self._record_eval_schedule(default_steps, default_chunk, deterministic)
			return default_steps, default_log_prob, default_chunk, default_log_prob

		difficulty_prior_source_score = self._difficulty_source_score(obs, w_noise, self.difficulty_prior_signal_mode)
		if self.difficulty_signal_mode == self.difficulty_prior_signal_mode:
			difficulty_source_score = difficulty_prior_source_score
		else:
			difficulty_source_score = self._difficulty_source_score(obs, w_noise, self.difficulty_signal_mode)
		prior_u = self._difficulty_signal_from_score(
			difficulty_prior_source_score,
			self._difficulty_signal_transform_mode(self.difficulty_prior_signal_mode),
			self.difficulty_prior_signal_scale,
		)
		prior_u = self._apply_signal_deadband(prior_u, self.difficulty_prior_deadband)
		difficulty = self._difficulty_signal_from_score(
			difficulty_source_score,
			self._difficulty_signal_transform_mode(self.difficulty_signal_mode),
			self.difficulty_signal_scale,
		)
		prior_steps, prior_chunk, prior_gate = self._prior_controls(prior_u, default_steps, default_chunk)
		schedule_step_gate = self._ramp_gate(float(self.schedule_heads_after), self.schedule_warmup_steps, self.schedule_gate_floor)
		schedule_gate = schedule_step_gate * self._range_actuator_gate()
		prior_authority = float(np.clip(float(prior_gate) * float(schedule_gate), 0.0, 1.0))
		handoff = prior_authority * prior_authority * (3.0 - 2.0 * prior_authority)
		active_residual_scale = self.preprior_residual_scale - handoff * (self.preprior_residual_scale - self.schedule_residual_scale)

		if self.schedule_control_mode == "prior_only":
			policy_steps = prior_steps
			policy_chunk = prior_chunk
			mode_policy_steps = prior_steps
			mode_policy_chunk = prior_chunk
			log_prob_steps = default_log_prob
			log_prob_chunk = default_log_prob
		else:
			assert self.steps_mu is not None and self.steps_log_std is not None
			assert self.chunk_mu is not None and self.chunk_log_std is not None
			features = self.actor.extract_features(obs, self.actor.features_extractor)
			latent = self.actor.latent_pi(features)
			steps_mu = self.steps_mu(latent)
			steps_log_std = self.steps_log_std(latent)
			chunk_mu = self.chunk_mu(latent)
			chunk_log_std = self.chunk_log_std(latent)
			learned_steps, log_prob_steps = self._sample_sigmoid_head(steps_mu, steps_log_std, deterministic=deterministic)
			learned_chunk, log_prob_chunk = self._sample_sigmoid_head(chunk_mu, chunk_log_std, deterministic=deterministic)
			mode_learned_steps, _ = self._sample_sigmoid_head(steps_mu, steps_log_std, deterministic=True)
			mode_learned_chunk, _ = self._sample_sigmoid_head(chunk_mu, chunk_log_std, deterministic=True)
			if self.schedule_control_mode == "learned":
				policy_steps = learned_steps
				policy_chunk = learned_chunk
				mode_policy_steps = mode_learned_steps
				mode_policy_chunk = mode_learned_chunk
			else:
				residual_center = th.full_like(learned_steps, 0.5)
				policy_steps = prior_steps + active_residual_scale * (learned_steps - residual_center)
				policy_chunk = prior_chunk + active_residual_scale * (learned_chunk - residual_center)
				mode_policy_steps = prior_steps + active_residual_scale * (mode_learned_steps - residual_center)
				mode_policy_chunk = prior_chunk + active_residual_scale * (mode_learned_chunk - residual_center)

		gate_snapshot = self._gate_snapshot()
		steps_ctrl = default_steps + schedule_gate * (policy_steps - default_steps)
		chunk_ctrl = default_chunk + schedule_gate * (policy_chunk - default_chunk)
		mode_steps_ctrl = default_steps + schedule_gate * (mode_policy_steps - default_steps)
		mode_chunk_ctrl = default_chunk + schedule_gate * (mode_policy_chunk - default_chunk)
		steps_ctrl = th.clamp(steps_ctrl, 0.0, 1.0)
		chunk_ctrl = th.clamp(chunk_ctrl, 0.0, 1.0)
		mode_steps_ctrl = th.clamp(mode_steps_ctrl, 0.0, 1.0)
		mode_chunk_ctrl = th.clamp(mode_chunk_ctrl, 0.0, 1.0)
		if not self.enable_chunk_elasticity:
			chunk_ctrl = default_chunk
			mode_chunk_ctrl = default_chunk
			log_prob_chunk = default_log_prob
		self._last_schedule_info = {
			"difficulty": difficulty.detach(),
			"difficulty_prior_u": prior_u.detach(),
			"difficulty_source_score": difficulty_source_score.detach(),
			"difficulty_prior_source_score": difficulty_prior_source_score.detach(),
			"mode_steps_ctrl": mode_steps_ctrl,
			"mode_chunk_ctrl": mode_chunk_ctrl,
			"schedule_gate": float(schedule_gate),
			"schedule_step_gate": float(schedule_step_gate),
			"schedule_residual_authority": float(prior_authority),
			"active_schedule_residual_scale": float(active_residual_scale),
			"difficulty_prior_gate": float(prior_gate),
			"difficulty_loss_gate": max(gate_snapshot["difficulty_combined_gate"], gate_snapshot["margin_floor_gate_after_range_success"]),
			"difficulty_margin_gate": gate_snapshot["margin_floor_gate_after_range_success"],
			"schedule_control_mode_id": self._mode_id(),
			**gate_snapshot,
		}
		self._record_eval_schedule(steps_ctrl, chunk_ctrl, deterministic)
		return steps_ctrl, log_prob_steps, chunk_ctrl, log_prob_chunk

	def start_eval_tracking(self) -> None:
		self._eval_tracking = True
		self._eval_nfes = []
		self._eval_steps = []
		self._eval_chunks = []
		self._eval_steps_target = []
		self._eval_chunks_target = []
		self._eval_difficulties = []
		self._eval_prior_u = []
		self._eval_source_scores = []
		self._eval_prior_source_scores = []

	def stop_eval_tracking(self) -> dict[str, float]:
		self._eval_tracking = False
		if len(self._eval_nfes) == 0:
			return {
				"avg_nfe": 0.0,
				"avg_steps": 0.0,
				"avg_chunk": 0.0,
				"avg_steps_target": 0.0,
				"avg_chunk_target": 0.0,
				"target_exec_mismatch_steps": 0.0,
				"target_exec_mismatch_chunk": 0.0,
				"avg_difficulty": 0.0,
				"avg_difficulty_abs": 0.0,
				"avg_difficulty_prior_u": 0.0,
				"avg_difficulty_prior_u_abs": 0.0,
				"avg_difficulty_source_score": 0.0,
				"std_difficulty_source_score": 0.0,
				"avg_difficulty_prior_source_score": 0.0,
				"std_difficulty_prior_source_score": 0.0,
			}
		steps_target = np.asarray(self._eval_steps_target, dtype=np.float32)
		chunk_target = np.asarray(self._eval_chunks_target, dtype=np.float32)
		steps_exec = np.asarray(self._eval_steps, dtype=np.float32)
		chunk_exec = np.asarray(self._eval_chunks, dtype=np.float32)
		return {
			"avg_nfe": float(np.mean(self._eval_nfes)),
			"avg_steps": float(np.mean(self._eval_steps)),
			"avg_chunk": float(np.mean(self._eval_chunks)),
			"avg_steps_target": float(np.mean(steps_target)) if steps_target.size > 0 else 0.0,
			"avg_chunk_target": float(np.mean(chunk_target)) if chunk_target.size > 0 else 0.0,
			"target_exec_mismatch_steps": float(np.mean(np.abs(steps_exec - steps_target))) if steps_target.size == steps_exec.size and steps_exec.size > 0 else 0.0,
			"target_exec_mismatch_chunk": float(np.mean(np.abs(chunk_exec - chunk_target))) if chunk_target.size == chunk_exec.size and chunk_exec.size > 0 else 0.0,
			"avg_difficulty": float(np.mean(self._eval_difficulties)) if len(self._eval_difficulties) > 0 else 0.0,
			"avg_difficulty_abs": float(np.mean(np.abs(self._eval_difficulties))) if len(self._eval_difficulties) > 0 else 0.0,
			"avg_difficulty_prior_u": float(np.mean(self._eval_prior_u)) if len(self._eval_prior_u) > 0 else 0.0,
			"avg_difficulty_prior_u_abs": float(np.mean(np.abs(self._eval_prior_u))) if len(self._eval_prior_u) > 0 else 0.0,
			"avg_difficulty_source_score": float(np.mean(self._eval_source_scores)) if len(self._eval_source_scores) > 0 else 0.0,
			"std_difficulty_source_score": float(np.std(self._eval_source_scores)) if len(self._eval_source_scores) > 0 else 0.0,
			"avg_difficulty_prior_source_score": float(np.mean(self._eval_prior_source_scores)) if len(self._eval_prior_source_scores) > 0 else 0.0,
			"std_difficulty_prior_source_score": float(np.std(self._eval_prior_source_scores)) if len(self._eval_prior_source_scores) > 0 else 0.0,
		}

	def update_phase_from_eval(self, success_rate: float, avg_nfe: float) -> None:
		success = float(success_rate)
		if self._eval_success_ema is None:
			self._eval_success_ema = success
		else:
			beta = self.eval_success_ema_beta
			self._eval_success_ema = beta * self._eval_success_ema + (1.0 - beta) * success
		if self._range_success_ema is None:
			self._range_success_ema = success
		else:
			beta = self.range_success_ema_beta
			self._range_success_ema = beta * self._range_success_ema + (1.0 - beta) * success
		if self._difficulty_success_ema is None:
			self._difficulty_success_ema = success
		else:
			beta = self.difficulty_success_ema_beta
			self._difficulty_success_ema = beta * self._difficulty_success_ema + (1.0 - beta) * success
		range_success_for_gate = float(self._range_success_ema)
		difficulty_success_for_gate = float(self._difficulty_success_ema)
		cost_success_for_gate = float(self._eval_success_ema)
		self._range_success_gate_raw = self._success_gate(
			range_success_for_gate,
			self.range_success_thresh_1,
			self.range_success_thresh_2,
		)
		self._difficulty_success_gate_raw = self._success_gate(
			difficulty_success_for_gate,
			self.difficulty_success_thresh_1,
			self.difficulty_success_thresh_2,
		)
		self._cost_success_gate_raw = self._success_gate(
			cost_success_for_gate,
			self.cost_success_thresh_1,
			self.cost_success_thresh_2,
		)
		prev_range_mix = self._range_success_mix
		prev_difficulty_mix = self._difficulty_success_mix
		prev_cost_mix = self._cost_success_mix
		self._range_success_mix = self._update_gate_mix(
			self._range_success_mix,
			self._range_success_gate_raw,
			open_rate=self.range_open_rate,
			close_rate=self.range_close_rate,
			no_close=self.range_success_no_close,
		)
		self._difficulty_success_mix = self._update_gate_mix(
			self._difficulty_success_mix,
			self._difficulty_success_gate_raw,
			open_rate=self.difficulty_success_open_rate,
			close_rate=self.difficulty_success_close_rate,
			no_close=self.difficulty_success_no_close,
		)
		self._cost_success_mix = self._update_gate_mix(
			self._cost_success_mix,
			self._cost_success_gate_raw,
			open_rate=self.cost_open_rate,
			close_rate=self.cost_close_rate,
			no_close=self.cost_no_rollback or self.cost_gate_mode in {"open_only_success", "success_mix"},
		)
		if self.range_success_no_close and self._range_success_mix + 1e-8 < prev_range_mix:
			self._gate_monotonic_violation_count += 1
		if self.difficulty_success_no_close and self._difficulty_success_mix + 1e-8 < prev_difficulty_mix:
			self._gate_monotonic_violation_count += 1
		if (self.cost_no_rollback or self.cost_gate_mode in {"open_only_success", "success_mix"}) and self._cost_success_mix + 1e-8 < prev_cost_mix:
			self._gate_monotonic_violation_count += 1
		_log = getattr(self, "_logger", None)
		if _log is not None:
			_log.record("budget/success_rate_eval", float(success_rate))
			_log.record("budget/avg_nfe_eval", float(avg_nfe))
			_log.record("gate/eval_success_ema", float(self._eval_success_ema))
			_log.record("gate/range_success_ema", range_success_for_gate)
			_log.record("gate/difficulty_success_ema", difficulty_success_for_gate)
			_log.record("gate/range_success_ema_beta", self.range_success_ema_beta)
			_log.record("gate/difficulty_success_ema_beta", self.difficulty_success_ema_beta)
			_log.record("gate/range_success_gate_raw", self._range_success_gate_raw)
			_log.record("gate/range_success_mix", self._range_success_mix)
			_log.record("gate/range_success_no_close", float(self.range_success_no_close))
			_log.record("gate/difficulty_success_gate_raw", self._difficulty_success_gate_raw)
			_log.record("gate/difficulty_success_mix", self._difficulty_success_mix)
			_log.record("gate/difficulty_success_open_rate", self.difficulty_success_open_rate)
			_log.record("gate/difficulty_success_close_rate", self.difficulty_success_close_rate)
			_log.record("gate/difficulty_success_no_close", float(self.difficulty_success_no_close))
			_log.record("gate/cost_gate_mode", self._cost_gate_mode_id())
			_log.record("gate/cost_progress", self._cost_progress())
			_log.record("gate/cost_success_gate_raw", self._cost_success_gate_raw)
			_log.record("gate/cost_success_mix", self._cost_success_mix)
			_log.record("gate/actor_compute_lambda_active", self._active_compute_lambda())
			_log.record("gate/gate_monotonic_violation_count", float(self._gate_monotonic_violation_count))

	def get_last_eval_schedule(self) -> dict[str, np.ndarray]:
		return {
			"steps": np.array([]) if self._last_eval_steps is None else self._last_eval_steps.copy(),
			"chunk_requested": np.array([]) if self._last_eval_chunks is None else self._last_eval_chunks.copy(),
			"steps_target": np.array([]) if self._last_eval_steps_target is None else self._last_eval_steps_target.copy(),
			"chunk_target": np.array([]) if self._last_eval_chunks_target is None else self._last_eval_chunks_target.copy(),
			"difficulty": np.array([]) if self._last_eval_difficulty is None else self._last_eval_difficulty.copy(),
			"difficulty_prior_u": np.array([]) if self._last_eval_prior_u is None else self._last_eval_prior_u.copy(),
			"difficulty_source_score": np.array([]) if self._last_eval_source_score is None else self._last_eval_source_score.copy(),
			"difficulty_prior_source_score": np.array([]) if self._last_eval_prior_source_score is None else self._last_eval_prior_source_score.copy(),
		}

	def get_last_predict_chunk_exec(self) -> Optional[np.ndarray]:
		if self._last_predict_chunk_exec is None:
			return None
		return self._last_predict_chunk_exec.copy()

	def _map_control_to_float(self, control: th.Tensor, min_value: int, max_value: int) -> th.Tensor:
		if max_value <= min_value:
			return th.full((control.shape[0],), float(min_value), device=control.device, dtype=control.dtype)
		scaled = min_value + (max_value - min_value) * control.squeeze(-1)
		return th.clamp(scaled, min=float(min_value), max=float(max_value))

	def _map_control_to_int(self, control: th.Tensor, min_value: int, max_value: int, stochastic: bool = False) -> th.Tensor:
		if max_value <= min_value:
			return th.full((control.shape[0],), int(min_value), device=control.device, dtype=th.int64)
		scaled = self._map_control_to_float(control, min_value, max_value)
		if not stochastic:
			return th.floor(scaled + 0.5).to(dtype=th.int64)
		low = th.floor(scaled)
		high = th.ceil(scaled)
		prob_high = scaled - low
		rand = th.rand_like(prob_high)
		mapped = th.where(rand < prob_high, high, low)
		return mapped.to(dtype=th.int64)

	def _actor_compute_cost(self, steps_ctrl: th.Tensor, chunk_ctrl: th.Tensor) -> th.Tensor:
		"""Differentiable NFE proxy D_cont / C_cont (same linear map as discrete schedule, no round)."""
		batch = steps_ctrl.shape[0]
		d_scale = self.max_denoising_steps - self.min_denoising_steps
		if d_scale <= 0:
			d_cont = th.full((batch,), float(self.min_denoising_steps), device=steps_ctrl.device, dtype=steps_ctrl.dtype)
		else:
			d_cont = self.min_denoising_steps + d_scale * steps_ctrl.squeeze(-1)
		c_scale = self.max_chunk_size - self.min_chunk_size
		if c_scale <= 0:
			c_cont = th.full((batch,), float(self.min_chunk_size), device=chunk_ctrl.device, dtype=chunk_ctrl.dtype)
		else:
			c_cont = self.min_chunk_size + c_scale * chunk_ctrl.squeeze(-1)
		return d_cont / th.clamp(c_cont, min=1.0)

	def _apply_diffusion(
		self,
		obs: th.Tensor,
		noise_actions: th.Tensor,
		steps_ctrl: Optional[th.Tensor] = None,
		chunk_ctrl: Optional[th.Tensor] = None,
		stochastic_rounding: Optional[bool] = None,
		mapped_steps: Optional[th.Tensor] = None,
		mapped_chunk: Optional[th.Tensor] = None,
	) -> th.Tensor:
		num_steps = mapped_steps
		chunk_size = mapped_chunk
		use_stochastic_rounding = self.stochastic_rounding if stochastic_rounding is None else bool(stochastic_rounding)
		if num_steps is None and steps_ctrl is not None:
			num_steps = self._map_control_to_int(steps_ctrl, self.min_denoising_steps, self.max_denoising_steps, stochastic=use_stochastic_rounding)
		if chunk_size is None and chunk_ctrl is not None:
			chunk_size = self._map_control_to_int(chunk_ctrl, self.min_chunk_size, self.max_chunk_size, stochastic=use_stochastic_rounding)
		try:
			return self.diffusion_policy(
				obs,
				noise_actions,
				return_numpy=False,
				num_steps=num_steps,
				chunk_size=chunk_size,
			)
		except TypeError as exc:
			# Backward compatibility only: older wrappers may not accept elastic kwargs.
			# TypeErrors raised inside the sampler must surface instead of silently disabling elastic steps.
			message = str(exc)
			if "num_steps" in message or "chunk_size" in message:
				return self.diffusion_policy(obs, noise_actions, return_numpy=False)
			raise

	def _create_aliases(self) -> None:
		self.actor = self.policy.actor
		self.critic = self.policy.critic
		self.critic_target = self.policy.critic_target

	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		# Switch to train mode (this affects batch norm / dropout)
		self.policy.set_training_mode(True)
		self.critic_noise.set_training_mode(True)
		train_scheduling = self._schedule_training_active()
		if self.enable_three_head:
			assert self.steps_mu is not None and self.steps_log_std is not None
			assert self.chunk_mu is not None and self.chunk_log_std is not None
			self.steps_mu.train()
			self.steps_log_std.train()
			self.chunk_mu.train()
			self.chunk_log_std.train()
		# Update optimizers learning rate
		optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_noise.optimizer]
		if self.enable_three_head and self.schedule_head_optimizer is not None:
			optimizers += [self.schedule_head_optimizer]
		if self.ent_coef_optimizer is not None:
			optimizers += [self.ent_coef_optimizer]

		# Update learning rate according to lr schedule
		self._update_learning_rate(optimizers)

		ent_coef_losses, ent_coefs = [], []
		actor_losses, critic_losses, noise_critic_losses = [], [], []
		actor_compute_cost_batch: list[float] = []
		difficulty_abs_batch: list[float] = []
		difficulty_prior_abs_batch: list[float] = []
		difficulty_score_mean_batch: list[float] = []
		difficulty_score_std_batch: list[float] = []
		difficulty_prior_score_mean_batch: list[float] = []
		difficulty_prior_score_std_batch: list[float] = []
		difficulty_losses: list[float] = []
		schedule_gates: list[float] = []
		prior_gates: list[float] = []
		step_prior_authorities: list[float] = []
		denoise_steps_mean, chunk_size_mean = [], []
		denoise_steps_target_mean, chunk_size_target_mean = [], []
		target_exec_mismatch_steps, target_exec_mismatch_chunk = [], []

		if self.actor_gradient_steps < 0:
			actor_gradient_idx = np.linspace(0, gradient_steps - 1, gradient_steps, dtype=int)
		else:
			actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps - 1, self.actor_gradient_steps, dtype=int)

		for gradient_step in range(gradient_steps):
			# Sample replay buffer
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

			# We need to sample because `log_std` may have changed between two gradient steps
			if self.use_sde:
				self.actor.reset_noise()

			# Action by the current actor for the sampled state
			actions_pi, log_prob_w = self.actor.action_log_prob(replay_data.observations)
			log_prob_w = log_prob_w.reshape(-1, 1)
			steps_ctrl, log_prob_steps, chunk_ctrl, log_prob_chunk = self._sample_schedule_controls(
				replay_data.observations,
				deterministic=False,
				train_scheduling=train_scheduling,
				w_noise=actions_pi.detach(),
			)
			schedule_info = dict(self._last_schedule_info)
			steps_actor, chunk_actor = steps_ctrl, chunk_ctrl
			log_prob_sched = log_prob_steps + log_prob_chunk
			log_prob = log_prob_w + log_prob_sched
			if self.enable_three_head:
				step_targets = self._map_control_to_float(steps_ctrl, self.min_denoising_steps, self.max_denoising_steps).to(dtype=th.float32)
				chunk_targets = self._map_control_to_float(chunk_ctrl, self.min_chunk_size, self.max_chunk_size).to(dtype=th.float32)
				step_vals = self._map_control_to_int(
					steps_ctrl,
					self.min_denoising_steps,
					self.max_denoising_steps,
					stochastic=self.stochastic_rounding,
				).to(dtype=th.float32)
				chunk_vals = self._map_control_to_int(
					chunk_ctrl,
					self.min_chunk_size,
					self.max_chunk_size,
					stochastic=self.stochastic_rounding,
				).to(dtype=th.float32)
				denoise_steps_mean.append(step_vals.mean().item())
				chunk_size_mean.append(chunk_vals.mean().item())
				denoise_steps_target_mean.append(step_targets.mean().item())
				chunk_size_target_mean.append(chunk_targets.mean().item())
				target_exec_mismatch_steps.append((step_vals - step_targets).abs().mean().item())
				target_exec_mismatch_chunk.append((chunk_vals - chunk_targets).abs().mean().item())

			ent_coef_loss = None
			if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
				# Important: detach the variable from the graph
				# so we don't change it with other losses
				# see https://github.com/rail-berkeley/softlearning/issues/60
				ent_coef = th.exp(self.log_ent_coef.detach())
				schedule_entropy_active = self.enable_three_head and train_scheduling and self.schedule_control_mode != "prior_only"
				ent_target = self._ent_coef_target_joint if schedule_entropy_active else self._ent_coef_target_w_only
				ent_coef_loss = -(self.log_ent_coef * (log_prob + ent_target).detach()).mean()
				ent_coef_losses.append(ent_coef_loss.item())
			else:
				ent_coef = self.ent_coef_tensor

			ent_coefs.append(ent_coef.item())

			# Optimize entropy coefficient, also called entropy temperature or alpha in the paper
			if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
				self.ent_coef_optimizer.zero_grad()
				ent_coef_loss.backward()
				self.ent_coef_optimizer.step()

			with th.no_grad():
				# Select action according to policy
				next_actions, next_log_prob_w = self.actor.action_log_prob(replay_data.next_observations)
				next_steps_ctrl, next_log_prob_steps, next_chunk_ctrl, next_log_prob_chunk = self._sample_schedule_controls(
					replay_data.next_observations,
					deterministic=False,
					train_scheduling=train_scheduling,
					w_noise=next_actions.detach(),
				)
				next_log_prob_joint = next_log_prob_w.reshape(-1, 1) + next_log_prob_steps + next_log_prob_chunk
				# Q^A backup: same joint entropy as Actor / ent_coef.
				next_actions = th.tensor(self.policy.unscale_action(next_actions.cpu().numpy())).to(self.device)
				next_actions = self._apply_diffusion(
					replay_data.next_observations,
					next_actions.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
					steps_ctrl=next_steps_ctrl,
					chunk_ctrl=next_chunk_ctrl,
				)
				next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
				# Compute the next Q values: min over all critics targets
				next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
				if self.critic_backup_combine_type == 'min':
					next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
				elif self.critic_backup_combine_type == 'mean':
					next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
				next_q_values = next_q_values - ent_coef * next_log_prob_joint
				if getattr(replay_data, "chunk_exec", None) is None:
					chunk_exec = th.ones_like(replay_data.rewards)
				else:
					chunk_exec = th.clamp(replay_data.chunk_exec.to(device=self.device, dtype=replay_data.rewards.dtype), min=1.0)
				macro_discount = th.pow(th.full_like(chunk_exec, float(self.gamma)), chunk_exec)
				target_q_values = replay_data.rewards + (1 - replay_data.dones) * macro_discount * next_q_values

			# Get current Q-values estimates for each critic network using action from the replay buffer
			current_q_values = self.critic(replay_data.observations, replay_data.actions)

			# Compute critic loss
			critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
			assert isinstance(critic_loss, th.Tensor)  # for type checker
			critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()

			if gradient_step in actor_gradient_idx:
				# Compute actor loss
				noise_input = actions_pi
				if self.enable_three_head:
					noise_input = th.cat([noise_input, steps_actor, chunk_actor], dim=1)
				q_values_pi = th.cat(self.critic_noise(replay_data.observations, noise_input), dim=1)
				if self.critic_backup_combine_type == 'min':
					min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
				elif self.critic_backup_combine_type == 'mean':
					min_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
				actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
				if self.enable_three_head:
					awc_cost = self._actor_compute_cost(steps_actor, chunk_actor).reshape(-1, 1)
					actor_compute_cost_batch.append(float(awc_cost.mean().item()))
					difficulty = schedule_info.get("difficulty")
					if difficulty is not None:
						difficulty = difficulty.detach().to(device=steps_actor.device, dtype=steps_actor.dtype)

						mode_steps_actor = schedule_info.get("mode_steps_ctrl")
						mode_chunk_actor = schedule_info.get("mode_chunk_ctrl")
						if mode_steps_actor is not None and mode_chunk_actor is not None:
							mode_steps_actor = mode_steps_actor.to(device=steps_actor.device, dtype=steps_actor.dtype)
							mode_chunk_actor = mode_chunk_actor.to(device=steps_actor.device, dtype=steps_actor.dtype)
						else:
							mode_steps_actor = None
							mode_chunk_actor = None
						diff_loss, _ = self._difficulty_alignment_loss(steps_actor, chunk_actor, difficulty, mode_steps_actor, mode_chunk_actor)
						if diff_loss.requires_grad or diff_loss.item() != 0.0:
							actor_loss = actor_loss + diff_loss
						difficulty_losses.append(float(diff_loss.detach().item()))
						difficulty_abs_batch.append(float(difficulty.detach().abs().mean().item()))
					prior_u = schedule_info.get("difficulty_prior_u")
					if prior_u is not None:
						difficulty_prior_abs_batch.append(float(prior_u.detach().abs().mean().item()))
					difficulty_score = schedule_info.get("difficulty_source_score")
					if difficulty_score is not None:
						difficulty_score = difficulty_score.detach()
						difficulty_score_mean_batch.append(float(difficulty_score.mean().item()))
						difficulty_score_std_batch.append(float(difficulty_score.std(unbiased=False).item()))
					difficulty_prior_score = schedule_info.get("difficulty_prior_source_score")
					if difficulty_prior_score is not None:
						difficulty_prior_score = difficulty_prior_score.detach()
						difficulty_prior_score_mean_batch.append(float(difficulty_prior_score.mean().item()))
						difficulty_prior_score_std_batch.append(float(difficulty_prior_score.std(unbiased=False).item()))
					schedule_gates.append(float(schedule_info.get("schedule_gate", 0.0)))
					prior_gates.append(float(schedule_info.get("difficulty_prior_gate", 0.0)))
					step_prior_authorities.append(float(schedule_info.get("schedule_residual_authority", 0.0)) * self.difficulty_prior_scale)
					if self.schedule_entropy_weight > 0.0 and train_scheduling and self.schedule_control_mode != "prior_only":
						schedule_entropy = -(log_prob_steps + log_prob_chunk).mean()
						actor_loss = actor_loss - self.schedule_entropy_weight * schedule_entropy
				actor_losses.append(actor_loss.item())

				# Optimize the actor
				self.actor.optimizer.zero_grad()
				if self.enable_three_head and self.schedule_head_optimizer is not None:
					self.schedule_head_optimizer.zero_grad()
				actor_loss.backward()
				self.actor.optimizer.step()
				if self.enable_three_head and self.schedule_head_optimizer is not None and train_scheduling:
					self.schedule_head_optimizer.step()

			# Update target networks
			if gradient_step % self.target_update_interval == 0:
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
				# Copy running stats, see GH issue #996
				polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

		for gradient_step in range(self.noise_critic_grad_steps):
			# Sample replay buffer
			critic_distill_loss = 0
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
			critic_distill_loss = critic_distill_loss + self.update_noise_critic(replay_data)
			noise_critic_losses.append(critic_distill_loss.item())
			self.critic_noise.optimizer.zero_grad()
			critic_distill_loss.backward()
			self.critic_noise.optimizer.step()

		self.critic_noise.set_training_mode(False)
		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/ent_coef", np.mean(ent_coefs))
		self.logger.record("train/actor_loss", np.mean(actor_losses))
		self.logger.record("train/critic_loss", np.mean(critic_losses))
		self.logger.record("train/noise_critic_loss", np.mean(noise_critic_losses))
		if len(actor_compute_cost_batch) > 0:
			self.logger.record("train/actor_compute_cost_mean", np.mean(actor_compute_cost_batch))
		if self.enable_three_head and len(denoise_steps_mean) > 0:
			self.logger.record("train/denoising_steps_mean", np.mean(denoise_steps_mean))
			self.logger.record("train/chunk_size_mean", np.mean(chunk_size_mean))
			self.logger.record("train/denoising_steps_target_mean", np.mean(denoise_steps_target_mean))
			self.logger.record("train/chunk_size_target_mean", np.mean(chunk_size_target_mean))
			self.logger.record("train/target_exec_mismatch_steps", np.mean(target_exec_mismatch_steps))
			self.logger.record("train/target_exec_mismatch_chunk", np.mean(target_exec_mismatch_chunk))
			if len(schedule_gates) > 0:
				self.logger.record("train/schedule_gate", float(np.mean(schedule_gates)))
			if len(prior_gates) > 0:
				self.logger.record("train/effective_difficulty_prior_scale", self.difficulty_prior_scale * float(np.mean(prior_gates)))
			if len(step_prior_authorities) > 0:
				self.logger.record("train/effective_step_prior_authority", float(np.mean(step_prior_authorities)))
		if len(difficulty_abs_batch) > 0:
			self.logger.record("train/difficulty_abs_mean", np.mean(difficulty_abs_batch))
		if len(difficulty_prior_abs_batch) > 0:
			self.logger.record("train/difficulty_prior_u_abs_mean", np.mean(difficulty_prior_abs_batch))
		if len(difficulty_score_std_batch) > 0:
			self.logger.record("train/difficulty_source_score_mean", np.mean(difficulty_score_mean_batch))
			self.logger.record("train/difficulty_source_score_std", np.mean(difficulty_score_std_batch))
			self.logger.record("train/compute_advantage_std", np.mean(difficulty_score_std_batch))
		if len(difficulty_prior_score_std_batch) > 0:
			self.logger.record("train/difficulty_prior_source_score_mean", np.mean(difficulty_prior_score_mean_batch))
			self.logger.record("train/difficulty_prior_source_score_std", np.mean(difficulty_prior_score_std_batch))
		if len(difficulty_losses) > 0:
			self.logger.record("train/difficulty_loss", np.mean(difficulty_losses))
		gate_snapshot = self._gate_snapshot()
		self.logger.record("train/range_alpha", gate_snapshot["range_alpha"])
		self.logger.record("train/difficulty_success_mix", gate_snapshot["difficulty_success_mix"])
		self.logger.record("train/actor_compute_lambda_active", gate_snapshot["actor_compute_lambda_active"])
		self.logger.record("train/gate_timesteps", gate_snapshot["gate_timesteps"])
		if len(self._episode_budget_ratios) > 0:
			ratios = np.asarray(self._episode_budget_ratios, dtype=np.float32)
			self.logger.record("train/episode_budget_ratio", float(np.mean(ratios)))
			self.logger.record("train/episode_budget_penalty", float(np.mean(self._episode_budget_penalty_totals)))
			self.logger.record("train/episode_budget_success_rate", float(np.mean(self._episode_budget_successes)))
			self.logger.record("train/episode_budget_saving_active_rate", float(np.mean(self._episode_budget_saving_active)))
			self._episode_budget_ratios = []
			self._episode_budget_unders = []
			self._episode_budget_overs = []
			self._episode_budget_penalty_totals = []
			self._episode_budget_savings = []
			self._episode_budget_saving_bonuses = []
			self._episode_budget_gross_penalties = []
			self._episode_budget_successes = []
			self._episode_budget_saving_active = []
		if len(ent_coef_losses) > 0:
			self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


	def update_noise_critic(self, replay_data):
		# Distill Q^w toward Q^A. For elastic scheduling, critic_noise must cover the
		# schedule points used by compute_advantage, not only the actor's current schedule.
		obs = replay_data.observations
		batch = obs.shape[0]
		train_scheduling = self._schedule_training_active()

		def _scaled_to_noise_tensor(w_scaled: th.Tensor) -> th.Tensor:
			w_unscaled = th.as_tensor(
				self.policy.unscale_action(w_scaled.detach().cpu().numpy()),
				device=self.device,
				dtype=w_scaled.dtype,
			)
			return w_unscaled.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim)

		def _random_scaled_noise(dtype: th.dtype) -> th.Tensor:
			noise_actions = th.randn(batch, self.diffusion_act_chunk, self.diffusion_act_dim, device=self.device, dtype=dtype)
			noise_flat = noise_actions.reshape(batch, self.diffusion_act_chunk * self.diffusion_act_dim)
			return th.as_tensor(self.policy.scale_action(noise_flat.detach().cpu().numpy()), device=self.device, dtype=dtype)

		def _distill_case(
			w_scaled: th.Tensor,
			steps_ctrl: Optional[th.Tensor] = None,
			chunk_ctrl: Optional[th.Tensor] = None,
		) -> th.Tensor:
			with th.no_grad():
				diffused_actions = self._apply_diffusion(
					obs,
					_scaled_to_noise_tensor(w_scaled),
					steps_ctrl=steps_ctrl,
					chunk_ctrl=chunk_ctrl,
				)
				diffused_actions = diffused_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
				current_q_values = self.critic(obs, diffused_actions)
			noise_input = w_scaled.detach()
			if self.enable_three_head:
				assert steps_ctrl is not None and chunk_ctrl is not None
				noise_input = th.cat([noise_input, steps_ctrl.detach(), chunk_ctrl.detach()], dim=1)
			current_q_noise_vals = self.critic_noise(obs, noise_input)
			case_loss = 0
			for i in range(len(current_q_values)):
				case_loss = case_loss + 0.5 * F.mse_loss(current_q_values[i].detach(), current_q_noise_vals[i])
			return case_loss

		with th.no_grad():
			w_pi, _ = self.actor.action_log_prob(obs)
			w_pi = w_pi.detach()

		if not self.enable_three_head:
			# Match official DSRL: train Q^w on broad random latent noise, not only actor samples.
			return _distill_case(_random_scaled_noise(w_pi.dtype))

		with th.no_grad():
			steps_ctrl, _, chunk_ctrl, _ = self._sample_schedule_controls(
				obs,
				deterministic=False,
				train_scheduling=train_scheduling,
				w_noise=w_pi,
			)
			easy_steps = self._control_from_value(
				self.difficulty_easy_steps_target,
				self.min_denoising_steps,
				self.max_denoising_steps,
				batch,
				device=obs.device,
				dtype=w_pi.dtype,
			)
			hard_steps = self._control_from_value(
				self.difficulty_hard_steps_target,
				self.min_denoising_steps,
				self.max_denoising_steps,
				batch,
				device=obs.device,
				dtype=w_pi.dtype,
			)
			easy_chunk = self._control_from_value(
				self.difficulty_easy_chunk_target,
				self.min_chunk_size,
				self.max_chunk_size,
				batch,
				device=obs.device,
				dtype=w_pi.dtype,
			)
			hard_chunk = self._control_from_value(
				self.difficulty_hard_chunk_target,
				self.min_chunk_size,
				self.max_chunk_size,
				batch,
				device=obs.device,
				dtype=w_pi.dtype,
			)
			default_steps = self._control_from_fixed(
				self.fixed_denoising_steps,
				self.min_denoising_steps,
				self.max_denoising_steps,
				batch,
			).to(device=obs.device, dtype=w_pi.dtype)
			default_chunk = self._control_from_fixed(
				self.fixed_chunk_size,
				self.min_chunk_size,
				self.max_chunk_size,
				batch,
			).to(device=obs.device, dtype=w_pi.dtype)

		losses = [
			_distill_case(w_pi, steps_ctrl, chunk_ctrl),
			_distill_case(w_pi, hard_steps, hard_chunk),
			_distill_case(w_pi, easy_steps, easy_chunk),
			_distill_case(_random_scaled_noise(w_pi.dtype), default_steps, default_chunk),
		]
		return sum(losses) / float(len(losses))


	def _set_last_rollout_schedule(self, mapped_steps: th.Tensor, mapped_chunk: th.Tensor) -> None:
		steps = mapped_steps.detach().cpu().numpy().astype(np.float32).reshape(-1)
		chunks = mapped_chunk.detach().cpu().numpy().astype(np.float32).reshape(-1)
		self._last_rollout_steps = steps
		self._last_rollout_chunks = chunks
		self._last_rollout_nfe = steps / np.maximum(chunks, 1.0)
		info = self._last_schedule_info
		prior_u = info.get("difficulty_prior_u")
		if prior_u is None:
			self._last_rollout_prior_u = np.zeros_like(self._last_rollout_nfe, dtype=np.float32)
		else:
			self._last_rollout_prior_u = prior_u.detach().cpu().numpy().astype(np.float32).reshape(-1)

	def _set_env_pending_chunk_exec(self, mapped_chunk: th.Tensor, env: Optional[Any] = None) -> None:
		chunks = mapped_chunk.detach().cpu().numpy().astype(np.int64).reshape(-1)
		self._last_predict_chunk_exec = chunks.astype(np.float32)
		target_env = self.env if env is None else env
		if target_env is None or not hasattr(target_env, "set_attr"):
			return
		n_envs = int(getattr(target_env, "num_envs", len(chunks)))
		for idx, chunk in enumerate(chunks[:n_envs]):
			try:
				target_env.set_attr("pending_chunk_exec", int(max(1, chunk)), indices=[idx])
			except Exception:
				break

	def _ensure_episode_budget_state(self, n_envs: int) -> None:
		if (
			self._episode_budget_positions is not None
			and len(self._episode_budget_positions) == n_envs
			and self._episode_budget_steps is not None
			and len(self._episode_budget_steps) == n_envs
			and self._episode_budget_chunks is not None
			and len(self._episode_budget_chunks) == n_envs
		):
			return
		self._episode_budget_positions = [[] for _ in range(n_envs)]
		self._episode_budget_steps = [[] for _ in range(n_envs)]
		self._episode_budget_chunks = [[] for _ in range(n_envs)]

	def _episode_band_penalty_np(
		self,
		total_steps: float,
		total_chunks: float,
		success: bool,
	) -> tuple[float, float, float, float, float, float, float]:
		chunks = max(float(total_chunks), 1.0)
		steps = float(total_steps)
		ratio = steps / chunks
		over = max(0.0, steps - self.nfe_target_upper * chunks - self.nfe_debt_limit)
		under = max(0.0, self.nfe_target_lower * chunks - steps)
		under_penalty = self.nfe_under_weight * under if success else 0.0
		gross_penalty = (over + under_penalty) / self.nfe_budget_penalty_scale
		inside_band = self.nfe_target_lower <= ratio <= self.nfe_target_upper
		if success and inside_band and self.nfe_target_upper > self.nfe_target_lower:
			saving_norm = (self.nfe_target_upper - ratio) / max(1e-6, self.nfe_target_upper - self.nfe_target_lower)
			saving_norm = float(np.clip(saving_norm, 0.0, 1.0))
		else:
			saving_norm = 0.0
		saving_bonus = self.nfe_saving_weight * saving_norm
		net_penalty = gross_penalty - saving_bonus
		return ratio, under / chunks, over / chunks, gross_penalty, saving_norm, saving_bonus, net_penalty

	def _annotate_rollout_infos(self, infos: list[dict[str, Any]]) -> None:
		if self._last_rollout_steps is None or self._last_rollout_chunks is None:
			return
		n_envs = min(len(infos), len(self._last_rollout_steps), len(self._last_rollout_chunks))
		for i in range(n_envs):
			steps = float(self._last_rollout_steps[i])
			chunk_requested = float(max(1.0, self._last_rollout_chunks[i]))
			chunk_exec = float(max(1.0, infos[i].get("chunk_exec", chunk_requested)))
			infos[i]["rollout_steps"] = steps
			infos[i]["rollout_chunk_requested"] = chunk_requested
			infos[i]["rollout_chunk_exec"] = chunk_exec
			infos[i]["rollout_nfe_requested"] = steps / max(chunk_requested, 1.0)
			infos[i]["rollout_nfe_executed"] = steps / max(chunk_exec, 1.0)

	def _apply_episode_band_budget(
		self,
		replay_buffer: ReplayBuffer,
		buffer_pos: int,
		dones: np.ndarray,
		infos: list[dict[str, Any]],
	) -> None:
		coef = self._rollout_nfe_penalty_coef()
		n_envs = len(infos)
		self._ensure_episode_budget_state(n_envs)
		if self._episode_budget_positions is None or self._episode_budget_steps is None or self._episode_budget_chunks is None:
			return
		for env_idx in range(n_envs):
			steps = float(infos[env_idx].get("rollout_steps", 0.0))
			chunk_requested = float(max(1.0, infos[env_idx].get(
				"rollout_chunk_requested",
				infos[env_idx].get("chunk_requested", infos[env_idx].get("rollout_chunk_exec", 1.0)),
			)))
			self._episode_budget_positions[env_idx].append((buffer_pos, env_idx))
			self._episode_budget_steps[env_idx].append(steps)
			self._episode_budget_chunks[env_idx].append(chunk_requested)
			if not bool(np.asarray(dones).reshape(-1)[env_idx]):
				continue
			total_steps = float(np.sum(self._episode_budget_steps[env_idx]))
			total_chunks = float(np.sum(self._episode_budget_chunks[env_idx]))
			success_signal = float(infos[env_idx].get("chunk_success_signal", -np.inf))
			success = bool(success_signal > self.episode_success_threshold)
			ratio, under, over, gross_penalty, saving, saving_bonus, net_penalty = self._episode_band_penalty_np(total_steps, total_chunks, success)
			positions = self._episode_budget_positions[env_idx]
			per_query_penalty = coef * net_penalty / max(1, len(positions))
			if per_query_penalty != 0.0:
				for pos, pos_env_idx in positions:
					replay_buffer.rewards[pos, pos_env_idx] -= np.float32(per_query_penalty)
			infos[env_idx]["episode_budget_ratio"] = float(ratio)
			infos[env_idx]["episode_budget_under"] = float(under)
			infos[env_idx]["episode_budget_over"] = float(over)
			infos[env_idx]["episode_budget_gross_penalty"] = float(gross_penalty)
			infos[env_idx]["episode_budget_saving"] = float(saving)
			infos[env_idx]["episode_budget_saving_bonus"] = float(saving_bonus)
			infos[env_idx]["episode_budget_penalty"] = float(net_penalty)
			infos[env_idx]["episode_budget_success"] = float(success)
			self._episode_budget_ratios.append(float(ratio))
			self._episode_budget_unders.append(float(under))
			self._episode_budget_overs.append(float(over))
			self._episode_budget_penalty_totals.append(float(net_penalty))
			self._episode_budget_savings.append(float(saving))
			self._episode_budget_saving_bonuses.append(float(saving_bonus))
			self._episode_budget_gross_penalties.append(float(gross_penalty))
			self._episode_budget_successes.append(float(success))
			self._episode_budget_saving_active.append(float(saving_bonus > 0.0))
			self._episode_budget_positions[env_idx] = []
			self._episode_budget_steps[env_idx] = []
			self._episode_budget_chunks[env_idx] = []

	def _store_transition(
		self,
		replay_buffer: ReplayBuffer,
		buffer_action: np.ndarray,
		new_obs: Union[np.ndarray, dict[str, np.ndarray]],
		reward: np.ndarray,
		dones: np.ndarray,
		infos: list[dict[str, Any]],
	) -> None:
		if self._vec_normalize_env is not None:
			new_obs_ = self._vec_normalize_env.get_original_obs()
			reward_ = self._vec_normalize_env.get_original_reward()
		else:
			self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
		self._annotate_rollout_infos(infos)
		self._elapsed_env_steps += float(np.sum([
			max(1.0, float(info.get("chunk_exec", 1.0)))
			for info in infos
		]))
		next_obs = deepcopy(new_obs_)
		for i, done in enumerate(dones):
			if done and infos[i].get("terminal_observation") is not None:
				if isinstance(next_obs, dict):
					next_obs_ = infos[i]["terminal_observation"]
					if self._vec_normalize_env is not None:
						next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
					for key in next_obs.keys():
						next_obs[key][i] = next_obs_[key]
				else:
					next_obs[i] = infos[i]["terminal_observation"]
					if self._vec_normalize_env is not None:
						next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]
		buffer_pos = replay_buffer.pos
		replay_buffer.add(
			self._last_original_obs,  # type: ignore[arg-type]
			next_obs,  # type: ignore[arg-type]
			buffer_action,
			reward_,
			dones,
			infos,
		)
		self._apply_episode_band_budget(replay_buffer, buffer_pos, dones, infos)
		self._last_obs = new_obs
		if self._vec_normalize_env is not None:
			self._last_original_obs = new_obs_


	def learn(
		self: SelfDSRL,
		total_timesteps: int,
		callback: MaybeCallback = None,
		log_interval: int = 4,
		tb_log_name: str = "SAC",
		reset_num_timesteps: bool = True,
		progress_bar: bool = False,
	) -> SelfDSRL:
		return super().learn(
			total_timesteps=total_timesteps,
			callback=callback,
			log_interval=log_interval,
			tb_log_name=tb_log_name,
			reset_num_timesteps=reset_num_timesteps,
			progress_bar=progress_bar,
		)

	def _excluded_save_params(self) -> list[str]:
		return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

	def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
		state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "critic_noise"]
		if self.enable_three_head:
			state_dicts += ["steps_mu", "steps_log_std", "chunk_mu", "chunk_log_std"]
			if self.schedule_head_optimizer is not None:
				state_dicts += ["schedule_head_optimizer"]
		if self.ent_coef_optimizer is not None:
			saved_pytorch_variables = ["log_ent_coef"]
			state_dicts.append("ent_coef_optimizer")
		else:
			saved_pytorch_variables = ["ent_coef_tensor"]
		# import pdb; pdb.set_trace()
		return state_dicts, saved_pytorch_variables
	
	def _sample_action(
		self,
		learning_starts: int,
		action_noise: Optional[ActionNoise] = None,
		n_envs: int = 1,
	) -> tuple[np.ndarray, np.ndarray]:
		"""
		Sample an action according to the exploration policy.
		This is either done by sampling the probability distribution of the policy,
		or sampling a random action (from a uniform distribution over the action space)
		or by adding noise to the deterministic output.

		:param action_noise: Action noise that will be used for exploration
			Required for deterministic policy (e.g. TD3). This can also be used
			in addition to the stochastic policy for SAC.
		:param learning_starts: Number of steps before learning for the warm-up phase.
		:param n_envs:
		:return: action to take in the environment
			and scaled action that will be stored in the replay buffer.
			The two differs when the action space is not normalized (bounds are not [-1, 1]).
		"""
		# Select action randomly or according to policy
		if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
			# Warmup phase
			unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
		else:
			# Note: when using continuous actions,
			# we assume that the policy uses tanh to scale the action
			# We use non-deterministic action in the case of SAC, for TD3, it does not matter
			assert self._last_obs is not None, "self._last_obs was not set"
			unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

		# Rescale the action from [low, high] to [-1, 1]
		if isinstance(self.action_space, spaces.Box):
			scaled_action = self.policy.scale_action(unscaled_action)

			# Add noise to the action (improve exploration)
			if action_noise is not None:
				scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

			# We store the scaled action in the buffer
			buffer_action = scaled_action
			action = self.policy.unscale_action(scaled_action)
		else:
			# Discrete case, no need to normalize or clip
			buffer_action = unscaled_action
			action = buffer_action
		action = th.as_tensor(action, device=self.device, dtype=th.float32)
		if isinstance(self.action_space, spaces.Box):
			schedule_noise = th.as_tensor(scaled_action, device=self.device, dtype=th.float32)
		else:
			schedule_noise = action
		schedule_noise = schedule_noise.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
		obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
		steps_ctrl, _, chunk_ctrl, _ = self._sample_schedule_controls(
			obs,
			deterministic=False,
			train_scheduling=self._schedule_training_active(),
			w_noise=schedule_noise,
		)
		mapped_steps = self._map_control_to_int(
			steps_ctrl,
			self.min_denoising_steps,
			self.max_denoising_steps,
			stochastic=self.stochastic_rounding,
		)
		mapped_chunk = self._map_control_to_int(
			chunk_ctrl,
			self.min_chunk_size,
			self.max_chunk_size,
			stochastic=self.stochastic_rounding,
		)
		self._set_last_rollout_schedule(mapped_steps, mapped_chunk)
		self._set_env_pending_chunk_exec(mapped_chunk, self.env)
		action = self._apply_diffusion(
			obs,
			action.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
			steps_ctrl=steps_ctrl,
			chunk_ctrl=chunk_ctrl,
			mapped_steps=mapped_steps,
			mapped_chunk=mapped_chunk,
		)
		action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
		action = action.cpu().numpy()
		buffer_action = action
		return action, buffer_action

	def predict_diffused(
		self,
		observation: Union[np.ndarray, dict[str, np.ndarray]],
		state: Optional[tuple[np.ndarray, ...]] = None,
		episode_start: Optional[np.ndarray] = None,
		deterministic: bool = False,
	) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
		unscaled_action, predict_second_return = self.policy.predict(observation, state, episode_start, deterministic)
		if isinstance(self.action_space, spaces.Box):
			scaled_action = self.policy.scale_action(unscaled_action)
			# We store the scaled action in the buffer
			buffer_action = scaled_action
			action = self.policy.unscale_action(scaled_action)
		else:
			# Discrete case, no need to normalize or clip
			buffer_action = unscaled_action
			action = buffer_action
		action = th.as_tensor(action, device=self.device, dtype=th.float32)
		if isinstance(self.action_space, spaces.Box):
			schedule_noise = th.as_tensor(scaled_action, device=self.device, dtype=th.float32)
		else:
			schedule_noise = action
		schedule_noise = schedule_noise.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
		obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
		steps_ctrl, _, chunk_ctrl, _ = self._sample_schedule_controls(
			obs,
			deterministic=deterministic,
			train_scheduling=self._schedule_training_active(),
			w_noise=schedule_noise,
		)
		mapped_steps = self._map_control_to_int(
			steps_ctrl,
			self.min_denoising_steps,
			self.max_denoising_steps,
			stochastic=self.stochastic_rounding and (not deterministic),
		)
		mapped_chunk = self._map_control_to_int(
			chunk_ctrl,
			self.min_chunk_size,
			self.max_chunk_size,
			stochastic=self.stochastic_rounding and (not deterministic),
		)
		self._last_predict_chunk_exec = mapped_chunk.detach().cpu().numpy().astype(np.float32).reshape(-1)
		action = self._apply_diffusion(
			obs,
			action.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
			steps_ctrl=steps_ctrl,
			chunk_ctrl=chunk_ctrl,
			stochastic_rounding=self.stochastic_rounding and (not deterministic),
			mapped_steps=mapped_steps,
			mapped_chunk=mapped_chunk,
		)
		action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
		action = action.cpu().numpy()
		return action, predict_second_return
