import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True, num_steps=None, chunk_size=None):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(
				cond=cond,
				deterministic=True,
				num_steps=num_steps,
				chunk_size=chunk_size,
			)
		diffused_actions = (samples.trajectories.detach())
		diffused_actions = self._apply_chunk_control(diffused_actions, chunk_size)
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	

	def _apply_chunk_control(self, diffused_actions, chunk_size):
		"""
		Keep fixed horizon for compatibility while letting chunk_size affect execution.
		For each sample, actions after chunk_size repeat the last action in the chunk.
		"""
		if chunk_size is None:
			return diffused_actions
		if not torch.is_tensor(chunk_size):
			chunk_size = torch.as_tensor(chunk_size, device=diffused_actions.device)
		chunk_size = chunk_size.to(device=diffused_actions.device).reshape(-1).to(dtype=torch.long)
		horizon = diffused_actions.shape[1]
		chunk_size = torch.clamp(chunk_size, min=1, max=horizon)
		for b in range(diffused_actions.shape[0]):
			k = int(chunk_size[b].item())
			if k < horizon:
				diffused_actions[b, k:, :] = diffused_actions[b, k - 1, :].unsqueeze(0).expand(horizon - k, -1)
		return diffused_actions


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
		video_env=None,
		save_eval_video=False,
		eval_video_fps=20,
		eval_video_max_frames=300,
		eval_video_freq=1,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval
		self.video_env = video_env
		self.save_eval_video = bool(save_eval_video)
		self.eval_video_fps = max(1, int(eval_video_fps))
		self.eval_video_max_frames = max(1, int(eval_video_max_frames))
		self.eval_video_freq = max(1, int(eval_video_freq))
		self._eval_video_calls = 0
		self._eval_video_warned = False

	def _set_pending_chunk_exec(self, env, agent):
		if self.algorithm != 'dsrl_na' or not hasattr(agent, "get_last_predict_chunk_exec"):
			return
		chunks = agent.get_last_predict_chunk_exec()
		if chunks is None or not hasattr(env, "set_attr"):
			return
		n_envs = int(getattr(env, "num_envs", len(chunks)))
		for idx, chunk in enumerate(chunks[:n_envs]):
			try:
				env.set_attr("pending_chunk_exec", int(max(1, round(float(chunk)))), indices=[idx])
			except Exception:
				break

	def _normalize_video_frame(self, frame):
		if frame is None:
			return None
		if isinstance(frame, (list, tuple)):
			frame = next((item for item in frame if item is not None), None)
			if frame is None:
				return None
		if torch.is_tensor(frame):
			frame = frame.detach().cpu().numpy()
		frame = np.asarray(frame)
		if frame.ndim == 4:
			frame = frame[0]
		if frame.ndim != 3:
			return None
		if frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
			frame = np.transpose(frame, (1, 2, 0))
		if frame.shape[-1] == 4:
			frame = frame[..., :3]
		if frame.shape[-1] != 3:
			return None
		if frame.dtype != np.uint8:
			if frame.size > 0 and float(np.nanmax(frame)) <= 1.0:
				frame = frame * 255.0
			frame = np.clip(frame, 0, 255).astype(np.uint8)
		return np.ascontiguousarray(frame)

	def _capture_eval_frame(self, env):
		frame = None
		if hasattr(env, "get_images"):
			try:
				images = env.get_images()
				if isinstance(images, (list, tuple)):
					frame = next((img for img in images if img is not None), None)
				else:
					frame = images
			except Exception:
				frame = None
		if frame is None:
			try:
				frame = env.render(mode="rgb_array")
			except TypeError:
				try:
					frame = env.render()
				except Exception:
					frame = None
			except Exception:
				frame = None
		if frame is None and hasattr(env, "envs"):
			for sub_env in getattr(env, "envs", []):
				try:
					frame = sub_env.render(mode="rgb_array")
				except TypeError:
					try:
						frame = sub_env.render()
					except Exception:
						frame = None
				except Exception:
					frame = None
				if frame is not None:
					break
		return self._normalize_video_frame(frame)

	def _frames_to_wandb_video(self, frames):
		if len(frames) == 0:
			return None
		base_shape = frames[0].shape
		frames = [frame for frame in frames if frame.shape == base_shape]
		if len(frames) == 0:
			return None
		video = np.stack(frames, axis=0)
		return np.transpose(video, (0, 3, 1, 2))

	def _collect_eval_video(self, agent, deterministic=False):
		if self.video_env is None:
			return None
		env = self.video_env
		frames = []
		try:
			obs = env.reset()
			frame = self._capture_eval_frame(env)
			if frame is not None:
				frames.append(frame)
			done_i = np.zeros(int(getattr(env, "num_envs", 1)), dtype=bool)
			query_cap = self.max_steps
			if self.algorithm == 'dsrl_na' and self.max_steps > 0:
				query_cap = self.max_steps * self.action_chunk
			if query_cap <= 0:
				query_cap = self.eval_video_max_frames
			query_cap = min(int(query_cap), self.eval_video_max_frames)
			for _ in range(query_cap):
				if self.algorithm == 'dsrl_sac':
					action, _ = agent.predict(obs, deterministic=deterministic)
				elif self.algorithm == 'dsrl_na':
					action, _ = agent.predict_diffused(obs, deterministic=deterministic)
					self._set_pending_chunk_exec(env, agent)
				else:
					return None
				obs, reward, done, info = env.step(action)
				frame = self._capture_eval_frame(env)
				if frame is not None:
					frames.append(frame)
				if len(frames) >= self.eval_video_max_frames:
					break
				done_i = np.logical_or(done_i, np.asarray(done, dtype=bool)[:len(done_i)])
				if np.all(done_i):
					break
		except Exception as exc:
			if not self._eval_video_warned:
				print(f"eval video capture failed: {exc}")
				self._eval_video_warned = True
			return None
		return self._frames_to_wandb_video(frames)

	def _log_eval_video(self, agent, deterministic=False):
		if not (self.use_wandb and self.save_eval_video and self.video_env is not None):
			return
		self._eval_video_calls += 1
		if (self._eval_video_calls - 1) % self.eval_video_freq != 0:
			return
		video = self._collect_eval_video(agent, deterministic=deterministic)
		if video is None:
			if not self._eval_video_warned:
				print("eval video capture returned no frames")
				self._eval_video_warned = True
			return
		key = "eval/video_deterministic" if deterministic else "eval/video"
		wandb.log({
			key: wandb.Video(video, fps=self.eval_video_fps, format="mp4")
		}, step=self.log_count)

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		success_signal = np.array([
			info.get('chunk_success_signal', rew[idx])
			for idx, info in enumerate(self.locals['infos'])
		])
		self.episode_success[success_signal > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		step_increments = [
			float(info.get('chunk_exec', self.action_chunk))
			for info in self.locals['infos']
		]
		self.total_timesteps += int(np.sum(step_increments))
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					logger_values = self.locals['self'].logger.name_to_value
					log_payload = {
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
					}
					for key in [
						"train/ent_coef",
						"train/actor_loss",
						"train/critic_loss",
						"train/ent_coef_loss",
						"train/noise_critic_loss",
						"train/q_value_mean",
						"train/q_value_std",
						"train/target_q_value_mean",
						"train/target_q_value_std",
						"train/actor_q_value_mean",
						"train/noise_q_value_mean",
						"train/q_target_gap_mean",
						"train/actor_compute_cost_mean",
						"train/episode_budget_ratio",
						"train/episode_budget_under",
						"train/episode_budget_over",
						"train/episode_budget_penalty",
						"train/episode_budget_gross_penalty",
						"train/episode_budget_saving_bonus",
						"train/episode_budget_success_rate",
						"train/episode_budget_saving_active_rate",
						"train/difficulty_mean",
						"train/difficulty_prior_u_mean",
						"train/difficulty_loss",
						"train/schedule_gate",
						"train/active_schedule_residual_scale",
						"train/effective_schedule_residual_scale",
						"train/schedule_residual_authority",
						"train/range_alpha",
						"train/range_actuator_gate",
						"train/difficulty_prior_gate",
						"train/effective_difficulty_prior_scale",
						"train/difficulty_combined_gate",
						"train/difficulty_loss_gate",
						"train/difficulty_margin_gate",
						"train/cost_progress",
						"train/actor_compute_lambda_active",
						"train/gate_monotonic_violation_count",
						"train/denoising_steps_mean",
						"train/denoising_steps_min",
						"train/denoising_steps_max",
						"train/chunk_size_mean",
						"train/chunk_size_min",
						"train/chunk_size_max",
						"train/rollout_nfe_penalty_coef",
					]:
						if key in logger_values:
							log_payload[key] = logger_values[key]
					wandb.log({
						**log_payload
					}, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				if self.algorithm == 'dsrl_na' and (not deterministic) and hasattr(agent, "start_eval_tracking"):
					agent.start_eval_tracking()
				success, rews = [], []
				rew_total, total_ep = 0, 0
				eval_steps_sum = 0.0
				eval_chunk_requested_sum = 0.0
				eval_chunk_exec_sum = 0.0
				eval_chunk_requested_values = []
				eval_chunk_exec_values = []
				query_idx_requested_nfe = {}
				query_idx_executed_nfe = {}
				query_idx_chunk_requested = {}
				query_idx_chunk_exec = {}
				query_idx_denoising_steps = {}
				query_idx_difficulty = {}
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					done_i = np.zeros(obs.shape[0], dtype=bool)
					rew_ep_i = np.zeros(obs.shape[0])
					r = []
					query_cap = self.max_steps
					if self.algorithm == 'dsrl_na' and self.max_steps > 0:
						query_cap = self.max_steps * self.action_chunk
					for query_idx in range(query_cap):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
							self._set_pending_chunk_exec(env, agent)
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						active = ~done_i
						rew_ep_i[active] += reward[active]
						finished = np.logical_and(done, active)
						rew_total += sum(rew_ep_i[finished])
						total_ep += np.sum(finished)
						success_signal = np.array([
							info_j.get('chunk_success_signal', reward[idx])
							for idx, info_j in enumerate(info)
						])
						success_i[np.logical_and(active, success_signal > -self.rew_offset)] = 1
						done_i = np.logical_or(done_i, finished)
						if np.any(active):
							r.append(float(np.mean(reward[active])))
						if (
							self.algorithm == 'dsrl_na'
							and not deterministic
							and hasattr(agent, "get_last_eval_schedule")
						):
							schedule = agent.get_last_eval_schedule()
							steps = schedule.get("steps", np.array([]))
							chunk_requested = schedule.get("chunk_requested", np.array([]))
							difficulty = schedule.get("difficulty", np.array([]))
							n = min(len(active), len(steps), len(chunk_requested), len(info))
							if n > 0:
								active_n = active[:n]
								steps_n = np.asarray(steps[:n], dtype=np.float32)
								chunk_req_n = np.maximum(np.asarray(chunk_requested[:n], dtype=np.float32), 1.0)
								chunk_exec_n = np.asarray([
									float(info_j.get('chunk_exec', chunk_req_n[idx]))
									for idx, info_j in enumerate(info[:n])
								], dtype=np.float32)
								if np.any(active_n):
									steps_a = steps_n[active_n]
									chunk_req_a = chunk_req_n[active_n]
									chunk_exec_a = np.maximum(chunk_exec_n[active_n], 1.0)
									eval_steps_sum += float(np.sum(steps_a))
									eval_chunk_requested_sum += float(np.sum(chunk_req_a))
									eval_chunk_exec_sum += float(np.sum(chunk_exec_a))
									eval_chunk_requested_values.extend(chunk_req_a.tolist())
									eval_chunk_exec_values.extend(chunk_exec_a.tolist())
									query_idx_denoising_steps.setdefault(query_idx, []).append(float(np.mean(steps_a)))
									query_idx_chunk_requested.setdefault(query_idx, []).append(float(np.mean(chunk_req_a)))
									query_idx_chunk_exec.setdefault(query_idx, []).append(float(np.mean(chunk_exec_a)))
									query_idx_requested_nfe.setdefault(query_idx, []).append(float(np.mean(steps_a / chunk_req_a)))
									query_idx_executed_nfe.setdefault(query_idx, []).append(float(np.mean(steps_a / chunk_exec_a)))
									if len(difficulty) >= n:
										diff_a = np.asarray(difficulty[:n], dtype=np.float32)[active_n]
										query_idx_difficulty.setdefault(query_idx, []).append(float(np.mean(diff_a)))
						if np.all(done_i):
							break
					success.append(success_i.mean())
					rews.append(float(np.mean(r)) if len(r) > 0 else 0.0)
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)
				if (
					self.algorithm == 'dsrl_na'
					and not deterministic
					and hasattr(agent, "stop_eval_tracking")
					and hasattr(agent, "update_phase_from_eval")
				):
					elastic_stats = agent.stop_eval_tracking()
					requested_nfe = eval_steps_sum / max(eval_chunk_requested_sum, 1.0)
					executed_nfe = eval_steps_sum / max(eval_chunk_exec_sum, 1.0)
					agent.update_phase_from_eval(success_rate=success_rate, avg_nfe=executed_nfe)
					if self.use_wandb:
						elastic_payload = {
							"eval/avg_denoising_steps": elastic_stats.get("avg_steps", 0.0),
							"eval/avg_chunk_size": elastic_stats.get("avg_chunk", 0.0),
							"eval/avg_denoising_steps_target": elastic_stats.get("avg_steps_target", 0.0),
							"eval/avg_chunk_size_target": elastic_stats.get("avg_chunk_target", 0.0),
							"eval/target_exec_mismatch_steps": elastic_stats.get("target_exec_mismatch_steps", 0.0),
							"eval/target_exec_mismatch_chunk": elastic_stats.get("target_exec_mismatch_chunk", 0.0),
							"eval/avg_difficulty": elastic_stats.get("avg_difficulty", 0.0),
							"eval/avg_difficulty_prior_u": elastic_stats.get("avg_difficulty_prior_u", 0.0),
							"evaluation/requested_nfe_amortized": requested_nfe,
							"evaluation/nfe_amortized": executed_nfe,
							"evaluation/chunk_requested_mean": float(np.mean(eval_chunk_requested_values)) if len(eval_chunk_requested_values) > 0 else 0.0,
							"evaluation/chunk_exec_mean": float(np.mean(eval_chunk_exec_values)) if len(eval_chunk_exec_values) > 0 else 0.0,
							"evaluation/chunk_exec_over_requested_ratio": eval_chunk_exec_sum / max(eval_chunk_requested_sum, 1.0),
						}
						query_idx_keys = sorted(set().union(
							query_idx_requested_nfe.keys(),
							query_idx_executed_nfe.keys(),
							query_idx_chunk_requested.keys(),
							query_idx_chunk_exec.keys(),
							query_idx_denoising_steps.keys(),
							query_idx_difficulty.keys(),
						))
						if query_idx_keys:
							def _query_mean(bucket, q_idx):
								values = bucket.get(q_idx, [])
								return float(np.mean(values)) if values else 0.0
							query_idx_schedule_rows = [
								[
									int(q_idx),
									_query_mean(query_idx_requested_nfe, q_idx),
									_query_mean(query_idx_executed_nfe, q_idx),
									_query_mean(query_idx_chunk_requested, q_idx),
									_query_mean(query_idx_chunk_exec, q_idx),
									_query_mean(query_idx_denoising_steps, q_idx),
									_query_mean(query_idx_difficulty, q_idx),
									max(
										len(query_idx_requested_nfe.get(q_idx, [])),
										len(query_idx_chunk_requested.get(q_idx, [])),
										len(query_idx_denoising_steps.get(q_idx, [])),
										len(query_idx_difficulty.get(q_idx, [])),
									),
								]
								for q_idx in query_idx_keys
							]
							query_idx_schedule_table = wandb.Table(
								data=query_idx_schedule_rows,
								columns=[
									"query_idx",
									"requested_nfe_amortized_mean",
									"executed_nfe_amortized_mean",
									"chunk_requested_mean",
									"chunk_exec_mean",
									"denoising_steps_mean",
									"difficulty_mean",
									"count",
								],
							)
							elastic_payload["eval/query_idx_schedule_table"] = query_idx_schedule_table
							elastic_payload["eval/query_idx_nfe"] = wandb.plot.line(
								query_idx_schedule_table,
								"query_idx",
								"requested_nfe_amortized_mean",
								title="Requested Amortized NFE by Query Index",
							)
							elastic_payload["eval/query_idx_chunk"] = wandb.plot.line(
								query_idx_schedule_table,
								"query_idx",
								"chunk_requested_mean",
								title="Requested Chunk by Query Index",
							)
							elastic_payload["eval/query_idx_denoising_steps"] = wandb.plot.line(
								query_idx_schedule_table,
								"query_idx",
								"denoising_steps_mean",
								title="Denoising Steps by Query Index",
							)
							elastic_payload["eval/query_idx_difficulty"] = wandb.plot.line(
								query_idx_schedule_table,
								"query_idx",
								"difficulty_mean",
								title="Difficulty by Query Index",
							)
						wandb.log(elastic_payload, step=self.log_count)
				self._log_eval_video(agent, deterministic=deterministic)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()
