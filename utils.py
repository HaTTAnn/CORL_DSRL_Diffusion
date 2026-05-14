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
		video_env=None,
		save_eval_video=False,
		eval_video_fps=20,
		eval_video_max_frames=300,
		eval_video_freq=1,
		target_env_timesteps=None,
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
		self.video_env = video_env
		self.save_eval_video = bool(save_eval_video)
		self.eval_video_fps = max(1, int(eval_video_fps))
		self.eval_video_max_frames = max(1, int(eval_video_max_frames))
		self.eval_video_freq = max(1, int(eval_video_freq))
		self.target_env_timesteps = None if target_env_timesteps is None else int(target_env_timesteps)
		self._eval_video_calls = 0
		self._eval_video_warned = False

	def _finite_array(self, values):
		arr = np.asarray(values, dtype=np.float32).reshape(-1)
		return arr[np.isfinite(arr)]

	def _safe_mean(self, values):
		arr = self._finite_array(values)
		return float(np.mean(arr)) if arr.size > 0 else 0.0

	def _safe_std(self, values):
		arr = self._finite_array(values)
		return float(np.std(arr)) if arr.size > 0 else 0.0

	def _safe_percentile(self, values, percentile):
		arr = self._finite_array(values)
		return float(np.percentile(arr, percentile)) if arr.size > 0 else 0.0

	def _safe_min(self, values):
		arr = self._finite_array(values)
		return float(np.min(arr)) if arr.size > 0 else 0.0

	def _safe_max(self, values):
		arr = self._finite_array(values)
		return float(np.max(arr)) if arr.size > 0 else 0.0

	def _safe_ratio(self, numerator, denominator):
		denominator = float(denominator)
		if abs(denominator) < 1e-8:
			return 0.0
		return float(numerator) / denominator

	def _rankdata_average(self, values):
		arr = self._finite_array(values)
		ranks = np.zeros(arr.shape[0], dtype=np.float32)
		if arr.size == 0:
			return ranks
		order = np.argsort(arr, kind="mergesort")
		sorted_arr = arr[order]
		start = 0
		while start < arr.size:
			end = start + 1
			while end < arr.size and sorted_arr[end] == sorted_arr[start]:
				end += 1
			ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
			start = end
		return ranks

	def _safe_corr(self, x_values, y_values, spearman=False):
		x = self._finite_array(x_values)
		y = self._finite_array(y_values)
		n = min(x.size, y.size)
		if n < 2:
			return 0.0
		x = x[:n]
		y = y[:n]
		if spearman:
			x = self._rankdata_average(x)
			y = self._rankdata_average(y)
		if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
			return 0.0
		return float(np.corrcoef(x, y)[0, 1])

	def _add_distribution_metrics(self, metrics, prefix, values):
		arr = self._finite_array(values)
		metrics[f"{prefix}_mean"] = self._safe_mean(arr)
		metrics[f"{prefix}_std"] = self._safe_std(arr)
		metrics[f"{prefix}_min"] = self._safe_min(arr)
		metrics[f"{prefix}_p05"] = self._safe_percentile(arr, 5)
		metrics[f"{prefix}_p50"] = self._safe_percentile(arr, 50)
		metrics[f"{prefix}_p95"] = self._safe_percentile(arr, 95)
		metrics[f"{prefix}_max"] = self._safe_max(arr)
		metrics[f"{prefix}_cv"] = self._safe_ratio(np.std(arr), np.mean(arr)) if arr.size > 0 else 0.0

	def _budget_band_metrics(self, metrics, prefix, values, target, lower, upper):
		arr = self._finite_array(values)
		if arr.size == 0:
			metrics[f"{prefix}_hit_rate"] = 0.0
			metrics[f"{prefix}_under_rate"] = 0.0
			metrics[f"{prefix}_over_rate"] = 0.0
			metrics[f"{prefix}_mean_abs_target_error"] = 0.0
			metrics[f"{prefix}_mean_band_violation"] = 0.0
			return
		metrics[f"{prefix}_hit_rate"] = float(np.mean((arr >= lower) & (arr <= upper)))
		metrics[f"{prefix}_under_rate"] = float(np.mean(arr < lower))
		metrics[f"{prefix}_over_rate"] = float(np.mean(arr > upper))
		metrics[f"{prefix}_mean_abs_target_error"] = float(np.mean(np.abs(arr - target)))
		metrics[f"{prefix}_mean_band_violation"] = float(np.mean(np.maximum(lower - arr, 0.0) + np.maximum(arr - upper, 0.0)))

	def _build_eval_summary_table(self, metrics):
		rows = []
		for key in sorted(metrics.keys()):
			value = metrics[key]
			if isinstance(value, (int, float, np.integer, np.floating)):
				category = key.split("_", 1)[0]
				rows.append([category, key, float(value)])
		return wandb.Table(data=rows, columns=["category", "metric", "value"])

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

	def _collect_eval_video(self, agent):
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
					action, _ = agent.predict(obs, deterministic=False)
				elif self.algorithm == 'dsrl_na':
					action, _ = agent.predict_diffused(obs, deterministic=False)
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

	def _log_eval_video(self, agent):
		if not (self.use_wandb and self.save_eval_video and self.video_env is not None):
			return
		self._eval_video_calls += 1
		if (self._eval_video_calls - 1) % self.eval_video_freq != 0:
			return
		video = self._collect_eval_video(agent)
		if video is None:
			if not self._eval_video_warned:
				print("eval video capture returned no frames")
				self._eval_video_warned = True
			return
		wandb.log({
			"eval/video": wandb.Video(video, fps=self.eval_video_fps, format="mp4")
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
						"train/difficulty_source_score_mean",
						"train/difficulty_source_score_std",
						"train/difficulty_prior_source_score_mean",
						"train/difficulty_prior_source_score_std",
						"train/effective_step_prior_authority",
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

		evaluated_now = False
		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'])
			evaluated_now = True
		if self.target_env_timesteps is not None and self.total_timesteps >= self.target_env_timesteps:
			if not evaluated_now:
				self.evaluate(self.locals['self'])
			print(f"target_env_timesteps reached: {self.total_timesteps} >= {self.target_env_timesteps}")
			return False
		return True
	
	def evaluate(self, agent):
		if self.eval_episodes <= 0:
			return
		env = self.eval_env
		with torch.no_grad():
			track_elastic = self.algorithm == 'dsrl_na' and hasattr(agent, "start_eval_tracking")
			if track_elastic:
				agent.start_eval_tracking()
			success = []
			rew_total, total_ep = 0, 0
			eval_steps_sum = 0.0
			eval_chunk_requested_sum = 0.0
			eval_chunk_exec_sum = 0.0
			eval_steps_values = []
			eval_steps_target_values = []
			eval_chunk_requested_values = []
			eval_chunk_exec_values = []
			eval_chunk_target_values = []
			eval_requested_nfe_values = []
			eval_executed_nfe_values = []
			eval_difficulty_values = []
			eval_prior_u_values = []
			eval_source_score_values = []
			eval_prior_source_score_values = []
			eval_query_idx_values = []
			eval_episode_success_values = []
			eval_episode_steps = []
			eval_episode_chunk_requested = []
			eval_episode_chunk_exec = []
			eval_episode_requested_nfe = []
			eval_episode_executed_nfe = []
			eval_episode_query_counts = []
			query_steps = {}
			query_steps_target = {}
			query_chunk_requested = {}
			query_chunk_target = {}
			query_chunk_exec = {}
			query_requested_nfe = {}
			query_executed_nfe = {}
			query_difficulty = {}
			query_difficulty_abs = {}
			query_prior_u = {}
			query_schedule_rows = []
			for i in range(self.eval_episodes):
				obs = env.reset()
				success_i = np.zeros(obs.shape[0])
				done_i = np.zeros(obs.shape[0], dtype=bool)
				env_step_i = np.zeros(obs.shape[0], dtype=np.int32)
				rew_ep_i = np.zeros(obs.shape[0])
				episode_steps_i = np.zeros(obs.shape[0], dtype=np.float32)
				episode_chunk_requested_i = np.zeros(obs.shape[0], dtype=np.float32)
				episode_chunk_exec_i = np.zeros(obs.shape[0], dtype=np.float32)
				episode_query_count_i = np.zeros(obs.shape[0], dtype=np.float32)
				query_cap = self.max_steps
				if self.algorithm == 'dsrl_na' and self.max_steps > 0:
					query_cap = self.max_steps * self.action_chunk
				for query_idx in range(query_cap):
					if self.algorithm == 'dsrl_sac':
						action, _ = agent.predict(obs, deterministic=False)
					elif self.algorithm == 'dsrl_na':
						action, _ = agent.predict_diffused(obs, deterministic=False)
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
					if (
						self.algorithm == 'dsrl_na'
						and hasattr(agent, "get_last_eval_schedule")
					):
						schedule = agent.get_last_eval_schedule()
						steps = schedule.get("steps", np.array([]))
						chunk_requested = schedule.get("chunk_requested", np.array([]))
						steps_target = schedule.get("steps_target", np.array([]))
						chunk_target = schedule.get("chunk_target", np.array([]))
						difficulty = schedule.get("difficulty", np.array([]))
						prior_u = schedule.get("difficulty_prior_u", np.array([]))
						source_score = schedule.get("difficulty_source_score", np.array([]))
						prior_source_score = schedule.get("difficulty_prior_source_score", np.array([]))
						n = min(len(active), len(steps), len(chunk_requested), len(info))
						if n > 0:
							active_n = active[:n]
							steps_n = np.asarray(steps[:n], dtype=np.float32)
							chunk_req_n = np.maximum(np.asarray(chunk_requested[:n], dtype=np.float32), 1.0)
							steps_target_n = np.asarray(steps_target[:n], dtype=np.float32) if len(steps_target) >= n else steps_n
							chunk_target_n = np.asarray(chunk_target[:n], dtype=np.float32) if len(chunk_target) >= n else chunk_req_n
							difficulty_n = np.asarray(difficulty[:n], dtype=np.float32) if len(difficulty) >= n else np.zeros(n, dtype=np.float32)
							prior_u_n = np.asarray(prior_u[:n], dtype=np.float32) if len(prior_u) >= n else np.zeros(n, dtype=np.float32)
							source_score_n = np.asarray(source_score[:n], dtype=np.float32) if len(source_score) >= n else np.zeros(n, dtype=np.float32)
							prior_source_score_n = np.asarray(prior_source_score[:n], dtype=np.float32) if len(prior_source_score) >= n else np.zeros(n, dtype=np.float32)
							chunk_exec_n = np.asarray([
								float(info_j.get('chunk_exec', chunk_req_n[idx]))
								for idx, info_j in enumerate(info[:n])
							], dtype=np.float32)
							step_start_n = env_step_i[:n].copy()
							step_end_n = step_start_n + np.maximum(chunk_exec_n, 1.0).astype(np.int32)
							if np.any(active_n):
								steps_a = steps_n[active_n]
								chunk_req_a = chunk_req_n[active_n]
								chunk_exec_a = np.maximum(chunk_exec_n[active_n], 1.0)
								step_target_a = steps_target_n[active_n]
								chunk_target_a = chunk_target_n[active_n]
								difficulty_a = difficulty_n[active_n]
								prior_u_a = prior_u_n[active_n]
								source_score_a = source_score_n[active_n]
								prior_source_score_a = prior_source_score_n[active_n]
								req_nfe_a = steps_a / np.maximum(chunk_req_a, 1.0)
								exec_nfe_a = steps_a / np.maximum(chunk_exec_a, 1.0)
								eval_steps_sum += float(np.sum(steps_a))
								eval_chunk_requested_sum += float(np.sum(chunk_req_a))
								eval_chunk_exec_sum += float(np.sum(chunk_exec_a))
								episode_steps_i[:n] += np.where(active_n, steps_n, 0.0)
								episode_chunk_requested_i[:n] += np.where(active_n, chunk_req_n, 0.0)
								episode_chunk_exec_i[:n] += np.where(active_n, np.maximum(chunk_exec_n, 1.0), 0.0)
								episode_query_count_i[:n] += active_n.astype(np.float32)
								eval_steps_values.extend(steps_a.tolist())
								eval_steps_target_values.extend(step_target_a.tolist())
								eval_chunk_requested_values.extend(chunk_req_a.tolist())
								eval_chunk_exec_values.extend(chunk_exec_a.tolist())
								eval_chunk_target_values.extend(chunk_target_a.tolist())
								eval_requested_nfe_values.extend(req_nfe_a.tolist())
								eval_executed_nfe_values.extend(exec_nfe_a.tolist())
								eval_difficulty_values.extend(difficulty_a.tolist())
								eval_prior_u_values.extend(prior_u_a.tolist())
								eval_source_score_values.extend(source_score_a.tolist())
								eval_prior_source_score_values.extend(prior_source_score_a.tolist())
								eval_query_idx_values.extend([float(query_idx)] * int(np.sum(active_n)))
								for env_idx in np.flatnonzero(active_n):
									step_v = float(steps_n[env_idx])
									step_target_v = float(steps_target_n[env_idx])
									chunk_req_v = float(chunk_req_n[env_idx])
									chunk_target_v = float(chunk_target_n[env_idx])
									chunk_exec_v = float(max(chunk_exec_n[env_idx], 1.0))
									diff_v = float(difficulty_n[env_idx])
									prior_u_v = float(prior_u_n[env_idx])
									source_score_v = float(source_score_n[env_idx])
									req_nfe_v = step_v / max(chunk_req_v, 1.0)
									exec_nfe_v = step_v / max(chunk_exec_v, 1.0)
									query_steps.setdefault(query_idx, []).append(step_v)
									query_steps_target.setdefault(query_idx, []).append(step_target_v)
									query_chunk_requested.setdefault(query_idx, []).append(chunk_req_v)
									query_chunk_target.setdefault(query_idx, []).append(chunk_target_v)
									query_chunk_exec.setdefault(query_idx, []).append(chunk_exec_v)
									query_requested_nfe.setdefault(query_idx, []).append(req_nfe_v)
									query_executed_nfe.setdefault(query_idx, []).append(exec_nfe_v)
									query_difficulty.setdefault(query_idx, []).append(diff_v)
									query_difficulty_abs.setdefault(query_idx, []).append(abs(diff_v))
									query_prior_u.setdefault(query_idx, []).append(prior_u_v)
									query_schedule_rows.append([
										int(i), int(env_idx), int(query_idx), int(step_start_n[env_idx]), int(step_end_n[env_idx]),
										step_target_v, step_v, chunk_target_v, chunk_req_v, req_nfe_v,
										diff_v, prior_u_v, source_score_v, int(done[env_idx]), int(success_i[env_idx]),
									])
							env_step_i[:n] = np.where(active_n, step_end_n, env_step_i[:n])
					if np.all(done_i):
						break
				episode_requested_nfe_i = episode_steps_i / np.maximum(episode_chunk_requested_i, 1.0)
				episode_executed_nfe_i = episode_steps_i / np.maximum(episode_chunk_exec_i, 1.0)
				eval_episode_success_values.extend(success_i.tolist())
				eval_episode_steps.extend(episode_steps_i.tolist())
				eval_episode_chunk_requested.extend(episode_chunk_requested_i.tolist())
				eval_episode_chunk_exec.extend(episode_chunk_exec_i.tolist())
				eval_episode_requested_nfe.extend(episode_requested_nfe_i.tolist())
				eval_episode_executed_nfe.extend(episode_executed_nfe_i.tolist())
				eval_episode_query_counts.extend(episode_query_count_i.tolist())
				success.append(success_i.mean())
				print(f'eval episode {i} at timestep {self.total_timesteps}')
			success_rate = np.mean(success)
			avg_rew = rew_total / total_ep if total_ep > 0 else 0
			eval_rollout_count = len(eval_episode_success_values)
			eval_success_count = float(np.sum(eval_episode_success_values)) if eval_rollout_count > 0 else 0.0
			if self.use_wandb:
				name = 'eval'
				wandb.log({
					f"{name}/success_rate": success_rate,
					f"{name}/reward": avg_rew,
					f"{name}/timesteps": self.total_timesteps,
					"eval_table/core_success_rate": success_rate,
					"eval_table/core_reward_mean": avg_rew,
					"eval_table/core_timesteps": float(self.total_timesteps),
					"eval_table/core_eval_rollouts": float(eval_rollout_count),
					"eval_table/core_success_count": eval_success_count,
					"eval_table/core_failure_count": max(float(eval_rollout_count) - eval_success_count, 0.0),
				}, step=self.log_count)
			if (
				self.algorithm == 'dsrl_na'
				and hasattr(agent, "stop_eval_tracking")
				and hasattr(agent, "update_phase_from_eval")
			):
				elastic_stats = agent.stop_eval_tracking()
				requested_nfe = eval_steps_sum / max(eval_chunk_requested_sum, 1.0)
				executed_nfe = eval_steps_sum / max(eval_chunk_exec_sum, 1.0)
				agent.update_phase_from_eval(success_rate=success_rate, avg_nfe=executed_nfe)
				if self.use_wandb:
					fixed_steps = float(getattr(agent, "fixed_denoising_steps", 0.0))
					fixed_chunk = float(max(getattr(agent, "fixed_chunk_size", 1.0), 1.0))
					fixed_nfe = fixed_steps / fixed_chunk if fixed_steps > 0.0 else 0.0
					target_nfe = float(getattr(agent, "target_nfe", executed_nfe))
					nfe_target_lower = float(getattr(agent, "nfe_target_lower", target_nfe))
					nfe_target_upper = float(getattr(agent, "nfe_target_upper", target_nfe))
					min_steps = float(getattr(agent, "min_denoising_steps", 0.0))
					max_steps = float(getattr(agent, "max_denoising_steps", 0.0))
					min_chunk = float(max(getattr(agent, "min_chunk_size", 1.0), 1.0))
					max_chunk = float(max(getattr(agent, "max_chunk_size", 1.0), 1.0))
					episode_query_arr = self._finite_array(eval_episode_query_counts)
					valid_episode_mask = episode_query_arr > 0.0
					episode_requested_nfe_arr = self._finite_array(eval_episode_requested_nfe)
					episode_executed_nfe_arr = self._finite_array(eval_episode_executed_nfe)
					if episode_requested_nfe_arr.size == valid_episode_mask.size:
						episode_requested_nfe_arr = episode_requested_nfe_arr[valid_episode_mask]
					if episode_executed_nfe_arr.size == valid_episode_mask.size:
						episode_executed_nfe_arr = episode_executed_nfe_arr[valid_episode_mask]
					eval_table_metrics = {
						"core_success_rate": float(success_rate),
						"core_reward_mean": float(avg_rew),
						"core_timesteps": float(self.total_timesteps),
						"core_eval_rollouts": float(eval_rollout_count),
						"core_success_count": float(eval_success_count),
						"core_failure_count": max(float(eval_rollout_count) - float(eval_success_count), 0.0),
						"cost_total_queries": float(len(eval_steps_values)),
						"cost_total_denoising_steps": float(eval_steps_sum),
						"cost_total_requested_actions": float(eval_chunk_requested_sum),
						"cost_total_executed_actions": float(eval_chunk_exec_sum),
						"cost_amortized_nfe_requested": float(requested_nfe),
						"cost_amortized_nfe_executed": float(executed_nfe),
						"cost_mean_query_nfe_requested": self._safe_mean(eval_requested_nfe_values),
						"cost_mean_query_nfe_executed": self._safe_mean(eval_executed_nfe_values),
						"cost_nfe_per_success": self._safe_ratio(eval_steps_sum, eval_success_count),
						"cost_successes_per_1k_nfe": self._safe_ratio(1000.0 * eval_success_count, eval_steps_sum),
						"cost_success_rate_per_nfe": self._safe_ratio(success_rate, executed_nfe),
						"cost_denoising_steps_per_rollout": self._safe_ratio(eval_steps_sum, eval_rollout_count),
						"cost_executed_actions_per_rollout": self._safe_ratio(eval_chunk_exec_sum, eval_rollout_count),
						"cost_queries_per_rollout": self._safe_mean(eval_episode_query_counts),
						"fixed_denoising_steps": fixed_steps,
						"fixed_chunk_size": fixed_chunk,
						"fixed_nfe": fixed_nfe,
						"fixed_compute_saving_vs_fixed_executed": 1.0 - self._safe_ratio(executed_nfe, fixed_nfe) if fixed_nfe > 0.0 else 0.0,
						"fixed_compute_saving_vs_fixed_requested": 1.0 - self._safe_ratio(requested_nfe, fixed_nfe) if fixed_nfe > 0.0 else 0.0,
						"range_min_possible_nfe": min_steps / max_chunk if max_chunk > 0.0 else 0.0,
						"range_max_possible_nfe": max_steps / min_chunk if min_chunk > 0.0 else 0.0,
						"budget_target_nfe": target_nfe,
						"budget_lower_nfe": nfe_target_lower,
						"budget_upper_nfe": nfe_target_upper,
						"control_steps_target_exec_mae": self._safe_mean(np.abs(self._finite_array(eval_steps_values) - self._finite_array(eval_steps_target_values))) if len(eval_steps_values) == len(eval_steps_target_values) and len(eval_steps_values) > 0 else 0.0,
						"control_chunk_target_requested_mae": self._safe_mean(np.abs(self._finite_array(eval_chunk_requested_values) - self._finite_array(eval_chunk_target_values))) if len(eval_chunk_requested_values) == len(eval_chunk_target_values) and len(eval_chunk_requested_values) > 0 else 0.0,
						"control_chunk_target_executed_mae": self._safe_mean(np.abs(self._finite_array(eval_chunk_exec_values) - self._finite_array(eval_chunk_target_values))) if len(eval_chunk_exec_values) == len(eval_chunk_target_values) and len(eval_chunk_exec_values) > 0 else 0.0,
						"control_chunk_request_exec_mae": self._safe_mean(np.abs(self._finite_array(eval_chunk_exec_values) - self._finite_array(eval_chunk_requested_values))) if len(eval_chunk_exec_values) == len(eval_chunk_requested_values) and len(eval_chunk_exec_values) > 0 else 0.0,
						"control_chunk_truncation_rate": float(np.mean(self._finite_array(eval_chunk_exec_values) < self._finite_array(eval_chunk_requested_values))) if len(eval_chunk_exec_values) == len(eval_chunk_requested_values) and len(eval_chunk_exec_values) > 0 else 0.0,
						"adapt_spearman_difficulty_nfe_executed": self._safe_corr(eval_difficulty_values, eval_executed_nfe_values, spearman=True),
						"adapt_spearman_difficulty_steps": self._safe_corr(eval_difficulty_values, eval_steps_values, spearman=True),
						"adapt_spearman_difficulty_chunk_requested": self._safe_corr(eval_difficulty_values, eval_chunk_requested_values, spearman=True),
						"adapt_spearman_difficulty_chunk_executed": self._safe_corr(eval_difficulty_values, eval_chunk_exec_values, spearman=True),
						"adapt_pearson_difficulty_nfe_executed": self._safe_corr(eval_difficulty_values, eval_executed_nfe_values, spearman=False),
						"adapt_spearman_source_steps": self._safe_corr(eval_source_score_values, eval_steps_values, spearman=True),
						"adapt_spearman_source_nfe_requested": self._safe_corr(eval_source_score_values, eval_requested_nfe_values, spearman=True),
						"adapt_spearman_source_difficulty": self._safe_corr(eval_source_score_values, eval_difficulty_values, spearman=True),
						"adapt_pearson_source_steps": self._safe_corr(eval_source_score_values, eval_steps_values, spearman=False),
						"adapt_pearson_source_nfe_requested": self._safe_corr(eval_source_score_values, eval_requested_nfe_values, spearman=False),
						"adapt_spearman_prior_source_steps": self._safe_corr(eval_prior_source_score_values, eval_steps_values, spearman=True),
						"adapt_spearman_prior_nfe_executed": self._safe_corr(eval_prior_u_values, eval_executed_nfe_values, spearman=True),
						"adapt_spearman_query_idx_nfe_executed": self._safe_corr(eval_query_idx_values, eval_executed_nfe_values, spearman=True),
					}
					query_idx_arr = self._finite_array(eval_query_idx_values)
					query_nfe_arr = self._finite_array(eval_executed_nfe_values)
					if query_idx_arr.size == query_nfe_arr.size and query_idx_arr.size > 0:
						early_mask = query_idx_arr <= np.percentile(query_idx_arr, 25)
						late_mask = query_idx_arr >= np.percentile(query_idx_arr, 75)
						eval_table_metrics["adapt_early_query_nfe_executed"] = self._safe_mean(query_nfe_arr[early_mask])
						eval_table_metrics["adapt_late_query_nfe_executed"] = self._safe_mean(query_nfe_arr[late_mask])
						eval_table_metrics["adapt_late_minus_early_nfe_executed"] = eval_table_metrics["adapt_late_query_nfe_executed"] - eval_table_metrics["adapt_early_query_nfe_executed"]
					self._budget_band_metrics(eval_table_metrics, "budget_requested_episode", episode_requested_nfe_arr, target_nfe, nfe_target_lower, nfe_target_upper)
					self._budget_band_metrics(eval_table_metrics, "budget_executed_episode", episode_executed_nfe_arr, target_nfe, nfe_target_lower, nfe_target_upper)
					self._add_distribution_metrics(eval_table_metrics, "dist_steps", eval_steps_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_steps_target", eval_steps_target_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_chunk_requested", eval_chunk_requested_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_chunk_executed", eval_chunk_exec_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_nfe_requested", eval_requested_nfe_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_nfe_executed", eval_executed_nfe_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_difficulty", eval_difficulty_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_difficulty_source_score", eval_source_score_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_difficulty_prior_source_score", eval_prior_source_score_values)
					self._add_distribution_metrics(eval_table_metrics, "dist_episode_nfe_executed", episode_executed_nfe_arr)
					elastic_payload = {
						"eval/source_score_mean": elastic_stats.get("avg_difficulty_source_score", 0.0),
						"eval/source_score_std": elastic_stats.get("std_difficulty_source_score", 0.0),
					}
					elastic_payload.update({
						f"eval_table/{key}": value
						for key, value in eval_table_metrics.items()
						if isinstance(value, (int, float, np.integer, np.floating))
					})
					elastic_payload["eval_table/summary"] = self._build_eval_summary_table(eval_table_metrics)
					if query_schedule_rows:
						query_schedule_columns = [
							"episode_id", "env_id", "query_idx", "env_step_start", "env_step_end",
							"steps_target", "steps_exec", "chunk_target", "chunk_requested",
							"requested_nfe", "difficulty", "difficulty_prior_u",
							"difficulty_source_score", "done_after_query", "success_so_far",
						]
						elastic_payload["eval/query_schedule_table"] = wandb.Table(
							data=query_schedule_rows,
							columns=query_schedule_columns,
						)
						traces = {}
						for row in query_schedule_rows:
							traces.setdefault((row[0], row[1]), []).append(row)
						selected_trace = None
						selected_success = 0.0
						for trace_key in sorted(traces.keys()):
							trace_rows = sorted(traces[trace_key], key=lambda row: row[2])
							if any(row[14] for row in trace_rows):
								selected_trace = trace_rows
								selected_success = 1.0
								break
						if selected_trace is None and traces:
							trace_key = sorted(traces.keys())[0]
							selected_trace = sorted(traces[trace_key], key=lambda row: row[2])
						if selected_trace:
							trace_table = wandb.Table(
								data=[
									[
										int(row[2]), int(row[3]), int(row[4]),
										float(row[6]), float(row[8]), float(row[9]),
										float(row[10]), float(row[12]),
									]
									for row in selected_trace
								],
								columns=[
									"query_idx", "env_step_start", "env_step_end",
									"steps_exec", "chunk_requested", "requested_nfe",
									"difficulty", "difficulty_source_score",
								],
							)
							elastic_payload["eval/trace_success"] = selected_success
							elastic_payload["eval/trace_schedule_table"] = trace_table
							elastic_payload["eval/trace_steps_exec"] = wandb.plot.line(trace_table, "query_idx", "steps_exec", title="Denoising Steps by Query")
							elastic_payload["eval/trace_chunk_requested"] = wandb.plot.line(trace_table, "query_idx", "chunk_requested", title="Requested Chunk by Query")
							elastic_payload["eval/trace_requested_nfe"] = wandb.plot.line(trace_table, "query_idx", "requested_nfe", title="Requested NFE by Query")
							elastic_payload["eval/trace_difficulty"] = wandb.plot.line(trace_table, "query_idx", "difficulty", title="Difficulty by Query")
							elastic_payload["eval/trace_source_score"] = wandb.plot.line(trace_table, "query_idx", "difficulty_source_score", title="Raw Difficulty Source by Query")
					wandb.log(elastic_payload, step=self.log_count)
			self._log_eval_video(agent)

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
