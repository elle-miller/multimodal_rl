"""Training loop for PPO agents with evaluation and logging.

Handles the main training loop, environment interaction, evaluation,
checkpointing, and video logging for RL agents.
"""

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
import tqdm
import wandb

from multimodal_rl.ssl.dynamics import ForwardDynamics
from multimodal_rl.ssl.reconstruction import Reconstruction


@dataclass
class EpisodeTracker:
    """Tracks episode-level metrics for evaluation environments.
    
    Accumulates rewards and info metrics, handling episode boundaries
    with masking to avoid counting terminated/truncated episodes.
    """
    num_eval_envs: int
    device: torch.device
    info_keys: list = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize tracking tensors."""
        # Returns tracking
        self.returns = torch.zeros((self.num_eval_envs, 1), device=self.device)
        self.unmasked_returns = torch.zeros((self.num_eval_envs, 1), device=self.device)
        self.steps_to_term = torch.zeros((self.num_eval_envs, 1), device=self.device)
        self.steps_to_trunc = torch.zeros((self.num_eval_envs, 1), device=self.device)
        
        # Info metrics tracking
        self.info_metrics = {
            k: torch.zeros((self.num_eval_envs, 1), device=self.device)
            for k in self.info_keys
        }
        
        # Active episode mask (1 = active, 0 = done)
        self.active_mask = torch.ones((self.num_eval_envs, 1), device=self.device)
    
    def update(self, rewards, terminated, truncated, info_metrics=None):
        """Update tracking with new timestep data.
        
        Args:
            rewards: Rewards tensor for eval envs.
            terminated: Termination flags.
            truncated: Truncation flags.
            info_metrics: Optional dict of info metrics to accumulate.
        """
        # Update masks
        done = terminated | truncated
        self.active_mask *= (1 - done.float())
        
        # Accumulate returns (masked and unmasked)
        self.unmasked_returns += rewards
        self.returns += rewards * self.active_mask
        
        # Track steps until termination/truncation (increment for active episodes)
        self.steps_to_term += self.active_mask * (1 - terminated.float())
        self.steps_to_trunc += self.active_mask * (1 - truncated.float())

        # Accumulate info metrics (scalar mean per timestep, weighted by active mask)
        if info_metrics is not None:
            for k, v in info_metrics.items():
                if k in self.info_metrics:
                    # v is already sliced to eval envs, compute mean across all dimensions
                    # Then weight by active mask mean (fraction of active episodes)
                    # This gives a scalar that can be added to self.info_metrics[k] which is [num_eval_envs, 1]
                    scalar_value = v.mean().item() if v.numel() > 0 else 0.0
                    mask_weight = self.active_mask.mean().item()
                    # Add scalar to all elements of the tensor (broadcasting)
                    self.info_metrics[k] += scalar_value * mask_weight
    
    def reset(self, info_keys=None):
        """Reset all tracking to initial state.
        
        Args:
            info_keys: Optional new list of info keys (if changed).
        """
        self.returns.zero_()
        self.unmasked_returns.zero_()
        self.steps_to_term.zero_()
        self.steps_to_trunc.zero_()
        self.active_mask.fill_(1.0)
        
        if info_keys is not None:
            self.info_keys = info_keys
            self.info_metrics = {
                k: torch.zeros((self.num_eval_envs, 1), device=self.device)
                for k in info_keys
            }
        else:
            for v in self.info_metrics.values():
                v.zero_()
    
    def get_mean_returns(self):
        """Get mean returns across environments."""
        return {
            "mean_returns": self.returns.mean().item(),
            "unmasked_returns": self.unmasked_returns.mean().item(),
            "max_returns": self.returns.max().item(),
            "min_returns": self.returns.min().item(),
            "std_returns": self.returns.std().item(),
            "median_returns": self.returns.median().item(),
        }
    
    def get_mean_info(self):
        """Get mean info metrics across environments."""
        return {k: v.mean().item() for k, v in self.info_metrics.items()}

SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,
    "headless": False,
    "disable_progressbar": False,
    "close_environment_at_exit": False,
    "staggered_resets": True,  # Stagger training environment resets for better sample diversity
}


class Trainer:
    """Training loop manager for RL agents.
    
    Manages the main training loop, separating training and evaluation environments,
    handling PPO updates, logging metrics, and saving checkpoints.
    
    Args:
        env: Training environment (gymnasium-compatible).
        agents: PPO agent instance.
        agent_cfg: Agent configuration dictionary.
        num_timesteps_M: Number of timesteps in millions (default: 0).
        num_eval_envs: Number of evaluation environments (default: 1).
        ssl_task: Optional SSL auxiliary task (default: None).
        writer: Writer instance for logging (default: None).
    """

    def __init__(
        self,
        env,
        agents,
        agent_cfg,
        num_timesteps_M=0,
        num_eval_envs=1,
        ssl_task=None,
        writer=None,
    ) -> None:
        self.cfg = deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        self.env = env
        self.agent = agents
        self.agent_cfg = agent_cfg
        self.writer = writer
        self.ssl_task = ssl_task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.agent.encoder

        # Configuration
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.staggered_resets = self.cfg.get("staggered_resets", True)

        # Training parameters
        # Global steps accumulate over all environments: global_step = num_train_envs * training_steps
        self.timesteps = int(num_timesteps_M * 1e6 / env.num_train_envs)
        self.global_step = 0
        self.num_train_envs = env.num_train_envs
        self.num_eval_envs = num_eval_envs
        self.num_envs = env.num_train_envs  + num_eval_envs # Total environments (train + eval)
        assert num_eval_envs > 0, "Number of evaluation environments must be greater than 0"
        self.training_timestep = 0
        
        # Create frozen copies of policy and encoder for evaluation
        # These will be updated only when starting a new evaluation
        self.eval_policy = deepcopy(self.agent.policy)
        self.eval_encoder = deepcopy(self.agent.encoder)
        self.eval_policy.eval()
        self.eval_encoder.eval()
        
        # Pre-allocate actions tensor for efficiency (reused each timestep)
        # We'll determine the action dimension from the first action computation
        self.actions = None

    def train(self, play=False, trial=None):
        """Execute main training loop.
        
        Runs the training loop with the following steps:
        1. Reset environments
        2. Compute actions (deterministic for eval, stochastic for training)
        3. Step environments
        4. Record transitions
        5. Update agent periodically
        6. Log metrics and videos
        7. Reset evaluation environments at episode boundaries
        
        Args:
            play: If True, skip checkpoint saving (default: False).
            trial: Optuna trial for hyperparameter optimization (default: None).
            
        Returns:
            Tuple of (best_return, should_prune) for Optuna integration.
        """
        # Hard reset all environments to begin training
        states, infos = self.env.reset(hard=True)
        train_states, eval_states = self.split_train_eval_obs(states, self.num_eval_envs)
        self.agent.memory.reset()
        
        # Initialize staggered resets for training environments
        # This prevents all environments from resetting simultaneously,
        # improving sample diversity and training stability
        if self.staggered_resets:
            max_ep_len = self.env.env.unwrapped.max_episode_length
            # Only stagger training environments (exclude eval envs)
            train_env_ids = torch.arange(
                self.num_eval_envs, self.num_envs, dtype=torch.long, device=self.device
            )
            # Uniformly distribute initial episode lengths across [0, max_episode_length)
            # This ensures training environments timeout at different times
            self.env.env.unwrapped.episode_length_buf[train_env_ids] = torch.randint(
                0, max_ep_len, (len(train_env_ids),), device=self.device, dtype=torch.long
            )

        # Initialize episode tracker
        info_keys = list(infos.get("log", {}).keys())
        self.episode_tracker = EpisodeTracker(
            num_eval_envs=self.num_eval_envs,
            device=self.device,
            info_keys=info_keys
        )
        ep_length = self.env.env.unwrapped.max_episode_length - 1

        # Episode-level metrics for logging
        wandb_episode_dict = {"global_step": self.global_step}
        wandb_images = []
        best_return = 0.0
        rollout = 0
        self.rl_update = 0
        
        # Start first evaluation episode immediately with initial random policy
        self._snapshot_policy_for_eval()

        actions = torch.zeros((self.num_envs, self.agent.action_space.shape[0]), device=self.device)

        for timestep in tqdm.tqdm(
            range(self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):

            # update global step
            self.global_step = timestep * self.num_train_envs

            # Compute actions
            with torch.no_grad():

                # Evaluation: use frozen snapshot for consistent policy
                eval_z = self.eval_encoder(eval_states)
                eval_actions, _, _ = self.eval_policy.act(eval_z, deterministic=True)
                
                # Training: use live policy
                train_z = self.encoder(train_states)
                train_actions, train_log_prob, _ = self.agent.policy.act(train_z)


                # Combine actions from eval and training environments
                actions[: self.num_eval_envs] = eval_actions.detach()
                actions[self.num_eval_envs :] = train_actions.detach()

                # Step environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                next_train_states, next_eval_states = self.split_train_eval_obs(next_states, self.num_eval_envs)

                # Render if not headless
                if not self.headless:
                    self.env.render()

                # Save transition to RL memory for training
                self.save_transition_to_memory(
                    train_states, train_actions, train_log_prob, rewards[self.num_eval_envs :, :], next_train_states, terminated[self.num_eval_envs :, :], truncated[self.num_eval_envs :, :], infos
                )
        
                # Update episode tracker
                eval_info = None
                if "log" in infos:
                    eval_info = {k: v[: self.num_eval_envs] for k, v in infos["log"].items()}
                self.episode_tracker.update(
                    rewards=rewards[: self.num_eval_envs, :],
                    terminated=terminated[: self.num_eval_envs, :],
                    truncated=truncated[: self.num_eval_envs, :],
                    info_metrics=eval_info
                )

            # Update agent after collecting enough rollouts
            rollout += 1
            if rollout % self.agent._rollouts == 0:
                nan_encountered = self.agent._update()
                if nan_encountered:
                    return float("nan"), True
                self.rl_update += 1
                rollout = 0

            # Collect video frames periodically
            FRAME_SKIP = 10
            if timestep % FRAME_SKIP == 0 and self.writer.wandb_session is not None:
                frame_data = {}
                if "rgb" in next_states["policy"]:
                    frame_data["rgb"] = next_states["policy"]["rgb"][:][0]
                if "depth" in next_states["policy"]:
                    frame_data["depth"] = next_states["policy"]["depth"][:][0]
                if frame_data:
                    wandb_images.append(frame_data)
            
            # Check episode boundaries - always evaluate after each episode
            if timestep % ep_length == 0:
                # End current evaluation episode if one is running
                result = self._evaluate_and_log(
                    infos, wandb_episode_dict, wandb_images, play, trial, best_return
                )
                if result is None:  # Early termination requested
                    return float("nan"), True
                best_return = result
                wandb_episode_dict = {"global_step": self.global_step}
                wandb_images = []
                
                # Start new evaluation episode after episode boundary (snapshot current policy)
                # Snapshot current policy and encoder for evaluation
                self._snapshot_policy_for_eval()
                # Reset eval envs to start fresh episode with frozen policy
                states, infos = self.env.reset_eval_envs()
                train_states, eval_states = self.split_train_eval_obs(states, self.num_eval_envs)
                info_keys = list[Any](infos.get("log", {}).keys())
                self.episode_tracker.reset(info_keys=info_keys)
            else:
                # Update states for next iteration (skip if we just reset above)
                states = next_states
                train_states = next_train_states
                eval_states = next_eval_states

        return best_return, False

    def _snapshot_policy_for_eval(self):
        """Create a frozen snapshot of policy and encoder for evaluation.
        
        This ensures the entire evaluation episode uses the same policy version,
        even if training continues and updates the live policy.
        """
        # Copy state dicts (more efficient than deepcopy for large networks)
        self.eval_policy.load_state_dict(self.agent.policy.state_dict())
        self.eval_encoder.load_state_dict(self.agent.encoder.state_dict())
        self.eval_policy.eval()
        self.eval_encoder.eval()

    def _evaluate_and_log(self, infos, wandb_episode_dict, wandb_images, play, trial, best_return):
        """Evaluate current policy and log metrics.
        
        Args:
            infos: Environment info dictionary.
            wandb_episode_dict: Dictionary for wandb logging.
            wandb_images: List of video frames.
            play: Whether in play mode (skip checkpoints).
            trial: Optuna trial (if applicable).
            best_return: Current best return.
            
        Returns:
            Updated best_return, or None if trial should be pruned.
        """
        # Log counter metrics
        if "counters" in infos:
            for k, v in infos["counters"].items():
                metric_value = v[: self.num_eval_envs].mean().cpu()
                wandb_episode_dict[f"Eval episode counters / {k}"] = metric_value
                if self.writer.tb_writer is not None:
                    self.writer.tb_writer.add_scalar(k, metric_value, global_step=self.global_step)

        # Get episode metrics from tracker
        returns = self.episode_tracker.get_mean_returns()
        info_metrics = self.episode_tracker.get_mean_info()
        
        # Log episode returns
        for k, v in returns.items():
            wandb_episode_dict[f"Eval episode returns / {k}"] = v
            if self.writer.tb_writer is not None:
                self.writer.tb_writer.add_scalar(k, v, global_step=self.global_step)

        # Log info metrics
        for k, v in info_metrics.items():
            wandb_episode_dict[f"Eval episode info / {k}"] = v
            if self.writer.tb_writer is not None:
                self.writer.tb_writer.add_scalar(k, v, global_step=self.global_step)

        # Log to wandb
        if self.writer.wandb_session is not None:
            self.writer.wandb_session.log(wandb_episode_dict)

        mean_eval_return = returns["mean_returns"]
        print(f"{self.global_step/1e6:.2f}M steps (update {self.rl_update}): {mean_eval_return:.4f}")

        # Save checkpoint
        if not play:
            self.writer.write_checkpoint(mean_eval_return, timestep=self.global_step)

        # Upload videos to wandb
        if wandb_images:
            self._upload_videos_to_wandb(wandb_images)

        # Optuna trial reporting
        if trial is not None:
            if mean_eval_return > best_return:
                best_return = mean_eval_return
            trial.report(mean_eval_return, self.global_step)
            if trial.should_prune():
                return None  # Signal early termination
        
        return best_return

    def _upload_videos_to_wandb(self, wandb_images):
        """Process and upload RGB/depth videos to wandb.
        
        Args:
            wandb_images: List of frame dictionaries containing "rgb" and/or "depth" keys.
        """
        if "rgb" in wandb_images[0]:
            # Stack RGB frames: [T, H, W, 3] -> [T, C, H, W]
            rgb_frames = [img["rgb"][..., :3] for img in wandb_images if "rgb" in img]
            rgb_tensor = torch.stack(rgb_frames).cpu()
            rgb_array = rgb_tensor.numpy()
            if rgb_array.dtype != np.uint8:
                rgb_array = (rgb_array * 255).astype(np.uint8)
            rgb_array = rgb_array.transpose(0, 3, 1, 2)  # [T, H, W, 3] -> [T, C, H, W]
            wandb.log({"rgb_array": wandb.Video(rgb_array, fps=10, format="mp4")}, step=self.global_step)

        if "depth" in wandb_images[0]:
            # Process depth frames: extract first channel, apply colormap
            depth_frames = [img["depth"][..., 0] for img in wandb_images if "depth" in img]
            depth_tensor = torch.stack(depth_frames).cpu()
            depth_raw = depth_tensor.numpy()
            depth_color = cm.viridis(depth_raw)[..., :3]  # Apply colormap, remove alpha
            depth_color = (depth_color * 255).astype(np.uint8)
            depth_color = depth_color.transpose(0, 3, 1, 2)  # [T, H, W, 3] -> [T, C, H, W]
            wandb.log({"depth_array": wandb.Video(depth_color, fps=10, format="mp4")}, step=self.global_step)

    def save_transition_to_memory(
        self, states, actions, log_prob, rewards, next_states, terminated, truncated, infos
    ):
        """Save transition to memory buffers and update evaluation metrics.
        
        Handles both RL memory and SSL auxiliary task memory (if separate).
        Updates evaluation tracking dictionaries for episode metrics.
        
        Args:
            states: Current state observations.
            actions: Actions taken.
            log_prob: Log probabilities of training actions.
            rewards: Rewards received.
            next_states: Next state observations.
            terminated: Termination flags.
            truncated: Truncation flags.
            infos: Environment info dictionary.
        """
        # Save to SSL auxiliary memory if using separate memory
        if self.ssl_task is not None and not self.ssl_task.use_same_memory:
            # Deep copy to avoid modifying tensors used by PPO
            if isinstance(self.ssl_task, ForwardDynamics):
                self.ssl_task.add_samples(
                    states=deepcopy(states),
                    actions=deepcopy(actions),
                    done=deepcopy(terminated | truncated),
                )
            elif isinstance(self.ssl_task, Reconstruction):
                self.ssl_task.add_samples(states=states)
            else:
                raise ValueError(f"Unknown SSL task type: {type(self.ssl_task)}")

        # Record to PPO memory
        self.agent.record_transition(
            states=states,
            actions=actions,
            log_prob=log_prob,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            timestep=self.global_step,
        )

    def split_train_eval_obs(self, obs, n):
        """Extract observations from last n environments.
        
        Args:
            obs: Observation dictionary.
            n: Number of environments to skip from the beginning.
            
        Returns:
            Observation dictionary containing only training environments.
        """
        eval_obs = {}
        train_obs = {}
        for k, v in obs.items():
            eval_obs[k] = {}
            train_obs[k] = {}
            for key, value in obs[k].items():
                eval_obs[k][key] = value[:][:n]
                train_obs[k][key] = value[:][n:]
        return train_obs, eval_obs
