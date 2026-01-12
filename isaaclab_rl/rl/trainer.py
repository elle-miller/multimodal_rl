import copy
import sys
import torch
import tqdm
import wandb 
import numpy as np
from copy import deepcopy
from isaaclab_rl.ssl.dynamics import ForwardDynamics
from isaaclab_rl.ssl.reconstruction import Reconstruction
import matplotlib.cm as cm

SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,  # number of timesteps to train for
    "headless": False,  # whether to use headless mode (no rendering)
    "disable_progressbar": False,  # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": False,  # whether to close the environment on normal program termination
}


class Trainer:
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
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        self.cfg = _cfg
        self.env = env
        self.agent = agents
        self.agent_cfg = agent_cfg
        self.writer = writer
        self.ssl_task = ssl_task
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.encoder = self.agent.encoder

        # configure and instantiate a custom RL trainer for logging episode events
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)

        # global steps accumulates over all environments i.e. global step = num envs * training steps
        self.timesteps = int(num_timesteps_M * 1e6 / env.num_envs)
        self.global_step = 0
        self.num_envs = env.num_envs
        self.num_eval_envs = num_eval_envs

        # this is the timesteps per environment
        self.training_timestep = 0

    def train(self, play=False, trial=None) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """

        # HARD reset of all environments to begin evaluation
        states, infos = self.env.reset(hard=True)
        self.agent.memory.reset()

        # get ready
        self.returns_dict, self.infos_dict, self.mask, self.term_mask, self.trunc_mask = self.get_empty_return_dicts(
            infos
        )
        ep_length = self.env.env.unwrapped.max_episode_length - 1

        # metrics where we only care about mean over whole episode in context of training
        wandb_episode_dict = {}
        wandb_episode_dict["global_step"] = self.global_step

        wandb_images = []

        best_return = 0

        # counter variable for which step we are on
        rollout = 0
        self.rl_update = 0

        for timestep in tqdm.tqdm(
            range(self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):

            # update global step
            self.global_step = timestep * self.num_envs

            # create new wandb dict
            self.wandb_timestep_dict = {}
            self.wandb_timestep_dict["global_step"] = self.global_step

            # compute actions
            with torch.no_grad():
                z = self.encoder(states)

                # For evaluation environments
                if self.num_eval_envs > 0:
                    eval_z = z[: self.num_eval_envs]
                    eval_actions, _, _ = self.agent.policy.act(eval_z, deterministic=True)
                    # eval_actions, _, _ = self.agent.policy.act(eval_z)

                # For training environments
                train_z = z[self.num_eval_envs :]
                train_actions, train_log_prob, outputs = self.agent.policy.act(train_z)

                # Combine actions
                actions = torch.zeros((self.num_envs, train_actions.shape[1])).to(self.device)
                if self.num_eval_envs > 0:
                    actions[: self.num_eval_envs] = eval_actions
                    actions[self.num_eval_envs :] = train_actions

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                self.save_transition_to_memory(
                    states, actions, train_log_prob, rewards, next_states, terminated, truncated, infos
                )

            # begin grad!
            rollout += 1
            if not rollout % self.agent._rollouts:

                nan_encountered = self.agent._update()
                if nan_encountered:
                    return float("nan"), True

                self.rl_update += 1
                rollout = 0

            states = next_states 
            
            # save the first eval env video frames
            frame_skip = 10
            if timestep % frame_skip == 0 and self.writer.wandb_session is not None:
                # Collect rgb and/or depth for video logging from first env
                frame_data = {}
                if "rgb" in next_states["policy"].keys():
                    frame_data["rgb"] = next_states["policy"]["rgb"][:][0]
                if "depth" in next_states["policy"].keys():
                    frame_data["depth"] = next_states["policy"]["depth"][:][0]
                if frame_data:
                    wandb_images.append(frame_data)
            
            # reset environments
            # the eval episodes get manually reset every ep_length
            if timestep > 0 and (timestep % ep_length == 0) and self.num_eval_envs > 0:

                # take counter item, mean across eval envs
                if "counters" in infos.keys():
                    for k, v in infos["counters"].items():
                        wandb_episode_dict[f"Eval episode counters / {k}"] = v[: self.num_eval_envs].mean().cpu()
                        if self.writer.tb_writer is not None:
                            self.writer.tb_writer.add_scalar(
                                f"{k}", v[: self.num_eval_envs].mean().cpu(), global_step=self.global_step
                            )

                # reset eval envs
                states, _ = self.env.reset_eval_envs()

                # metrics where we only care about mean over whole episode in context of training
                # update the episode dict
                for k, v in self.returns_dict.items():
                    wandb_episode_dict[f"Eval episode returns / {k}"] = v.mean().cpu()
                    if self.writer.tb_writer is not None:
                        self.writer.tb_writer.add_scalar(f"{k}", v.mean().cpu(), global_step=self.global_step)

                for k, v in self.infos_dict.items():
                    wandb_episode_dict[f"Eval episode info / {k}"] = v.mean().cpu()
                    if self.writer.tb_writer is not None:
                        self.writer.tb_writer.add_scalar(f"{k}", v.mean().cpu(), global_step=self.global_step)

                if self.writer.wandb_session is not None:
                    self.writer.wandb_session.log(wandb_episode_dict)

                wandb_episode_dict = {}
                wandb_episode_dict["global_step"] = self.global_step

                mean_eval_return = self.returns_dict["returns"].mean().item()
                print(f"{self.global_step/1e6} M: {mean_eval_return}\n")

                # write checkpoints
                if not play:
                    self.writer.write_checkpoint(mean_eval_return, timestep=self.global_step)

                # upload video to wandb
                if len(wandb_images) > 0:
                    # Process rgb and depth separately
                    if "rgb" in wandb_images[0]:
                        # Stack RGB frames
                        rgb_frames = [img["rgb"][..., :3] for img in wandb_images if "rgb" in img]
                        rgb_tensor = torch.stack(rgb_frames).cpu()  # Result: [T, H, W, 3]
                        rgb_array = rgb_tensor.numpy()
                        if rgb_array.dtype != np.uint8:
                            rgb_array = (rgb_array * 255).astype(np.uint8)
                        # Transpose to [T, C, H, W]
                        rgb_array = rgb_array.transpose(0, 3, 1, 2)
                        wandb.log({"rgb_array": wandb.Video(rgb_array, fps=10, format="mp4")}, step=self.global_step)

                    if "depth" in wandb_images[0]:
                        # Process all depth frames to get [T, H, W] # get the first channel
                        depth_frames = [img["depth"][..., 0] for img in wandb_images if "depth" in img]
                        depth_tensor = torch.stack(depth_frames).cpu()
                        depth_raw = depth_tensor.numpy()
                        depth_color = cm.viridis(depth_raw)[..., :3] # Remove alpha channel
                        depth_color = (depth_color * 255).astype(np.uint8)
                        # Transpose to [T, C, H, W]
                        depth_color = depth_color.transpose(0, 3, 1, 2)
                        wandb.log({"depth_array": wandb.Video(depth_color, fps=10, format="mp4")}, step=self.global_step)

                    wandb_images = []

                self.returns_dict, self.infos_dict, self.mask, self.term_mask, self.trunc_mask = (
                    self.get_empty_return_dicts(infos)
                )

                # sweep stuff
                if trial is not None:
                    if mean_eval_return > best_return:
                        best_return = mean_eval_return
                    trial.report(mean_eval_return, self.global_step)
                    if trial.should_prune():
                        return best_return, True

        return best_return, False

    def save_transition_to_memory(
        self, states, actions, train_log_prob, rewards, next_states, terminated, truncated, infos
    ):
        
        # AUXILIARY MEMORY
        if self.ssl_task is not None and self.ssl_task.use_same_memory is False:
            # doing a deep copy because i observed changing the tensors in aux task messed up ppo
            # if not self.ssl_task.memory.filled:
            # deep copy doesn't seem to make much of a memory difference, so leaving in
            if isinstance(self.ssl_task, ForwardDynamics):
                self.ssl_task.add_samples(
                    states=deepcopy(states),
                    actions=deepcopy(actions),
                    # rewards=deepcopy(rewards),
                    done=deepcopy(terminated|truncated),
                )
            elif isinstance(self.ssl_task, Reconstruction):
                self.ssl_task.add_samples(
                    states=states,
                )
            else:
                raise ValueError

        # then mess up for PPO training
        train_states = self.get_last_n_obs(states, self.num_eval_envs)
        train_next_states = self.get_last_n_obs(next_states, self.num_eval_envs)
        train_actions = actions[self.num_eval_envs :, :]
        train_rewards = rewards[self.num_eval_envs :, :]
        train_terminated = terminated[self.num_eval_envs :, :]
        train_truncated = truncated[self.num_eval_envs :, :]

        # these are metrics added to self.extras["log"] in the environment at each timestep
        if self.num_eval_envs > 0:

            # compute eval rewards
            eval_rewards = rewards[: self.num_eval_envs, :]
            eval_terminated = terminated[: self.num_eval_envs, :]
            eval_truncated = truncated[: self.num_eval_envs, :]
            mask_update = 1 - torch.logical_or(eval_terminated, eval_truncated).float()


            if "log" in infos:
                for k, v in infos["log"].items():
                    # timestep logging
                    self.wandb_timestep_dict[f"Eval timestep / {k}"] = v[: self.num_eval_envs].cpu()
                    self.infos_dict[k] += v[: self.num_eval_envs].mean() * self.mask

            # update eval dicts
            self.returns_dict["unmasked_returns"] += eval_rewards
            self.returns_dict["returns"] += eval_rewards * self.mask
            self.returns_dict["steps_to_term"] += self.term_mask[: self.num_eval_envs]
            self.returns_dict["steps_to_trunc"] += self.trunc_mask[: self.num_eval_envs]
            self.mask *= mask_update
            self.term_mask *= 1 - terminated[: self.num_eval_envs].float()
            self.trunc_mask *= 1 - truncated[: self.num_eval_envs].float()

        # record to PPO memory
        self.agent.record_transition(
            states=train_states,
            actions=train_actions,
            log_prob=train_log_prob,
            rewards=train_rewards,
            next_states=train_next_states,
            terminated=train_terminated,
            truncated=train_truncated,
            timestep=self.global_step,
        )

    def get_last_n_obs(self, obs, n):
        result = {}
        for k, v in obs.items():
            result[k] = {}
            for key, value in obs[k].items():
                result[k][key] = value[:][n:]
        return result

    def get_empty_return_dicts(self, infos):
        returns = {
            "returns": None,
            "unmasked_returns": None,
            "steps_to_term": None,
            "steps_to_trunc": None,
        }

        returns_dict = {k: torch.zeros(size=(self.num_eval_envs, 1), device=self.device) for k in returns.keys()}
        infos_dict = {k: torch.zeros(size=(self.num_eval_envs, 1), device=self.device) for k in infos["log"].keys()}

        mask = torch.Tensor([[1] for _ in range(self.num_eval_envs)]).to(self.device)
        term_mask = torch.Tensor([[1] for _ in range(self.num_eval_envs)]).to(self.device)
        trunc_mask = torch.Tensor([[1] for _ in range(self.num_eval_envs)]).to(self.device)
        return returns_dict, infos_dict, mask, term_mask, trunc_mask
