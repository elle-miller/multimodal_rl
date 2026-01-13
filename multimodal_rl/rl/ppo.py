import copy
import gym
import gymnasium
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple, Union

from multimodal_rl.rl.memories import Memory
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler

# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,  # number of rollouts before updating
    "learning_epochs": 8,  # number of learning epochs during each update
    "mini_batches": 2,  # number of mini batches during each learning epoch
    "discount_factor": 0.99,  # discount factor (gamma)
    "lambda": 0.95,  # TD(lambda) coefficient (lam) for computing returns and advantages
    "learning_rate": 1e-3,  # learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,  # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},  # value preprocessor's kwargs (e.g. {"size": 1})
    "grad_norm_clip": 0.5,  # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,  # clip predicted values during value loss computation
    "entropy_loss_scale": 0.0,  # entropy loss scaling factor
    "value_loss_scale": 1.0,  # value loss scaling factor
    "kl_threshold": 0,  # KL divergence threshold for early stopping
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}
# [end-config-dict-torch]


class PPO:
    def __init__(
        self,
        encoder,
        policy,
        value,
        value_preprocessor,
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        ssl_task=None,
        writer=None,
        dtype=torch.float32,
        debug: bool = False
    ) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        """
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg if cfg is not None else {}
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.dtype = dtype
        self._device_type = torch.device(device).type
        self.memory = memory
        self.debug = debug
        self.global_step = 0

        self.writer = writer
        if self.writer == None:
            self.wandb_session = None
            self.tb_writer = None
        else:
            self.wandb_session = writer.wandb_session
            self.tb_writer = writer.tb_writer
        self.ssl_task = ssl_task

        # hyperparams
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0
        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._kl_threshold = self.cfg["kl_threshold"]
        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self._value_preprocessor = value_preprocessor
        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # models
        self.policy = policy
        self.value = value
        self.encoder = encoder
        self.policy.eval()
        self.value.eval()
        self.encoder.eval()

        # Create separate optimizers for different network components
        self.policy_optimiser = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        self.value_optimiser = torch.optim.Adam(self.value.parameters(), lr=self._learning_rate)
        self.encoder_optimiser = torch.optim.Adam(self.encoder.parameters(), lr=self._learning_rate)

        # checkpoint models
        # if self.writer.save_checkpoints > 0:
        self.writer.checkpoint_modules["policy"] = self.policy
        self.writer.checkpoint_modules["value"] = self.value
        self.writer.checkpoint_modules["encoder"] = self.encoder
        self.writer.checkpoint_modules["policy_optimiser"] = self.policy_optimiser
        self.writer.checkpoint_modules["value_optimiser"] = self.value_optimiser
        self.writer.checkpoint_modules["encoder_optimiser"] = self.encoder_optimiser

        if self.encoder.state_preprocessor is not None:
            self.writer.checkpoint_modules["state_preprocessor"] = self.encoder.state_preprocessor

        if self._value_preprocessor is not None:
            self.writer.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor
            
        self.num_actions = self.action_space.shape[0]

        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(self.optimiser, **self.cfg["learning_rate_scheduler_kwargs"])

        self.update_step = 0
        self.epoch_step = 0

        # set up automatic mixed precision
        # I OBSERVE THIS LEADS TO NUMERICAL INSTABILITIES - LEAVE IT OFF UNLESS NEED TO
        self._mixed_precision = False
        self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)

        # create tensors in memory
        if self.memory is not None:
            print("****RL Agent Memory****")
            self.observation_names = []
            
            for k, v in self.observation_space["policy"].items():
                # Determine the correct dtype based on the observation key
                if k == "rgb":
                    storage_dtype = torch.uint8
                elif k == "depth":
                    storage_dtype = torch.float32  # Keep high precision for meters
                elif k == "tactile":
                    # Check if binary_tactile is enabled in config
                    tactile_cfg = self.cfg.get("observations", {}).get("tactile_cfg", {})
                    if tactile_cfg.get("binary_tactile", False):
                        storage_dtype = torch.bool
                    else:
                        storage_dtype = torch.float32
                else:
                    storage_dtype = self.dtype

                print(f"PPO: {k}: tensor size {v.shape} | dtype: {storage_dtype}")
                
                self.memory.create_tensor(
                    name=k,
                    size=v.shape,
                    dtype=storage_dtype,
                )
                self.observation_names.append(k)

            self.memory.create_tensor(name="actions", size=self.action_space, dtype=self.dtype)
            self.memory.create_tensor(name="rewards", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="values", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="returns", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="advantages", size=1, dtype=self.dtype)

            self._tensors_names = self.observation_names + [
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
            ]

        # create temporary variables needed for storage and computation
        self._current_next_states = None

    def record_transition(
        self,
        states: Union[torch.Tensor, Dict],
        actions: torch.Tensor,
        log_prob: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        timestep: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        if self.memory is not None:

            self._current_next_states = next_states

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                # sometimes this z is NaN!!!!
                z = self.encoder(states)
                values = self.value.compute_value(z)
                if self.debug:
                    self._check_instability(states["policy"]["gt"], "PPO record_transition / gt")
                    self._check_instability(states["policy"]["prop"], "PPO record_transition / prop")
                    self._check_instability(z, "PPO record_transition / z")
                    self._check_instability(values, "PPO record_transition / values before preprocesor")
                values = self._value_preprocessor(values, inverse=True)
                if self.debug:
                    self._check_instability(values, "PPO record_transition / values after preprocesor inverse")

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # only store policy obs
            self.memory.add_samples(
                type="parallel",
                states={"policy": states["policy"]},
                actions=actions,
                rewards=rewards,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                values=values,
            )

    def _update(self) -> None:
        """Algorithm's main update step"""

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            z = self.encoder(self._current_next_states)
            last_values = self.value.compute_value(z)
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)
        
        last_values = last_values
        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")
        dones = self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated")

        if self.debug:
            self._check_instability(values, "Update start / values")
            self._check_instability(last_values, "Update start /  last_values")
            self._check_instability(rewards, "Update start /  rewards")
            self._check_instability(dones, "Update start /  dones")

        returns, advantages = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        if self.debug:
            self._check_instability(returns, "Update start / returns")
            self._check_instability(advantages, "Update start /  advantages")

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        if self.ssl_task is not None:
            if self.ssl_task.use_same_memory:
                # shouldn't this be better
                sampled_aux_batches = sampled_batches
                # sampled_aux_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
            else:
                sampled_aux_batches = self.ssl_task.memory.sample_all(mini_batches=self._mini_batches)
            assert len(sampled_aux_batches) == len(sampled_batches)
        else:
            sampled_aux_batches = None

        # turn policy and networks on
        self.policy.train()
        self.value.train()
        self.encoder.train()

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_aux_loss = 0
        epoch_aux_loss = 0

        wandb_dict = {}

        # Lists to store gradient norms
        policy_grad_norms = 0
        value_grad_norms = 0


        from torch.autograd.functional import jacobian

        prop_jacobian_sum = 0
        tactile_jacobian_sum = 0

        aux_learning_epochs = max(1, int(self._learning_epochs * self.ssl_task.learning_epochs_ratio)) if self.ssl_task is not None else 1

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []
            epoch_policy_loss = 0
            epoch_value_loss = 0

            # mini-batches loop
            for i, minibatch in enumerate(sampled_batches):
                if len(minibatch) == 6:
                    # sampled_states is a dict
                    (
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                    ) = minibatch  # noqa
                else:
                    raise ValueError("Check length of sampled states, should be dict")

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = {"policy": sampled_states}

                    # torch autograd engine does not store gradients for leaf nodes by default
                    self.policy.log_std_parameter.retain_grad()

                    # --- Policy Path ---
                    # Get z from the encoder. Crucially, make it a leaf node that requires gradients.
                    # Clone and detach to ensure we're getting gradients for this specific `z` tensor,
                    # not just passing through to the encoder's parameters.
                    # If you want gradients *through* the encoder, just `requires_grad_(True)` on `z_policy` is enough.
                    # Let's assume you want gradients on the encoder's *output* `z`.
                    z_policy = self.encoder(sampled_states, train=not epoch)
                    z_policy.requires_grad_(True) # Mark as requiring gradients for autograd.grad
                    _, next_log_prob, _ = self.policy.act(z_policy, taken_actions=sampled_actions)
                
                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # --- Value Path ---
                    # Get z from the encoder for the value function. This will be a *new* tensor.
                    z_value = self.encoder(sampled_states) # Assuming same input structure
                    z_value.requires_grad_(True) # Mark as requiring gradients for autograd.grad
                    predicted_values = self.value.compute_value(z_value)

                    # make sure predicted values have only moved a little bit for stability
                    if self._value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )

                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # --- Calculate Gradients of z with respect to individual losses ---
                    # This is the crucial part to get the gradients on `z` itself.
                    # Use `retain_graph=True` because the computational graph is still needed
                    # for the subsequent combined loss backward pass.

                    # Get gradient of policy_loss with respect to z_policy
                    # torch.autograd.grad returns a tuple, so we take the first element [0]
                    policy_grad_z_tensor = torch.autograd.grad(policy_loss, z_policy, retain_graph=True)[0]
                    policy_grad_norm = policy_grad_z_tensor.norm(2).item()
                    # You would append policy_grad_norm to a list like self.policy_z_grad_norms_per_minibatch
                    # For now, just print to confirm it's working
                    # print(f"Minibatch {i}: Policy Z Grad Norm = {policy_grad_norm:.4f}")

                    # Get gradient of value_loss with respect to z_value
                    value_grad_z_tensor = torch.autograd.grad(value_loss, z_value, retain_graph=True)[0]
                    value_grad_norm = value_grad_z_tensor.norm(2).item()
                    # You would append value_grad_norm to a list like self.value_z_grad_norms_per_minibatch
                    # print(f"Minibatch {i}: Value Z Grad Norm = {value_grad_norm:.4f}")

                    policy_grad_norms += policy_grad_norm
                    value_grad_norms += value_grad_norm

                    ## aux loss
                    if self.ssl_task is not None and epoch < aux_learning_epochs:
                        aux_minibatch_full = sampled_aux_batches[i]
                        aux_minibatch = (aux_minibatch_full[0], aux_minibatch_full[1])
                        aux_loss, aux_info = self.ssl_task.compute_loss(aux_minibatch)
                        aux_loss *= self.ssl_task.aux_loss_weight
                        loss = policy_loss + entropy_loss + value_loss + aux_loss
                    else:
                        loss = policy_loss + entropy_loss + value_loss

                # optimization step
                self.encoder_optimiser.zero_grad()
                self.policy_optimiser.zero_grad()
                self.value_optimiser.zero_grad()
                if self.ssl_task is not None:
                    self.ssl_task.optimiser.zero_grad()

                # Check loss before backward
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN/Inf detected in loss at epoch {epoch}, pruning trial")
                    self._check_instability(policy_loss, "policy_loss")
                    self._check_instability(value_loss, "value_loss")
                    self._check_instability(predicted_values, "predicted_values")
                    self._check_instability(sampled_actions, "sampled_actions")
                    self._check_instability(sampled_states["policy"]["prop"], "prop")
                    self._check_instability(sampled_values, "sampled_values")
                    self._check_instability(sampled_returns, "sampled_returns")
                    self._check_instability(sampled_values, "sampled_values")
                    self._check_instability(sampled_log_prob, "sampled_log_prob")
                    self._check_instability(sampled_advantages, "sampled_advantages")
                    self.wandb_session.finish()
                    return True

                self.scaler.scale(loss).backward()

                # clip
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.encoder_optimiser)
                    self.scaler.unscale_(self.policy_optimiser)
                    self.scaler.unscale_(self.value_optimiser)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(self.policy.parameters(), self.value.parameters(), self.encoder.parameters()),
                        self._grad_norm_clip,
                    )

                self.scaler.step(self.encoder_optimiser)
                self.scaler.step(self.policy_optimiser)
                self.scaler.step(self.value_optimiser)
                if self.ssl_task is not None:
                    self.scaler.step(self.ssl_task.optimiser)

                self.scaler.update()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()

                if self.ssl_task is not None:
                    epoch_aux_loss += aux_loss.item()
                    cumulative_aux_loss += aux_loss.item()

                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # if self.wandb_session is not None:
            #     wandb_dict["global_step"] = self.epoch_step
            #     wandb_dict["Loss / epoch_policy_loss"] = epoch_policy_loss
            #     wandb_dict["Loss / epoch_value_loss"] = epoch_value_loss
            #     self.wandb_session.log(wandb_dict)

            # update cumulative losses
            epoch_aux_loss = 0
            self.epoch_step += 1
            cumulative_policy_loss += epoch_policy_loss
            cumulative_value_loss += epoch_value_loss


        # After training, track metrics
        # prop_weight_norm, tactile_weight_norm = self.encoder.get_first_layer_weight_norms()
        # # mini-batches loop
        # for i, minibatch in enumerate(sampled_batches):
        #     (
        #         sampled_states,
        #         sampled_actions,
        #         sampled_log_prob,
        #         sampled_values,
        #         sampled_returns,
        #         sampled_advantages,
        #     ) = minibatch 


        #     from torch.autograd.functional import jacobian

        #     s_t = self.encoder.concatenate_obs(sampled_states)
        #     s_t.requires_grad_(True)

        #     z_t = self.encoder(sampled_states) 
        #     z_t.requires_grad_(True)

        #     # output shape = [256, 340]
        #     dz_ds = jacobian(self.encoder.net, s_t)

        #     dV_dz = jacobian(self.value.value_net, z_t)

        #     dV_ds_matrix = torch.einsum('bij,bjk->bik', dV_dz, dz_ds)

            # # output shape: [340]
            # jacobian_matrix_norm = torch.norm(dz_ds_matrix, dim=0)

            # prop_jacobian = jacobian_matrix_norm[:self.num_prop_inputs]
            # tactile_jacobian = jacobian_matrix_norm[self.num_prop_inputs:]

            # # Calculate the L2-norm for each portion
            # prop_norm = torch.norm(prop_jacobian, p=2)
            # tactile_norm = torch.norm(tactile_jacobian, p=2)

        # wandb log
        if self.wandb_session is not None:
            wandb_dict["global_step"] = self.update_step
            avg_policy_loss = cumulative_policy_loss / (self._learning_epochs * self._mini_batches)
            avg_value_loss = cumulative_value_loss / (self._learning_epochs * self._mini_batches)
            avg_policy_gradient = policy_grad_norms /  (self._learning_epochs * self._mini_batches)
            avg_value_gradient = value_grad_norms /  (self._learning_epochs * self._mini_batches)

            wandb_dict["Loss / Policy loss"] = avg_policy_loss
            wandb_dict["Loss / Value loss"] = avg_value_loss
            # wandb_dict["Weights / prop_weight_norm"] = prop_weight_norm
            # wandb_dict["Weights / tactile_weight_norm"] = tactile_weight_norm
            wandb_dict["Weights / avg_policy_gradient"] = avg_policy_gradient
            wandb_dict["Weights / avg_value_gradient"] = avg_value_gradient
            # wandb_dict["Weights / prop_jacobian"] = prop_jacobian
            # wandb_dict["Weights / tactile_jacobian"] = tactile_jacobian
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("policy_loss", avg_policy_loss, global_step=self.global_step)
                self.tb_writer.add_scalar("value_loss", avg_value_loss, global_step=self.global_step)
                # self.tb_writer.add_scalar("prop_weight_norm", prop_weight_norm, global_step=self.global_step)
                # self.tb_writer.add_scalar("tactile_weight_norm", tactile_weight_norm, global_step=self.global_step)
                self.tb_writer.add_scalar("avg_policy_gradient", avg_policy_gradient, global_step=self.global_step)
                self.tb_writer.add_scalar("avg_value_gradient", avg_value_gradient, global_step=self.global_step)
                # self.tb_writer.add_scalar("prop_jacobian", prop_jacobian, global_step=self.global_step)
                # self.tb_writer.add_scalar("tactile_jacobian", tactile_jacobian, global_step=self.global_step)

            if self.ssl_task is not None:
                avg_aux_loss = cumulative_aux_loss / (self._learning_epochs * self._mini_batches)
                wandb_dict["Loss / Aux loss"] = avg_aux_loss
                wandb_dict["Memory / size"] = len(self.memory)
                wandb_dict["Memory / memory_index"] = self.ssl_task.memory.memory_index
                wandb_dict["Memory / N_filled"] = int(
                    self.ssl_task.memory.total_samples / self.ssl_task.memory.memory_size
                )
                wandb_dict["Memory / filled"] = float(self.ssl_task.memory.filled)
                wandb_dict["Loss / Entropy loss"] = cumulative_entropy_loss / (
                    self._learning_epochs * self._mini_batches
                )

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("aux_loss", avg_aux_loss, global_step=self.update_step)
                for k, v in aux_info.items():
                    wandb_dict[k] = v
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(k, v, global_step=self.update_step)

            if isinstance(self._value_preprocessor, RunningStandardScaler):
                wandb_dict["Value scaler / running_mean_mean"] = self._value_preprocessor.running_mean_mean
                wandb_dict["Value scaler / running_mean_median"] = self._value_preprocessor.running_mean_median
                wandb_dict["Value scaler / running_mean_min"] = self._value_preprocessor.running_mean_min
                wandb_dict["Value scaler / running_mean_max"] = self._value_preprocessor.running_mean_max
                wandb_dict["Value scaler / running_variance_mean"] = self._value_preprocessor.running_variance_mean
                wandb_dict["Value scaler / running_variance_median"] = self._value_preprocessor.running_variance_median
                wandb_dict["Value scaler / running_variance_min"] = self._value_preprocessor.running_variance_min
                wandb_dict["Value scaler / running_variance_max"] = self._value_preprocessor.running_variance_max

            wandb_dict["Policy / Standard deviation"] = self.policy.distribution(role="policy").stddev.mean().item()
            self.wandb_session.log(wandb_dict)

        self.update_step += 1
        self.num_train_envs = 4096-100
        self.global_step = self.update_step * self._rollouts * self.num_train_envs

        # turn policy and networks off for rollout collection
        self.policy.eval()
        self.value.eval()
        self.encoder.eval()

        # no NaN encountered
        return False

    def _check_instability(self, x, name):
        if torch.isnan(x).any():
            print(f"PPO / {name} is nan", torch.isnan(x).any())
        if torch.isinf(x).any():
            print(f"PPO / {name} is inf", torch.isinf(x).any())

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for name, data in modules.items():
                print(f"Loading {name} module...")
                module = self.writer.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    print(f"Cannot load the {name} module. The agent doesn't have such an instance")


    def _empty_preprocessor(self, _input, *args, **kwargs):
        """Empty preprocess method

        This method is defined because PyTorch multiprocessing can't pickle lambdas

        :param _input: Input to preprocess
        :type _input: Any

        :return: Preprocessed input
        :rtype: Any
        """
        return _input