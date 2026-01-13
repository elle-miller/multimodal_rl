import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F

from multimodal_rl.ssl.task import AuxiliaryTask
from multimodal_rl.ssl.physics_memory import DynamicsMemory
from multimodal_rl.ssl.reconstruction import CustomDecoder

from multimodal_rl.models.mlp import MLP, Projector
from multimodal_rl.models.dynamics import DynamicsMLP
from multimodal_rl.wrappers.frame_stack import LazyFrames


from copy import deepcopy

from collections import deque

class ForwardDynamics(AuxiliaryTask):
    def __init__(self, aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer):
        super().__init__(aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer)

        # sequence length has to be minimum 2 to collect the next state
        assert self.seq_length > 1
        self.obs_stack = env.obs_stack 
        self.tau = 0.01

        self.memory_size = rl_rollout * env.num_envs

        self.target_encoder = deepcopy(self.encoder).to(env.device)
        self.forward_model = DynamicsMLP(state_dim=self.z_dim, action_dim=self.action_dim).to(env.device)
        self.projector = Projector(input_dim=self.z_dim, state_dim=self.z_dim).to(env.device)

        print("***Forward dynamics***")
        print("MEMORY_SIZE", self.memory_size)
        print(self.forward_model)
        print(self.projector)

        # if we are doing tactile only FD, need a tactile decoder :)
        # else we use a target encoder for z loss
        if self.tactile_only:
            latent_dim = self.encoder.num_outputs
            self.num_tactile_obs = int(self.env.num_tactile_observations)
            self.tactile_decoder = CustomDecoder(latent_dim=latent_dim, output_dim=self.num_tactile_obs).to(self.device)
            print("***Tactile decoder for tactile dynamics****")
            print(self.tactile_decoder)
        
        # holding queues for past N states
        self.temp_states = deque(maxlen=self.seq_length)
        self.temp_actions = deque(maxlen=self.seq_length)
        self.temp_alive = deque(maxlen=self.seq_length)

        super()._post_init()

    def set_optimisable_networks(self):
        if self.tactile_only:
            return [self.encoder, self.forward_model, self.projector, self.tactile_decoder]

        else:
            return [self.encoder, self.forward_model, self.projector]

    def create_memory(self):
        return DynamicsMemory(self.env, self.encoder, self.value, self.value_preprocessor, self.memory_size, self.seq_length)

    def sample_minibatches(self):
        batch_list = []
        sampled_batches = self.memory.sample_all(mini_batches=self.mini_batches)
        batch_list.append(sampled_batches)
        return batch_list

    def compute_loss(self, minibatch):
        """
        Compute loss on minibatch
        
        """
        # in this case, states contains sequences of transitions
        states, actions = minibatch

        states, next_states = self.separate_memory_tensors(states)
        n = self.seq_length -1

        if next_states == None:
            next_obs_dict = None
            obs_dict_list = self.get_obs_as_dicts(states, augment=False) #self.augmentations)
            actions = actions.transpose(0, 1)
        else:
            raise ValueError
        
        info = {}
        # compute sequence loss
        loss = 0
        for t in range(0, n):
            # with torch.no_grad():
            z_t = self.encoder(obs_dict_list[t])
            # print(z_t.shape, actions[t].shape)
            z_hat = self.forward_model(z_t, actions[t])

            with torch.no_grad():
                z_target = self.target_encoder(next_obs_dict if next_obs_dict is not None else obs_dict_list[t+1])
                
            loss += F.mse_loss(self.projector(z_hat), z_target)
            
            # if self.tactile_only:
            #     # sigmoid prediction
            #     tactile_pred = self.tactile_decoder(z_hat)

            #     # GET NEXT TACTILE or the current
            #     tactile_true = obs_dict_list[t+1]["tactile"]

            #     # Use BCE loss for binary tactile signals 
            #     # If 1s are ~10x less common than 0s across all positions
            #     pos_weight = torch.ones(self.num_tactile_obs).to(self.device) * 10
            #     tactile_loss = F.binary_cross_entropy_with_logits(
            #         tactile_pred, tactile_true, 
            #         pos_weight=pos_weight  # Applies to all positions equally
            #     )
            #     loss += tactile_loss
            #     more_info = self.evaluate_binary_predictions(tactile_pred, tactile_true, step=t)
            #     info.update(more_info)

        info["Memory / percent alive"] = self.percent_alive

        self.soft_update_params(self.encoder, self.target_encoder, tau=self.tau)

        return loss, info

    def create_memory_tensors(self):
        """
        No longer used
        """
        pass
        

    def add_samples(self, states, actions, done):
        """
        Add samples to dedicated aux memory
        Re-implement this if you don't need all of these tensors
        
        """
        if not isinstance(states["policy"]["prop"], LazyFrames):
            raise TypeError("should be LazyFrames")

        # Store termination info with states and actions
        self.temp_states.append(states)
        self.temp_actions.append(actions.detach().clone())
        self.temp_alive.append(~(done).squeeze(1))  # Store alive status for each step

        # if none of the envs are full, nothing to add
        if len(self.temp_actions) != self.seq_length:
            return
        
        # return sequences with no termination/truncation signals
        # obs_dict_seq is [alive_envs, seq_length, obs_stack, obs_size]
        # action_seq is [alive_envs, seq_length, action_size]
        obs_dict_seq, action_seq = self.get_alive_sequences()
        if obs_dict_seq is None:
            return 

        # just save the policy key, no more need for aux
        if "policy" in obs_dict_seq.keys():
            obs_dict_seq = obs_dict_seq["policy"]

        # reshape stuff back to memory
        for obs_k, v in obs_dict_seq.items():
            # print(v.shape)
            obs_size = v.shape[-1]
            num_samples = v.shape[0]
            obs_dict_seq[obs_k] = v.reshape(num_samples, self.seq_length, self.obs_stack*obs_size)
        
        # in memory 
        self.memory.add_samples(
            obs_dict_seq,
            action_seq
        )

    def get_alive_sequences(self):
        
        # Create a mask for environments that have been alive throughout the entire sequence
        fully_alive_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.seq_length):
            fully_alive_mask = fully_alive_mask & self.temp_alive[i]
        self.percent_alive = torch.sum(fully_alive_mask) / self.env.num_envs

        # Only add transitions for environments that remained alive throughout the sequence
        if not torch.any(fully_alive_mask):
            print("no fully alive")
            return None, None, None    
        
        obs_dict_seq = {}
        for obs_k in self.env.observation_space["policy"].keys():
            # self.temp_states[i]["policy"][obs_k] is shape [num_envs, obs_size]
            # self.temp_states[i]["policy"][obs_k][fully_alive_mask] is [alive_envs, obs_size]

            # this produces [seq_length, obs_stack, alive_envs, obs_size]
            if isinstance(self.temp_states[0]["policy"][obs_k], LazyFrames):
                seq = torch.stack([
                    torch.stack([
                        self.temp_states[i]["policy"][obs_k][obs_idx][fully_alive_mask] 
                        for obs_idx in range(self.obs_stack)
                    ])
                    for i in range(self.seq_length)
                ])
                # Then permute the dimensions to get [alive_envs, seq_length, obs_stack, obs_size]
                seq = seq.permute(2, 0, 1, 3)

                obs_dict_seq[obs_k] = seq
            else:
                raise TypeError("not LazyFrames")

        # [seq_length, alive_envs, action_size] ->   must be [num_samples, seq_length, action_size] for memory
        action_seq = torch.stack([self.temp_actions[i][fully_alive_mask] for i in range(self.seq_length)])
        action_seq = action_seq.permute(1,0,2)

        return obs_dict_seq, action_seq #, reward_seq

    def get_obs_as_dicts(self, states, augment=False):
        # dict, each key contains batch of sequenced tensors
        obs_sequence = states["policy"]

        # get individual obs_dict for each state in order to get z
        obs_dict_list = [{} for _ in range(self.seq_length)]
        for k in sorted(obs_sequence.keys()):
            # shape (batch_size, seq_length, input_size)
            obs = obs_sequence[k] 
            if k == "prop" or k == "gt" or k == "tactile":
                for i in range(self.seq_length):
                    # saving as (batch_size, input_size)
                    obs_dict_list[i][k] = obs[:,i,:]

        return obs_dict_list


class InverseDynamics(AuxiliaryTask):
    pass
