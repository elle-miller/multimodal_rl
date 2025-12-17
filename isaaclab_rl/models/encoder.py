import torch
import torch.nn as nn

from isaaclab_rl.models.cnn import ImageEncoder
from isaaclab_rl.models.mlp import MLP
from isaaclab_rl.models.running_standard_scaler import RunningStandardScalerDict

methods = ["early", "intermediate"]


class Encoder(nn.Module):
    """encoder"""

    def __init__(self, observation_space, action_space, env_cfg, config_dict, device):
        super().__init__()

        self.method = config_dict["encoder"]["method"]
        assert self.method in methods
        self.device = device

        self.observation_space = observation_space
        self.action_space = action_space

        self.num_inputs = 0

        self.hiddens = config_dict["encoder"]["hiddens"]
        self.activations = config_dict["encoder"]["activations"]

        # standard scaler
        if config_dict["encoder"]["state_preprocessor"] is not None:
            self.state_preprocessor = RunningStandardScalerDict(size=observation_space, device=device)
        else:
            self.state_preprocessor = None

        # configure relevant preprocessing
        for k, v in observation_space.items():

            if k == "pixel_cfg":
                # always use cnn for pixels
                self.pixel_obs_dim = v.shape
                self.img_dim = config_dict["observations"]["pixel_cfg"]["width"]
                latent_img_dim = config_dict["observations"]["pixel_cfg"]["latent_img_dim"]
                self.cnn = ImageEncoder(self.pixel_obs_dim, latent_dim=latent_img_dim, num_layers=2).to(device)
                self.num_inputs += latent_img_dim

            # TODO: figure out prop + gt case
            elif k == "prop":
                self.num_prop_inputs = v.shape[0]
                if self.method == "early":
                    self.num_inputs += self.num_prop_inputs
                # make mlp for prop input
                elif self.method == "intermediate":
                    raise NotImplementedError

            elif k == "gt":
                num_gt_inputs = v.shape[0]
                if self.method == "early":
                    self.num_inputs += num_gt_inputs
                # make mlp for prop input
                elif self.method == "intermediate":
                    raise NotImplementedError

            elif k == "tactile":
                self.num_tactile_inputs = v.shape[0]
                if self.method == "early":
                    self.num_inputs += self.num_tactile_inputs
                # make mlp for prop input
                elif self.method == "intermediate":
                    raise NotImplementedError

        self.num_outputs = self.hiddens[-1]
        self.net = MLP(self.num_inputs, self.hiddens, self.activations, layernorm=config_dict["encoder"]["layernorm"]).to(device)


    def get_first_layer_weight_norms(self):

        """
        Calculates the normalized L2-norm for the proprioceptive and tactile
        portions of the first layer's weights.
        """
        # Access the weights of the first linear layer in the ModuleList
        first_layer_weights = self.net[0].weight
        
        # Split the weights into proprioceptive and tactile parts
        # The number of columns corresponds to the number of input features
        prop_weights = first_layer_weights[:, :self.num_prop_inputs]
        tactile_weights = first_layer_weights[:, self.num_prop_inputs:]

        # Calculate the L2-norm for each portion
        prop_norm = torch.norm(prop_weights, p=2)
        tactile_norm = torch.norm(tactile_weights, p=2)

        # print(prop_weights.size())
        # print(tactile_weights.size())
        # print(prop_norm.item(), tactile_norm.item())
        # print("***")
        
        # Normalize by the number of input features (size of the weight vector)
        prop_norm_normalized = prop_norm  # / self.num_prop_inputs
        tactile_norm_normalized = tactile_norm # / self.num_tactile_inputs

        
        return prop_norm_normalized.item(), tactile_norm_normalized.item()


    # def get_jacobian(self, sampled_states):

    #     from torch.autograd.functional import jacobian

    #     concat_obs = self.concatenate_obs(sampled_states["policy"])

    #     batch_size = 10

    #     prop_jacobian_sum = 0 
    #     tactile_jacobian_sum = 0

    #     for i in range(batch_size):

    #         single_obs = concat_obs[i]

    #         # Assuming 'model' is your neural network and 'input_tensor' is your data
    #         # Make sure input_tensor has requires_grad=True
    #         single_obs.requires_grad_(True)

    #         # output shape = [256, 340]
    #         dz_ds_matrix = jacobian(self.net, single_obs)
    #         # da_ds_matrix = jacobian(self.policy, single_obs)
    #         # dV_ds_matrix = jacobian(self.value, single_obs)

    #         # output shape: [340]
    #         jacobian_matrix_norm = torch.norm(dz_ds_matrix, dim=0)

    #         prop_jacobian = jacobian_matrix_norm[:self.num_prop_inputs]
    #         tactile_jacobian = jacobian_matrix_norm[self.num_prop_inputs:]

    #         # Calculate the L2-norm for each portion
    #         prop_norm = torch.norm(prop_jacobian, p=2)
    #         tactile_norm = torch.norm(tactile_jacobian, p=2)

    #         prop_jacobian_sum += prop_norm
    #         tactile_jacobian_sum += tactile_norm

    #     # print(prop_jacobian_sum, tactile_jacobian_sum)

    #     return prop_jacobian_sum, tactile_jacobian_sum



    def concatenate_obs(self, obs_dict):
        # separate out components of obs dict
        # for early , concat raw inputs with image inputs
        if self.method == "early":
            raw_inputs = self.get_raw_inputs(obs_dict)
            latent_inputs = self.get_latent_inputs(obs_dict)
            concat_obs = torch.cat((raw_inputs, latent_inputs), dim=-1)

        # for intermediate , pass raw inputs through mlps
        else:
            raise NotImplementedError

        return concat_obs

    def forward(self, obs_dict, detach=False, train=False):
        """
        Take in an obs dict, and return z

        """
        # sometimes need to detach, e.g. for linear probing
        if detach:
            obs_dict = {key: value.detach() for key, value in obs_dict.items()}

        if "policy" in obs_dict.keys():
            obs_dict = obs_dict["policy"]

        # scale inputs
        if self.state_preprocessor is not None:
            obs_dict = self.state_preprocessor(obs_dict, train)

        concat_obs = self.concatenate_obs(obs_dict)

        # h, _ = self.lstm(concat_obs)
        z = self.net(concat_obs)

        if z.dim() == 1:
            z = z.unsqueeze(-1)  # Adds a trailing dimension to ensure (num_envs, obs) when only one observation

        return z

    def get_raw_inputs(self, obs_dict):
        """
        Retrieve prop or gt if exists
        """
        raw_inputs = torch.tensor([]).to(self.device)
        # LOOP THROUGH DICT IN ALPHABETICAL ORDER!!!!!!!!!!!!!!!!!! fml
        for k in sorted(obs_dict.keys()):
            if k == "prop" or k == "gt" or k == "tactile":
                raw_inputs = torch.cat((raw_inputs, obs_dict[k][:]), dim=-1)
        return raw_inputs

    def get_latent_inputs(self, obs_dict):

        latent_inputs = torch.tensor([]).to(self.device)
        for k in sorted(obs_dict.keys()):
            if k == "pixel_cfg" or k == "depth":
                z = self.cnn(obs_dict[k][:])
                latent_inputs = torch.cat((latent_inputs, z), dim=-1)
        return latent_inputs


# class AIRECEncoder(nn.Module):
#     """encoder"""

#     def __init__(self, observation_space, config_dict, device):
#         super().__init__()

#         print(config_dict)

#         self.device = device

#         self.observation_space = observation_space

#         self.num_inputs = 0

#         self.hiddens = config_dict["hiddens"]
#         self.activations = config_dict["activations"]

#         # standard scaler
#         if config_dict["state_preprocessor"] is not None:
#             self.state_preprocessor = RunningStandardScalerDict(size=observation_space, device=device)
#         else:
#             self.state_preprocessor = None

#         # configure relevant preprocessing
#         for k, v in observation_space.items():

#             # TODO: figure out prop + gt case
#             if k == "prop":
#                 num_prop_inputs = v.shape[0]
#                 self.num_inputs += num_prop_inputs

#             elif k == "gt":
#                 num_gt_inputs = v.shape[0]
#                 self.num_inputs += num_gt_inputs

#             elif k == "tactile":
#                 num_tactile_inputs = v.shape[0]
#                 self.num_inputs += num_tactile_inputs

#         if self.hiddens == []:
#             self.net = nn.Sequential(nn.Identity())
#             self.num_outputs = self.num_inputs

#         else:
#             layernorm = config_dict["layernorm"]
#             self.num_outputs = self.hiddens[-1]
#             self.net = MLP(self.num_inputs, self.hiddens, self.activations, layernorm=layernorm).to(device)

#         # print("*********Encoder*************")
#         # print(self.net)



#     def forward(self, obs_dict, detach=False, train=False):
#         """
#         Take in an obs dict, and return z

#         """
#         # sometimes need to detach, e.g. for linear probing
#         if detach:
#             obs_dict = {key: value.detach() for key, value in obs_dict.items()}

#         if "policy" in obs_dict.keys():
#             obs_dict = obs_dict["policy"]

#         concat_obs = self.concatenate_obs(obs_dict)

#         # h, _ = self.lstm(concat_obs)
#         z = self.net(concat_obs)

#         if z.dim() == 1:
#             z = z.unsqueeze(-1)  # Adds a trailing dimension to ensure (num_envs, obs) when only one observation

#         return z

#     def get_raw_inputs(self, obs_dict):
#         """
#         Retrieve prop or gt if exists
#         """
#         raw_inputs = torch.tensor([]).to(self.device)
#         # LOOP THROUGH DICT IN ALPHABETICAL ORDER!!!!!!!!!!!!!!!!!! fml
#         for k in sorted(obs_dict.keys()):
#             if k == "prop" or k == "gt" or k == "tactile":
#                 raw_inputs = torch.cat((raw_inputs, obs_dict[k][:]), dim=-1)
#         return raw_inputs
