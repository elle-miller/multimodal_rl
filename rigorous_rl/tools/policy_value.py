# sequence of observations, actions, next_observations


# encoder


# sequence of z's

import matplotlib.pyplot as plt
import torch


def visualize_policy_value(encoder, policy, value_function, z):
    """
    Process observations through encoder and visualize the latent representations
    """
    # Convert data to torch tensors
    # obs_tensor = torch.FloatTensor(observations)
    # next_obs_tensor = torch.FloatTensor(next_observations)

    fig = plt.figure(figsize=(18, 12))

    # Get latent representations
    with torch.no_grad():
        values = value_function.compute_value(z).squeeze().cpu().numpy()
        policy_outputs = policy.act(z)[2]["mean_actions"].cpu().numpy()

    # Plot 4: Value function over time
    ax4 = fig.add_subplot(234)
    ax4.plot(values, label="Value")
    ax4.set_title("Value Function Over Trajectory")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Value")
    ax4.legend()

    # Plot 5: First few policy outputs over time
    ax5 = fig.add_subplot(235)
    for i in range(policy_outputs.shape[1]):
        ax5.plot(policy_outputs[:, i], label=f"Action Dim {i}")
    ax5.set_title("Policy Outputs Over Trajectory")
    ax5.set_xlabel("Timestep")
    ax5.set_ylabel("Policy Output")
    ax5.legend()

    plt.show()
