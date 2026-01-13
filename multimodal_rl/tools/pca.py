import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 24,
    }
)


plt.rcParams.update(plt.rcParamsDefault)


def plot_single_pca(z, c=None, output_file="pca.pdf", title="Default"):
    # Scale the data (recommended for PCA)
    scale = False
    if scale:
        scaler = StandardScaler()
        z = scaler.fit_transform(z)

    # Perform PCA on the original trajectory data
    pca = PCA(n_components=2)
    z_pca_scaled = pca.fit_transform(z)
    z_pca = pca.fit_transform(z)

    # Create figure for combined PCA plot
    fig, ax = plt.subplots(figsize=(6, 4))
    # Plot the results
    if c is None:
        c = np.arange(len(z)) / max(1, len(z) - 1)
        label = "Time"
    else:
        c = c
        label = "Number of tactile activations"

    # Plot first two PCA components
    scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=c, cmap="plasma", alpha=1, s=50)
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label(label)
    # plt.legend()
    plt.title(title)
    plt.savefig("results/" + output_file, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def plot_group_pca(z, c=None, ep_length=250, n_eps=1):
    # Scale the data (recommended for PCA)
    scale = False
    if scale:
        scaler = StandardScaler()
        z = scaler.fit_transform(z)

    # Perform PCA on the original trajectory data
    pca = PCA(n_components=2)
    z_pca_scaled = pca.fit_transform(z)
    z_pca = pca.fit_transform(z)

    labels = ["Rollout 1", "Rollout 2", "Rollout 3", "Rollout 4"]
    markers = ["o", "s", "^", "d"]  # circle, square, triangle, diamond

    # Plot the results
    if c is None:
        c = np.arange(ep_length)
    else:
        c = c
    # Create figure for combined PCA plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_eps):

        # Plot first two PCA components
        ep_idxs = range(i * ep_length, (i + 1) * ep_length)
        # print(z_pca.shape, ep_idxs, time_array.shape)
        scatter = ax.scatter(
            z_pca[ep_idxs, 0],
            z_pca[ep_idxs, 1],
            c=c[ep_idxs],
            cmap="plasma",
            marker=markers[i],
            label=labels[i],
            alpha=0.9,
            s=70,
            linewidth=0.5,
            edgecolors="white",
        )

    # Add colorbar to show time mapping
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label("Time")

    # Add labels and legend
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.set_title("FrankaFind Latent Trajectories (Prop + Tactile Concatenated)")

    # Equal aspect ratio often makes trajectory plots easier to interpret
    # ax.set_aspect('equal', adjustable='box')

    # Add grid for easier reading
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_joint_pca(
    z, z_hat, c=None, ep_length=250, n_components=None, plot_style="seaborn-whitegrid", stack=False, scale=False
):
    """
    Perform PCA on trajectory data z, project forward model outputs z_hat into the same latent space,
    and create visualizations comparing them.

    Parameters:
    -----------
    z : array-like
        Original trajectory data with shape (n_samples, n_features)
    z_hat : array-like
        Predicted outputs from forward model with shape (n_samples, n_features)
    n_components : int, optional
        Number of PCA components to keep. If None, all components are kept.
    plot_style : str, optional
        Matplotlib style to use for plotting

    Returns:
    --------
    pca : sklearn.decomposition.PCA
        Fitted PCA model
    z_pca : array-like
        PCA-transformed original data
    z_hat_pca : array-like
        z_hat projected into PCA space of z
    fig_combined : matplotlib.figure.Figure
        Figure containing combined PCA plot
    fig_explained_var : matplotlib.figure.Figure
        Figure containing explained variance plot
    """
    # Scale the data (recommended for PCA)

    # Perform PCA on the original trajectory data
    pca = PCA(n_components=2)

    num_samples = z.shape[0]
    # Plot the results
    if c is None:
        c = np.arange(num_samples)
    else:
        c = c

    if stack:
        # stack
        if scale:
            scaler = StandardScaler()
            z = scaler.fit_transform(z)
            scaler = StandardScaler()
            z_hat = scaler.fit_transform(z_hat)
            # z = scaler.fit_transform(z)
            # z_hat = scaler.transform(z_hat)

        z_stacked = np.concatenate((z, z_hat), axis=0)

        z_stacked_pca = pca.fit_transform(z_stacked)
        z_pca = z_stacked_pca[:num_samples, :]
        z_hat_pca = z_stacked_pca[num_samples:, :]
        print(z_stacked.shape, z_stacked_pca.shape, z_pca.shape, z_hat_pca.shape)

    else:
        if scale:
            scaler = StandardScaler()
            z = scaler.fit_transform(z)
            scaler = StandardScaler()
            z_hat = scaler.fit_transform(z_hat)  # Use same scaling as z

            # scaler = StandardScaler()
            # z = scaler.fit_transform(z)
            # z_hat = scaler.transform(z_hat)

        # Fit PCA according to ground truth z
        z_pca = pca.fit_transform(z)

        # Project the predicted data onto the same PCA space
        z_hat_pca = pca.transform(z_hat)

    # Create figure for combined PCA plot
    fig_combined, ax = plt.subplots(figsize=(10, 8))

    # Plot first two PCA components
    ax.scatter(z_pca[:, 0], z_pca[:, 1], label="Original Trajectory (z)", c=c, cmap="plasma", alpha=0.7, s=50)
    ax.scatter(
        z_hat_pca[:, 0],
        z_hat_pca[:, 1],
        c=c,
        label="Forward Model Output (z_hat)",
        cmap="plasma",
        alpha=0.7,
        s=50,
        marker="x",
    )

    # Add connecting lines between corresponding points
    for i in range(len(z_pca)):
        ax.plot([z_pca[i, 0], z_hat_pca[i, 0]], [z_pca[i, 1], z_hat_pca[i, 1]], "k-", alpha=0.2, linewidth=0.5)

    # Add arrows to show direction from z to z_hat
    step = 3
    # for i in range(0, len(z_pca), step):  # Add arrows for subset of points
    #     dx = z_hat_pca[i, 0] - z_pca[i, 0]
    #     dy = z_hat_pca[i, 1] - z_pca[i, 1]
    #     ax.arrow(z_pca[i, 0], z_pca[i, 1], dx, dy, fc='red', ec='red', alpha=0.6) #head_width=0.2, head_length=0.3,

    # Add arrows to show direction from z to z next
    # step = 5
    # for i in range(0, len(z_pca)-1, step):  # Add arrows for subset of points
    #     dx = z_pca[i+1, 0] - z_pca[i, 0]
    #     dy = z_pca[i+1, 1] - z_pca[i, 1]
    #     ax.arrow(z_pca[i, 0], z_pca[i, 1], dx, dy, fc='red', ec='red', alpha=0.8) #head_width=0.1, head_length=0.1,

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("PCA: Original Trajectory vs Forward Model Output")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Create figure for explained variance plot
    plot_variance = False
    if plot_variance:
        fig_explained_var, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Plot explained variance ratio
        components = range(1, len(pca.explained_variance_ratio_) + 1)
        axs[0].bar(components, pca.explained_variance_ratio_, alpha=0.7)
        axs[0].set_xlabel("Principal Component")
        axs[0].set_ylabel("Explained Variance Ratio")
        axs[0].set_title("Explained Variance per Component")
        axs[0].grid(True, alpha=0.3)

        # Plot cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        axs[1].plot(components, cumulative_variance, "o-", linewidth=2)
        axs[1].set_xlabel("Number of Components")
        axs[1].set_ylabel("Cumulative Explained Variance")
        axs[1].set_title("Cumulative Explained Variance")
        axs[1].grid(True, alpha=0.3)

        # Add horizontal lines at common thresholds
        for threshold in [0.8, 0.9, 0.95]:
            axs[1].axhline(y=threshold, color="r", linestyle="--", alpha=0.5)
            # Find first component that exceeds threshold
            if any(cumulative_variance >= threshold):
                component_idx = np.where(cumulative_variance >= threshold)[0][0]
                axs[1].text(len(components), threshold, f" {threshold:.0%}", verticalalignment="bottom")
                axs[1].text(
                    component_idx + 1, threshold, f" {component_idx + 1} components", verticalalignment="bottom"
                )

        plt.tight_layout()

    # If you want additional visualizations like 3D plots:
    if z_pca.shape[1] >= 3:  # If we have at least 3 components
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        ax_3d.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], label="Original Trajectory (z)", alpha=0.7, s=50)
        ax_3d.scatter(
            z_hat_pca[:, 0],
            z_hat_pca[:, 1],
            z_hat_pca[:, 2],
            label="Forward Model Output (z_hat)",
            alpha=0.7,
            s=50,
            marker="x",
        )

        # Connect with lines
        for i in range(len(z_pca)):
            ax_3d.plot(
                [z_pca[i, 0], z_hat_pca[i, 0]],
                [z_pca[i, 1], z_hat_pca[i, 1]],
                [z_pca[i, 2], z_hat_pca[i, 2]],
                "k-",
                alpha=0.2,
                linewidth=0.5,
            )

        ax_3d.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax_3d.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax_3d.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
        ax_3d.set_title("3D PCA: Original vs Predicted")
        ax_3d.legend()

    plt.show()

    # Return fitted objects and figures
    # return pca, z_pca, z_hat_pca, fig_combined, fig_explained_var


# Example usage
if __name__ == "__main__":
    # Generate some example data - replace with your actual data
    n_samples = 100
    n_features = 20

    # Create trajectory data z with some structure
    np.random.seed(42)
    z = np.random.randn(n_samples, n_features)
    # Add some structure by making first few dimensions more important
    z[:, 0] *= 5
    z[:, 1] *= 3

    # Create z_hat as z with some noise and systematic shift (simulating prediction errors)
    z_hat = z + 0.5 * np.random.randn(n_samples, n_features)
    z_hat[:, 0] += 1  # Add systematic bias in first dimension

    # Run the analysis
    pca, z_pca, z_hat_pca, fig_combined, fig_explained_var = pca_projection_analysis(z, z_hat)
