<!-- # Multimodal RL (`multimodal_rl`) -->
![multimodal_rl](mmrl.png)

Real-world robotics must move beyond simple state vectors. `multimodal_rl` provides a streamlined and robust foundation for training robotic agents in Isaac Lab that perceive the world through multiple lenses.

This library is designed as a core research dependency. It handles the RL "heavy lifting" and multimodal fusion, allowing you to focus on your environment and task science. It works in tandem with [roto](https://github.com/elle-miller/roto), which provides ready-to-use example environments and optimised agents.

## ✨ Features
- **Multimodal perception**: Native support for flexible dictionary observations (RGB, Depth, Proprioception, Tactile, and Ground-truth states).
- **Self-supervised learning**: Built-in integration for SSL auxiliary tasks (reconstruction, world models 🌏) to accelerate representation learning from multimodal observations.
- **Observation stacking**: Uses `LazyFrame` stacking to handle partially observable environments, essential for real-world robotics.
- **Transparent codebase**: Most RL libraries sacrifice clarity for modularity. We condense the entire PPO logic into four readable files, making it easy to inspect "under-the-hood".
- **Robust research**: Integrated hyperparameter optimisation with Optuna to ensure fair comparisons and well-tuned agents.
- **Evaluation rigor**: Dedicated split for training and evaluation parallelised environments to ensure efficient and accurate performance reporting.



## Installation

1. Install Isaac Lab via pip with [these instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)

2. Install `multimodal_rl` as a local editable package.

```
git clone git@github.com:elle-miller/multimodal_rl.git
cd multimodal_rl
pip install -e .
```
You should now see it with `pip show multimodal_rl`.

3. Setup your own project! Check out [roto](https://github.com/elle-miller/roto) to use existing environments or as a template for your own.


## 🏗 How it Works
`multimodal_rl` contains the RL engine, while your project repo contains the environments/research/science. This separation allows you to pull updates from the core library without messy merge conflicts in your environment code.

`multimodal_rl` provides 5 core functionalities:

1. **rl**: Clean PPO implementation
2. **ssl**: Modules for self-supervision learning
2. **models**: Standardised backbones (MLPs, CNNs) and running scalers.
3. **tools**: Scripts to produce nice RL paper plots, and extra stuff like latent trajectory visualisation.
4. **wrappers**: Wrappers for observation stacking and Isaac Lab

![multimodal_rl](diagram.png)

## 📜 Credits
The PPO implementation is a streamlined and modified version of [SKRL](https://github.com/Toni-SM/skrl). This version has been refactored to prioritise multimodal fusion, evaluation rigor, and transparency.


## 📚 Citation
If this framework helps your research, please cite:
```
@misc{miller2026_multimodal_rl,
  author       = {Elle Miller},
  title        = {multimodal_rl: Multimodal RL for Real-world Robotics},
  year         = {2026},
  howpublished = {\url{https://github.com/elle-miller/multimodal_rl}},
  note         = {GitHub repository}
}
```
 
