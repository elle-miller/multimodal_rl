"""Unified logging and checkpointing interface.

Handles Weights & Biases logging, TensorBoard summaries, and model checkpointing
through a single interface that can be passed throughout the codebase.
"""

import copy
import glob
import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import wandb

# Configure wandb directories
os.environ["WANDB_DIR"] = "./wandb"
os.environ["WANDB_CACHE_DIR"] = "./wandb"
os.environ["WANDB_CONFIG_DIR"] = "./wandb"
os.environ["WANDB_DATA_DIR"] = "./wandb"


class Writer:
    """Unified writer for experiment logging and checkpointing.
    
    Manages:
    - Weights & Biases (wandb) logging
    - TensorBoard summaries
    - Model checkpoint saving (best model and/or periodic checkpoints)
    - Video logging
    
    Args:
        agent_cfg: Agent configuration dictionary containing:
            - experiment: Dictionary with logging settings
            - log_path: Base path for logs (defaults to current directory)
        play: If True, disable all logging (default: False).
        delay_wandb_startup: If True, delay wandb initialization (default: False).
    """

    def __init__(self, agent_cfg, play=False, delay_wandb_startup=False):
        self.exp_cfg = agent_cfg["experiment"]
        self.cfg_to_save = agent_cfg
        # Checkpoint saving modes: 0=none, 1=best only, 2=all checkpoints
        self.save_checkpoints = agent_cfg["experiment"]["save_checkpoints"]

        log_path = agent_cfg.get("log_path") or os.getcwd()
        self.log_root_path = os.path.join(
            log_path,
            "logs",
            agent_cfg["experiment"]["directory"],
            agent_cfg["experiment"]["experiment_name"],
        )
        
        self.checkpoint_modules = {}
        self.video_dir = agent_cfg["experiment"].get("video_dir", "./videos")
        self.last_uploaded = set()

        if agent_cfg["experiment"]["upload_videos"]:
            os.makedirs(self.video_dir, exist_ok=True)

        self.get_new_log_path()

        # Disable logging in play mode
        if play:
            self.wandb_session = None
            self.tb_writer = None
            return

        # Initialize wandb if not delayed
        if not delay_wandb_startup:
            self.setup_wandb()
        else:
            self.wandb_session = None



    def get_new_log_path(self):
        """Create a new log directory with timestamp and initialize writers."""
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.log_root_path, run_time)
        self.setup_tb_writer()

        # Initialize checkpoint tracking
        if self.save_checkpoints > 0:
            self.checkpoint_best_modules = {
                "timestep": 0,
                "reward": float("-inf"),
                "saved": False,
                "modules": {}
            }
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_wandb(self, name=None):
        """Initialize Weights & Biases logging session.
        
        Args:
            name: Run name (defaults to name from config).
        """
        if name is None:
            name = self.exp_cfg["wandb_kwargs"]["name"]
        
        if self.exp_cfg["wandb"]:
            code_to_save = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
            wandb.init(
                project=self.exp_cfg["wandb_kwargs"]["project"],
                entity=self.exp_cfg["wandb_kwargs"]["entity"],
                group=self.exp_cfg["wandb_kwargs"]["group"],
                name=str(name),
                config=self.cfg_to_save,
                settings=wandb.Settings(code_dir=code_to_save),
            )
            # Define global_step as the primary metric for all logged values
            wandb.define_metric("global_step")
            self.wandb_session = wandb
        else:
            self.wandb_session = None

    def close_wandb(self):
        """Close the wandb session."""
        if self.wandb_session is not None:
            self.wandb_session.finish()

    def setup_tb_writer(self):
        """Initialize TensorBoard summary writer if enabled."""
        if self.exp_cfg["tb_log"]:
            self.tb_writer = SummaryWriter(self.log_dir)
        else:
            self.tb_writer = None

    def write_checkpoint(self, mean_eval_return: float, timestep: int) -> None:
        """Save model checkpoints based on evaluation performance.
        
        Saves best model when performance improves, and optionally saves
        all checkpoints if save_checkpoints == 2.
        
        Args:
            mean_eval_return: Mean return from evaluation.
            timestep: Current training timestep.
        """
        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("mean_eval_return", mean_eval_return, global_step=timestep)

        if self.save_checkpoints == 0:
            return

        # Track best model
        if mean_eval_return > self.checkpoint_best_modules["reward"]:
            self.checkpoint_best_modules["timestep"] = timestep
            self.checkpoint_best_modules["reward"] = mean_eval_return
            self.checkpoint_best_modules["saved"] = False
            self.checkpoint_best_modules["modules"] = {
                k: copy.deepcopy(self._get_module_state(v))
                for k, v in self.checkpoint_modules.items()
            }

        # Save periodic checkpoint if enabled
        if self.save_checkpoints == 2:
            modules = {
                name: self._get_module_state(module)
                for name, module in self.checkpoint_modules.items()
            }
            checkpoint_path = os.path.join(self.checkpoint_dir, f"agent_{timestep}.pt")
            torch.save(modules, checkpoint_path)

        # Save best model if improved
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            modules = {
                name: self.checkpoint_best_modules["modules"][name]
                for name in self.checkpoint_modules.keys()
            }
            torch.save(modules, os.path.join(self.checkpoint_dir, "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True

    def _get_module_state(self, module):
        """Extract state dict from module or return as-is.
        
        Args:
            module: Module or object to extract state from.
            
        Returns:
            State dict if module has one, otherwise the object itself.
        """
        return module.state_dict() if hasattr(module, "state_dict") else module

    def log_videos(self, step):
        """Upload new video files to wandb.
        
        Scans video directory for new .mp4 files and uploads them once.
        
        Args:
            step: Global step for logging.
        """
        video_paths = glob.glob(os.path.join(self.video_dir, "*.mp4"))
        for path in video_paths:
            if path not in self.last_uploaded:
                wandb.log({"agent_video": wandb.Video(path, fps=4, format="mp4")}, step=step)
                self.last_uploaded.add(path)
