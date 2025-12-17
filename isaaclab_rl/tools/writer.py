"""

This class contains ANYTHING to do with logging on wandb, creating tensorboard summary writer for plotting,
or saving torch checkpoints.

The idea with this is so I can just pass this one object around that does everything

"""

import copy
import glob
import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import wandb

os.environ["WANDB_DIR"] = "./wandb"
os.environ["WANDB_CACHE_DIR"] = "./wandb"
os.environ["WANDB_CONFIG_DIR"] = "./wandb"
os.environ["WANDB_DATA_DIR"] = "./wandb"


class Writer:
    def __init__(self, agent_cfg, play=False, delay_wandb_startup=False):
        self.exp_cfg = agent_cfg["experiment"]
        self.cfg_to_save = agent_cfg
        # {0: no, 1: best agent only, 2: all agents}
        self.save_checkpoints = agent_cfg["experiment"]["save_checkpoints"]

        if agent_cfg["log_path"] is None:
            agent_cfg["log_path"] = os.getcwd()

        self.log_root_path = os.path.join(
            agent_cfg["log_path"],
            "logs",
            agent_cfg["experiment"]["directory"],
            agent_cfg["experiment"]["experiment_name"],
        )
        
        self.checkpoint_modules = {}

        if agent_cfg["experiment"]["video_dir"] is not None:
            self.video_dir = agent_cfg["experiment"]["video_dir"]
        else:
            self.video_dir = "./videos"

        self.last_uploaded = set()

        if agent_cfg["experiment"]["upload_videos"]:
            os.makedirs(self.video_dir, exist_ok=True)

        self.get_new_log_path()

        # don't save anything if we are just playin
        if play:
            self.wandb_session = None
            self.tb_writer = None
            return

        # create wandb session
        if not delay_wandb_startup:
            self.setup_wandb()
        else:
            self.wandb_session = None



    def get_new_log_path(self):
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.log_root_path, run_time)
        # save metrics for plotting
        self.setup_tb_writer()

        # setup checkpoint saving
        if self.save_checkpoints > 0:
            self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_wandb(self, name=None):

        if name is None:
            name = self.exp_cfg["wandb_kwargs"]["name"]
        # setup wandb
        if self.exp_cfg["wandb"] == True:

            code_to_save = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
            wandb.init(
                project=self.exp_cfg["wandb_kwargs"]["project"],
                entity=self.exp_cfg["wandb_kwargs"]["entity"],
                group=self.exp_cfg["wandb_kwargs"]["group"],
                name=str(name),
                config=self.cfg_to_save,
                settings=wandb.Settings(code_dir=code_to_save),
            )
            # global step is what all metrics are logged against, and must be included as a key in the log dict
            wandb.define_metric("global_step")
            self.wandb_session = wandb
        else:
            self.wandb_session = None

    def close_wandb(self):
        if self.wandb_session is not None:
            self.wandb_session.finish()

    def setup_tb_writer(self):
        # tensorboard writer
        if self.exp_cfg["tb_log"]:
            self.tb_writer = SummaryWriter(self.log_dir)
            print("Created tensorboard summary writer")
        else:
            self.tb_writer = None

    def write_checkpoint(self, mean_eval_return: float, timestep: int) -> None:
        """Write checkpoint (modules) to disk

        The checkpoints are saved in the directory 'checkpoints' in the log directory.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f"mean_eval_return", mean_eval_return, global_step=timestep)

        if self.save_checkpoints == 0:
            return
        # checkpoint_modules is a dict of "policy", "value", and "optimiser"
        if mean_eval_return > self.checkpoint_best_modules["reward"]:
            self.checkpoint_best_modules["timestep"] = timestep
            self.checkpoint_best_modules["reward"] = mean_eval_return
            self.checkpoint_best_modules["saved"] = False
            self.checkpoint_best_modules["modules"] = {
                k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()
            }

        tag = str(timestep)
        # if self.video_dir is not None:
        #     self.log_videos(timestep)

        # save this checkpoint no matter what
        if self.save_checkpoints == 2:
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = self._get_internal_value(module)
            print("saving", f"agent_{tag}.pt")
            torch.save(modules, os.path.join(self.checkpoint_dir, f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = self.checkpoint_best_modules["modules"][name]
            torch.save(modules, os.path.join(self.checkpoint_dir, f"best_agent.pt"))
            print("New best reward, saving to best_agent.pt")

            self.checkpoint_best_modules["saved"] = True

    def _get_internal_value(self, _module):
        """Get internal module/variable state/value"""
        return _module.state_dict() if hasattr(_module, "state_dict") else _module

    def log_videos(self, step):
        # Look for new video files
        video_paths = glob.glob(os.path.join(self.video_dir, "*.mp4"))

        for path in video_paths:
            if path not in self.last_uploaded:
                wandb.log({"agent_video": wandb.Video(path, fps=4, format="mp4")}, step=step)
                self.last_uploaded.add(path)
