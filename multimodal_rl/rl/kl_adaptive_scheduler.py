"""Adaptive KL scheduler.

Adjusts the learning rate according to the KL divergence.

Copied from SKRL
"""
from __future__ import annotations

from packaging import version

import torch
from torch.optim.lr_scheduler import _LRScheduler


class KLAdaptiveLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        kl_threshold: float = 0.008,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        kl_factor: float = 2,
        lr_factor: float = 1.5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Adaptive KL scheduler.

        Adjusts the learning rate according to the KL divergence.

        .. note::

            This scheduler is only available for the A2C, AMP, PPO and RPO single-agent algorithms,
            and IPPO and MAPPO multi-agent algorithms. Applying it to other agents will not change the learning rate.

        :param optimizer: Wrapped optimizer.
        :param kl_threshold: Threshold for KL divergence.
        :param min_lr: Lower bound for learning rate.
        :param max_lr: Upper bound for learning rate.
        :param kl_factor: The number used to modify the KL divergence threshold.
        :param lr_factor: The number used to modify the learning rate.
        :param last_epoch: The index of last epoch.
        :param verbose: Verbose mode.

        Example::

            >>> scheduler = KLAdaptiveLR(optimizer, kl_threshold=0.01)
            >>> for epoch in range(100):
            >>>     # ...
            >>>     kl_divergence = ...
            >>>     scheduler.step(kl_divergence)
        """
        if version.parse(torch.__version__) >= version.parse("2.7"):
            super().__init__(optimizer, last_epoch)
        else:
            if version.parse(torch.__version__) >= version.parse("2.2"):
                verbose = "deprecated"
            super().__init__(optimizer, last_epoch, verbose)

        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._kl_factor = kl_factor
        self._lr_factor = lr_factor

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, kl: torch.Tensor | float | None = None, *, epoch: int | None = None) -> None:
        """
        Step scheduler.

        :param kl: KL divergence. If None, no adjustment is made. If tensor, the number of elements must be 1.
        :param epoch: Epoch.

        Example::

            >>> kl = torch.distributions.kl_divergence(p, q)
            >>> kl
            tensor([0.0332, 0.0500, 0.0383,  ..., 0.0076, 0.0240, 0.0164])
            >>> scheduler.step(kl.mean())

            >>> kl = 0.0046
            >>> scheduler.step(kl)
        """
        if kl is not None:
            # Ensure kl is a float
            if isinstance(kl, torch.Tensor):
                kl = float(kl.item())
            else:
                kl = float(kl)
            
            for group in self.optimizer.param_groups:
                # Ensure lr is a float (handle case where it might be a tensor or string)
                current_lr_raw = group["lr"]
                if isinstance(current_lr_raw, torch.Tensor):
                    current_lr = float(current_lr_raw.item())
                elif isinstance(current_lr_raw, (int, float)):
                    current_lr = float(current_lr_raw)
                else:
                    # Try to convert string or other types
                    try:
                        current_lr = float(current_lr_raw)
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Learning rate must be numeric, got {type(current_lr_raw)}: {current_lr_raw}. "
                            f"Original error: {e}"
                        )
                
                # Ensure all comparison values are floats
                min_lr = float(self.min_lr)
                max_lr = float(self.max_lr)
                lr_factor = float(self._lr_factor)
                kl_threshold = float(self.kl_threshold)
                kl_factor = float(self._kl_factor)
                
                if kl > kl_threshold * kl_factor:
                    new_lr = max(current_lr / lr_factor, min_lr)
                    group["lr"] = float(new_lr)
                elif kl < kl_threshold / kl_factor:
                    new_lr = min(current_lr * lr_factor, max_lr)
                    group["lr"] = float(new_lr)

            self._last_lr = [
                float(group["lr"]) if not isinstance(group["lr"], torch.Tensor) 
                else float(group["lr"].item()) 
                for group in self.optimizer.param_groups
            ]