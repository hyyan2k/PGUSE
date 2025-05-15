import abc

import torch
import numpy as np


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(self, x, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            # t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, x, t, y, score_fn):
        # score_fn partial
        drift, diffusion = self.sde.sde(x, t, y)
        z = torch.randn_like(x)
        x_mean = x + (-drift + diffusion ** 2 * score_fn(x, t)) * self.sde.dt
        x = x_mean + diffusion * np.sqrt(self.sde.dt) * z
        return x, x_mean


class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_fn(self, x, t, *args):
        return x, x
