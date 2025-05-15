import abc
import torch


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class LangevinCorrector(Corrector):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, x, t, score_fn, n_steps=1, snr=0.5):
        # score_fn partial
        x_mean = x
        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = ((snr * noise_norm / grad_norm) ** 2 * 2).unsqueeze(0)
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, x, t, y, score_fn, n_steps=1, snr=0.5):
        # score_fn partial
        x_mean = x
        std = self.sde.marginal_prob(x, t, y)[1]
        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (snr * std) ** 2 * 2
            x_mean = x + step_size * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)

        return x, x_mean


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snr = 0
        self.n_steps = 0

    def update_fn(self, x, t, *args):
        return x, x


if __name__ == '__main__':
    from sdes import BBED

    bbed = BBED()
    corrector = AnnealedLangevinDynamics(bbed)
    x = torch.randn(1, 1, 10, 10)
    t = torch.tensor([0.25, ])
    y = torch.randn(1, 1, 10, 10)
    x, x_mean = corrector.update_fn(x, t, y, score_fn=lambda x, t: x + 1)
