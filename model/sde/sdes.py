"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings
import math
import scipy.special as sc
import numpy as np
import torch


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, T, N):
        """Construct an SDE."""
        super().__init__()
        self.dt = T / N

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)
            # y: terminal

        Returns:
            f, G (without noise, (b,1,1,1) )
        """
        drift, diffusion = self.sde(x, t, *args)
        f = drift * self.dt
        G = diffusion * torch.sqrt(torch.tensor(self.dt, device=t.device))
        return f, G

    def reverse_discretize(self, x, t, score_fn, *args):
        # score_fn  partial
        drift, diffusion = self.sde(x, t, *args)
        rev_f = (-drift + diffusion ** 2 * score_fn(x, t)) * self.dt
        rev_G = diffusion * torch.sqrt(torch.tensor(self.dt, device=t.device))
        return rev_f, rev_G


class BBED(SDE):

    def __init__(self, k=2.6, c=0.51, Trs=0.999, T=1.0, N=30):
        """Construct an Brownian Bridge with Exploding Diffusion Coefficient SDE with parameterization as in the paper.
        dx = (y-x)/(T-t) dt + sqrt(c)*k^t dw
        """
        super().__init__(Trs, N)
        self.k = k
        self.logk = np.log(self.k)
        self.c = c
        self.N = N
        self.Eilog = sc.expi(-2 * self.logk)
        self.T = T  # The theoretical terminal time of SDE
        self.Trs = Trs  # The reverse SDE from Trs in inference
        self.dt = Trs / N

    def sde(self, x, t, y):
        # x:(b,c,t,f), t:(b,), y:(b,c,t,f)
        # drift:(b,c,t,f), diffusion:(b,1,1,1)
        t = t[:, None, None, None]
        drift = (y - x) / (self.T - t)
        diffusion = np.sqrt(self.c) * self.k ** t
        return drift, diffusion

    def mean(self, x0, t, y):
        time = (t / self.T)[:, None, None, None]
        mean = x0 * (1 - time) + y * time
        return mean

    def std(self, t):
        # t:(b,), return: (b,1,1,1)
        t = t[:, None, None, None]
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2 * (t_np - 1) * self.logk) - self.Eilog
        h = 2 * self.k ** 2 * self.logk
        var = (self.k ** (2 * t_np) - 1 + t_np) + h * (1 - t_np) * Eis
        var = torch.tensor(var).to(device=t.device) * (1 - t) * self.c
        return torch.sqrt(var)

    def marginal_prob(self, x0, t, y):
        return self.mean(x0, t, y), self.std(t)

    def prior_sampling(self, y):
        # y:(b,c,t,f)
        std = self.std(self.Trs * torch.ones((y.shape[0],), device=y.device))  # (b,1,1,1)
        z = torch.randn_like(y)
        x_T = y + z * std
        return x_T, z

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for BBED not yet implemented!")


if __name__ == '__main__':
    bbed = BBED()
    x = torch.randn(1, 1, 10, 10)
    # t = torch.tensor([0.7, ])
    t = torch.tensor([0.999, ])
    y = torch.randn(1, 1, 10, 10)
    # f, G = bbed.discretize(x, t, y)
    print(bbed.std(t))
