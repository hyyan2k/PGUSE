import torch
from .sdes import BBED
from .predictors import EulerMaruyamaPredictor
from .correctors import AnnealedLangevinDynamics


class Inferencer:
    def __init__(
            self,
            k=2.6,
            c=0.51,
            Trs=0.999,
            T=1.0,
            N=30,
            t_eps=3e-2,
            n_steps=1,
            snr=0.5,
    ):
        self.sde = BBED(k=k, c=c, Trs=Trs, T=T, N=N)
        self.predictor = EulerMaruyamaPredictor(self.sde)
        self.corrector = AnnealedLangevinDynamics(self.sde)
        assert t_eps <= Trs / N
        self.time_steps = torch.linspace(Trs, Trs / N, N).tolist()  # [0.999,...,0.0333]
        self.n_steps = n_steps
        self.snr = snr
        self.nfe = N * (n_steps + 1)

    def inference(self, y, score_fn):
        x, _ = self.sde.prior_sampling(y)
        for t in self.time_steps:
            t = torch.ones(y.shape[0], device=y.device) * t
            x, x_mean = self.corrector.update_fn(x, t, y, score_fn, self.n_steps, self.snr)
            x, x_mean = self.predictor.update_fn(x, t, y, score_fn)
        return x_mean
