import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pesq import pesq
from pathlib import Path
import soundfile as sf
from torch_ema import ExponentialMovingAverage

from .backbone import ScoreNet
from .sde import Inferencer
from .loss import loss_fn


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = ScoreNet(**config['model_config'])
        self.ema = ExponentialMovingAverage(self.model.parameters(), **config['ema_config'])
        self.train_inferencer = Inferencer(**config['train_sde_config'])
        self.test_inferencer = Inferencer(**config['test_sde_config'])
        self.train_sde = self.train_inferencer.sde
        self.test_sde = self.test_inferencer.sde
        self.extract_feature_fn = self.model.extract_feature
        self.generate_wav = self.model.generate_wav
        self.forward_p = self.model.forward_p
        self.forward_g = self.model.forward_g

        self.current_traning_step = -1
    
    def forward(self, batch):
        pass
    

    # For ema
    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

    # For ema
    def train(self, mode=True):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if mode == False:
            # eval
            self.ema.store(self.model.parameters())        # store current params in EMA
            self.ema.copy_to(self.model.parameters())      # copy EMA parameters over current params for evaluation
        else:
            # train
            if self.ema.collected_params is not None:
                self.ema.restore(self.model.parameters())  # restore the EMA weights (if stored)
        return res

    # For ema
    def eval(self,):
        return self.train(False)

    # For ema
    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        src, tgt, length, _ = batch
        src_mag, src_ri, tgt_mag, tgt_ri = self.extract_feature_fn(src, tgt)

        x0, y = tgt_mag, src_mag
        t_eps = self.config['train_sde_config']['t_eps']
        t = torch.rand(x0.shape[0], device=x0.device) * (self.train_sde.Trs - t_eps) + t_eps
        mean, sigma = self.train_sde.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)
        x = mean + sigma * z
        est_ri, sigma_z = self.model(torch.cat([x, src_mag, src_ri], dim=1), t)
        loss = loss_fn(est_ri, sigma_z, tgt_ri, z, sigma)

        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True ,prog_bar=True)
        self.current_traning_step += 1
        
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        src, tgt, length, _ = batch
        src_mag, src_ri, tgt_mag, tgt_ri = self.extract_feature_fn(src, tgt)
        
        x0, y = tgt_mag, src_mag
        est_ri, feat_list = self.forward_p(torch.cat([src_mag, src_ri], dim=1))
        est_pha = torch.atan2(est_ri[:, 1], est_ri[:, 0])
        est_mag_p = torch.complex(est_ri[:, 0], est_ri[:, 1]).abs().unsqueeze(1)

        Trs = torch.tensor([self.test_sde.Trs], device=y.device)
        truncated_state = self.test_sde.mean(est_mag_p, Trs, y)
        score_fn = lambda x, t: -self.forward_g(x, feat_list, t) / self.test_sde.std(t) ** 2
        est_mag_g = self.test_inferencer.inference(truncated_state, score_fn)

        est_mag = self.config['p_g_alpha'] * est_mag_p + (1 - self.config['p_g_alpha']) * est_mag_g
        est = self.generate_wav(est_mag.squeeze(1), est_pha, length=src.size(-1))

        assert tgt.size(0) == 1
        tgt = tgt.squeeze(0).cpu().numpy()
        est = est.squeeze(0).cpu().numpy()
        pesq_score = pesq(16000, tgt, est, 'wb')
        
        self.log_dict({'val_pesq': pesq_score}, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_end(self,):
        # save ckpt when validation epoch finished
        if not self.trainer.sanity_checking:
            epoch = self.current_epoch
            step = self.current_traning_step
            pesq_score = self.trainer.callback_metrics['val_pesq']
            ckpt_name = f'epoch={epoch}-step={step}-pesq={pesq_score:.2f}.ckpt'
            self.trainer.save_checkpoint(self.config['ckpt_dir'] / ckpt_name)
    
    def test_step(self, batch, batch_idx):
        src, tgt, length, names = batch

        src_mag, src_ri, tgt_mag, tgt_ri = self.extract_feature_fn(src, tgt)
        x0, y = tgt_mag, src_mag
        est_ri, feat_list = self.forward_p(torch.cat([src_mag, src_ri], dim=1))
        est_pha = torch.atan2(est_ri[:, 1], est_ri[:, 0])
        est_mag_p = torch.complex(est_ri[:, 0], est_ri[:, 1]).abs().unsqueeze(1)

        Trs = torch.tensor([self.test_sde.Trs], device=y.device)
        truncated_state = self.test_sde.mean(est_mag_p, Trs, y)
        score_fn = lambda x, t: -self.forward_g(x, feat_list, t) / self.test_sde.std(t) ** 2
        est_mag_g = self.test_inferencer.inference(truncated_state, score_fn)

        est_mag = self.config['p_g_alpha'] * est_mag_p + (1 - self.config['p_g_alpha']) * est_mag_g
        est = self.generate_wav(est_mag.squeeze(1), est_pha, length=src.size(-1))

        assert tgt.size(0) == 1
        tgt = tgt.squeeze(0).cpu().numpy()
        est = est.squeeze(0).cpu().numpy()

        # save
        if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
            est = est / (np.max(np.abs(est)) + 1e-5)
            sf.write(Path(self.config['save_enhanced']) / f'{names[0]}.wav', est, samplerate=16000)
        
        pesq_score = pesq(16000, tgt, est, 'wb')
        self.log_dict({'test_pesq': pesq_score}, on_step=False, on_epoch=True)

    
    def on_test_epoch_end(self,):
        pass

    def on_save_checkpoint(self, ckpt):
        ckpt['current_traning_step'] = self.current_traning_step
        ckpt['ema'] = self.ema.state_dict()
    
    def on_load_checkpoint(self, ckpt):
        self.current_traning_step = ckpt['current_traning_step']
        self.ema.load_state_dict(ckpt['ema'])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), **self.config['opt'])
        sch = {
            'scheduler': torch.optim.lr_scheduler.StepLR(opt, **self.config['sch']), 
            'interval': 'epoch',
            'frequency': 1,
        }

        return [opt], [sch]


