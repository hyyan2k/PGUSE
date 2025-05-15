import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .former import Former, CustomLayerNorm


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x = torch.log(x)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (b, d)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs, temb_dim=None):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.high_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), stride=(1, 3), padding=1)
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, out_channels)

    def forward(self, x, temb=None):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]

        x = self.norm(x)
        x = self.act(x)
        return x


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs, temb_dim=None):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), r=3)

        self.norm = CustomLayerNorm((1, n_freqs * 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)
        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, out_channels)

    def forward(self, x, temb=None):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]
        x = self.norm(x)
        x = self.act(x)

        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=64, temb_dim=None):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )

        self.conv_2 = DSConv(num_channels // 4, num_channels // 2, n_freqs=257, temb_dim=temb_dim)
        self.conv_3 = DSConv(num_channels // 2, num_channels // 4 * 3, n_freqs=128, temb_dim=temb_dim)
        self.conv_4 = DSConv(num_channels // 4 * 3, num_channels, n_freqs=64, temb_dim=temb_dim)

    def forward(self, x, temb=None):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x, temb)
        out_list.append(x)  # 128
        x = self.conv_3(x, temb)
        out_list.append(x)  # 64
        x = self.conv_4(x, temb)
        out_list.append(x)  # 32
        return out_list


class Decoder(nn.Module):
    def __init__(self, num_channels=64, temb_dim=None, out_channels=1):
        super(Decoder, self).__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32, temb_dim=temb_dim)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64, temb_dim=temb_dim)  # 128
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128, temb_dim=temb_dim)  # 256
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels // 4, num_channels // 4, (3, 2), padding=1),  # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
            nn.Conv2d(num_channels // 4, out_channels, (1, 1)),
        )

    def forward(self, x, encoder_out_list, temb=None):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1), temb)  # 64
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1), temb)  # 128
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1), temb)  # 256
        x = self.conv(x)  # (B,1,T,F)
        return x


class Interaction(nn.Module):
    def __init__(self, num_channels, temb_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=(5, 5), padding=(2, 2))
        self.norm = CustomLayerNorm((1, 32), stat_dims=(1, 3))
        self.sigmoid = nn.Sigmoid()
        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, num_channels)

    def forward(self, feat_g, feat_p, temb=None):
        x = self.conv(feat_g + feat_p)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]
        x = self.norm(x)
        mask = self.sigmoid(x)
        outs = mask * feat_p + feat_g
        return outs


class ScoreNet(nn.Module):
    def __init__(
            self, 
            num_channels=64, 
            temb_dim=256, 
            n_blocks=3,
            n_heads=4,
            dropout_p=0.1,
            n_fft=512, 
            hop_length=192,
        ):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length

        self.embed = nn.Sequential(
            GaussianFourierProjection(temb_dim // 2),
            nn.Linear(temb_dim // 2, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
        )

        self.encoder_g = Encoder(in_channels=1, num_channels=num_channels, temb_dim=temb_dim)
        self.encoder_p = Encoder(in_channels=3, num_channels=num_channels, temb_dim=None)
        self.blocks_g = nn.ModuleList(
            [Former(
                emb_dim=num_channels,
                hidden_dim=num_channels * 2,
                n_freqs=self.n_freqs // 2 ** 3,
                n_heads=n_heads,
                dropout_p=dropout_p,
                temb_dim=None,
            ) for _ in range(n_blocks)]
        )
        self.blocks_p = nn.ModuleList(
            [Former(
                emb_dim=num_channels,
                hidden_dim=num_channels * 2,
                n_freqs=self.n_freqs // 2 ** 3,
                n_heads=n_heads,
                dropout_p=dropout_p,
                temb_dim=None,
            ) for _ in range(n_blocks)]
        )
        self.decoder_g = Decoder(num_channels=num_channels, temb_dim=temb_dim, out_channels=1)
        self.decoder_p = Decoder(num_channels=num_channels, temb_dim=None, out_channels=2)
        self.interactions = nn.ModuleList([Interaction(num_channels=num_channels, temb_dim=temb_dim) for _ in range(n_blocks + 1)])

    def apply_stft(self, x, return_complex=True):
        # x:(B,T)
        assert x.ndim == 2
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)  # (B,T,F)
        return spec

    def apply_istft(self, x, length=None):
        # x:(B,T,F)
        assert x.ndim == 3
        x = x.transpose(1, 2)  # (B,F,T)
        audio = torch.istft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            length=length,
            return_complex=False
        )  # (B,T)
        return audio

    @staticmethod
    def power_compress(x):
        # x:(B,T,F)
        mag = torch.abs(x) ** 0.3 * 0.3
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    @staticmethod
    def power_uncompress(x):
        # x:(B,T,F)
        mag = (torch.abs(x) / 0.3) ** (1.0 / 0.3)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))


    def extract_feature(self, src, tgt=None):
        if tgt is None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))  # (B,T,F)
        src_mag = src_spec.abs().unsqueeze(1)
        src_ri = torch.stack([src_spec.real, src_spec.imag], dim=1)

        tgt_spec = self.power_compress(self.apply_stft(tgt))  # (B,T,F)
        tgt_mag = tgt_spec.abs().unsqueeze(1)
        tgt_ri = torch.stack([tgt_spec.real, tgt_spec.imag], dim=1)

        return src_mag, src_ri, tgt_mag, tgt_ri


    def generate_wav(self, est_mag, est_pha, length):
        est_mag = torch.clip(est_mag, min=0)
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est_audio = self.apply_istft(self.power_uncompress(est_spec), length=length)
        return est_audio


    def forward_p(self, src_mag_ri):
        # x: (b,3,t,f)
        encoded_list = self.encoder_p(src_mag_ri)
        x = encoded_list[-1]

        feat_list = []
        feat_list.append(x)
        for block in self.blocks_p:
            x = block(x)
            feat_list.append(x)
        x = self.decoder_p(x, encoded_list)  # (B,2,T,F)

        return x, feat_list
    

    def forward_g(self, x, feat_list, t):
        # x: (b,1,t,f)
        temb = self.embed(t)
        encoded_list = self.encoder_g(x, temb)
        x = encoded_list[-1]

        x = self.interactions[0](x, feat_list[0], temb)
        for idx, block in enumerate(self.blocks_g):
            x = block(x)
            x = self.interactions[idx+1](x, feat_list[idx+1], temb)
        
        x = self.decoder_g(x, encoded_list, temb)  # (B,1,T,F)
        return x


    def forward(self, x, t=None):
        # x: (b,4,t,f), t: (b,)
        if t is None:
            t = torch.tensor([0.999,], device=x.device)

        xt, src_mag_ri = x[:, :1], x[:, 1:]
        est_ri, feat_list = self.forward_p(src_mag_ri)
        sigma_z = self.forward_g(xt, feat_list, t)
        return est_ri, sigma_z


if __name__ == '__main__':
    m = ScoreNet()
    x = torch.randn(1, 4, 84, 257)
    t = torch.rand(1, )
    est_ri, sigma_z = m(x, t)
    print(est_ri.shape)
    print(sigma_z.shape)
