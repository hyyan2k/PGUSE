import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class RNNAttention(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_heads=4,
            dropout_p=0.1,
            bidirectional=False,
    ):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.Wq = nn.Linear(hidden_dim * 2, emb_dim)
            self.Wk = nn.Linear(hidden_dim * 2, emb_dim)
            self.Wv = nn.Linear(hidden_dim * 2, emb_dim)
        else:
            self.Wq = nn.Linear(hidden_dim, emb_dim)
            self.Wk = nn.Linear(hidden_dim, emb_dim)
            self.Wv = nn.Linear(hidden_dim, emb_dim)
        self.Wo = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x:(b,t,d)
        B, T, _ = x.size()
        x, _ = self.rnn(x)  # (b,t,2*h)
        q = self.Wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (b,h,t,d/h)
        k = self.Wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (b,h,t,d/h)
        v = self.Wv(x).reshape(B, T, self.n_heads, -1).transpose(1, 2)  # (b,h,t,d/h)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.mul(attn, self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (b,h,t,d/h)

        out = out.transpose(1, 2).reshape(B, T, -1)  # (b,t,d)
        out = self.Wo(out)

        return out


class DualPathRNNAttention(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_freqs=32,
            n_heads=4,
            dropout_p=0.1,
            temb_dim=None,
    ):
        super().__init__()
        self.intra_norm = nn.LayerNorm([n_freqs, emb_dim])
        self.intra_rnn_attn = RNNAttention(emb_dim, hidden_dim, n_heads, dropout_p, bidirectional=True)


        self.inter_norm = nn.LayerNorm([n_freqs, emb_dim])
        self.inter_rnn_attn = RNNAttention(emb_dim, hidden_dim, n_heads, dropout_p, bidirectional=True)

        if temb_dim is not None:
            self.intra_t_proj = nn.Linear(temb_dim, emb_dim)
            self.inter_t_proj = nn.Linear(temb_dim, emb_dim)

    def forward(self, x, temb=None):
        # x:(b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b,t,f,d)

        x_res = x
        if temb is not None:
            x = x + self.intra_t_proj(temb)[:, None, None, :]
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)  # (b*t,f,d)
        x = self.intra_rnn_attn(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        x_res = x
        if temb is not None:
            x = x + self.inter_t_proj(temb)[:, None, None, :]
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)  # (b,f,t,d)
        x = x.reshape(B * F, T, D)
        x = self.inter_rnn_attn(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3)  # (b,t,f,d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x


class Former(nn.Module):
    def __init__(
            self,
            emb_dim=64,
            hidden_dim=128,
            n_freqs=32,
            n_heads=4,
            dropout_p=0.1,
            temb_dim=None,
    ):
        super().__init__()
        self.dp_rnn_attn = DualPathRNNAttention(emb_dim, hidden_dim, n_freqs, n_heads, dropout_p, temb_dim)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=4, dropout_p=dropout_p)

    def forward(self, x, temb=None):
        x = self.dp_rnn_attn(x, temb)
        x = self.conv_glu(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 32, 84, 32)
    m = Former(emb_dim=32, hidden_dim=64)
    print(sum([p.numel() for p in m.parameters()]))
    y = m(x)
    print(y.shape)
