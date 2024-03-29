import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class Linear(nn.Module):
    # Section 7
    def __init__(self, in_features, out_features, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float32))
        nn.init.uniform_(self.weight, -1/np.sqrt(in_features), 1/np.sqrt(in_features))

        self.bias = nn.Parameter(torch.zeros((out_features, 2), dtype=torch.float32))
        nn.init.uniform_(self.bias, -1/np.sqrt(in_features), 1/np.sqrt(in_features))

    def forward(self, x):
        '''
        x: tensor of shape [B, in_features, 3, num_points]
        return: tensor of shape [B, out_features, 3, num_points]
        '''
        x_out = torch.matmul(x.transpose(1,-1), self.weight).transpose(1,-1)
        u = self.eps * self.bias / torch.norm(self.bias, dim=1, keepdim=True)
        return x_out + u.unsqueeze(0).unsqueeze(-1)


class LeakyReLU(nn.Module):
    def __init__(self, in_features, negative_slope=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((in_features, in_features), dtype=torch.float32))
        nn.init.uniform_(self.weight, -1/np.sqrt(in_features), 1/np.sqrt(in_features))

        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: tensor of shape [B, F, 3, N]
        return: tensor of shape [B, F, 3, N]
        '''
        d = torch.matmul(x.transpose(1,-1), self.weight).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) \
                * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class ReLU(LeakyReLU):
    def __init__(self, in_features):
        super().__init__(in_features, negative_slope=0.0)


class MeanProject(nn.Module):
    # Section 6
    def __init__(self, n_emb: int, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((n_emb, out_features, in_features), dtype=torch.float32))
        nn.init.uniform_(self.weight, -1/np.sqrt(in_features), 1/np.sqrt(in_features))

    def forward(self, x):
        '''
        x: tensor of shape [B, C, 3, N]
        return: tensor of shape [B, C', 3, M]
        '''
        x_mean = torch.mean(x, dim=-1)
        out = torch.matmul(self.weight, x_mean.unsqueeze(1))
        out = out.permute(0,2,3,1)
        return out


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class LayerNorm(nn.Module):
    # Section 4.3
    def __init__(self, num_features):
        super().__init__()
        self.layernorm = nn.LayerNorm(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_points]
        '''
        norm = torch.norm(x, dim=2) + EPS
        norm_ln = self.layernorm(norm.transpose(1,2)).transpose(1,2)
        x = x / norm.unsqueeze(2) * norm_ln.unsqueeze(2)

        return x


class MultiHeadAttention(nn.Module):
    # Section 4.2
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        assert emb_dim % num_heads == 0, 'Embeddings must split evenly amongst heads'

        self.num_heads = num_heads
        self.emb_dim = emb_dim

        self.W_q = Linear(emb_dim, emb_dim)
        self.W_k = Linear(emb_dim, emb_dim)
        self.W_z = Linear(emb_dim, emb_dim)
        self.W_o = Linear(emb_dim, emb_dim)

    def forward(self, Q, K, Z):
        '''
        Q : tensor of shape [B, C, 3, M]
        K : tensor of shape [B, C, 3, N]
        Z : tensor of shape [B, C, 3, N]
        return: tensor of shape [B, C, 3, M]
        '''
        B = Q.size(0)

        Q_heads = split_heads(self.W_q(Q), self.num_heads)
        K_heads = split_heads(self.W_k(K), self.num_heads)
        Z_heads = split_heads(self.W_z(Z), self.num_heads)

        frob_norm = frob_inner_product(Q_heads, K_heads)
        constant = 1 / np.sqrt(2 * Q_heads.size(2))
        A_heads = F.softmax(constant * frob_norm, dim=-1)
        out = torch.einsum('bhcdn,bhmn->bhcdm', Z_heads, A_heads)
        y = self.W_o(out.flatten(start_dim=1, end_dim=2))
        return y


class MaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


class TransformerBlock(nn.Module):
    '''Figure 2'''
    def __init__(self,
                 f_dim,
                 num_heads,
                 bias_eps=1e-6,
                 leaky=0,
                ):
        super().__init__()
        self.attention = MultiHeadAttention(f_dim, num_heads)
        self.layer_norm1 = LayerNorm(f_dim)

        self.mlp = nn.Sequential(
            Linear(f_dim, f_dim),
            BatchNorm(f_dim),
            LeakyReLU(f_dim, leaky),
            # in the original Attention is all.. the FF has Linear,ReLU,Linear
            # but that is not how vn-tfm describes the mlp
            Linear(f_dim, f_dim),
        )
        self.layer_norm2 = LayerNorm(f_dim)

    def forward(self, x, queries=None):
        '''
        use queries for PerceiverIO-style encoder (queries = MeanProject(x)) or
        decoder (queries = original pc representation)
        '''
        if queries is None:
            queries = x

        identity = queries

        x = self.attention(queries, x, x)
        x = self.layer_norm1(x)
        x += identity

        new_identity = x
        x = self.mlp(x)
        x = self.layer_norm2(x)
        x += new_identity

        return x


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


def frob_inner_product(x, y):
    '''
    x: tensor of shape [*,C,3,M]
    y: tensor of shape [*,C,3,N]
    returns: tensor of shape [*, M, N]
    '''
    x = x.reshape(*x.shape[:-3], -1, x.shape[-1]).transpose(-1, -2)
    y = y.reshape(*y.shape[:-3], -1, y.shape[-1])
    return torch.matmul(x, y)


def invariant(x, y):
    '''
    x: tensor of shape [B,C,3,N]
    y: tensor of shape [B,3,3,N]
    returns: tensor of shape [B, C, 3, N]
    '''
    return torch.einsum('bcin,bjin->bcjn', x, y)


def rotate_point_features(x, rot):
    '''
    x: tensor of shape [B,C,3,N]
    rot: tensor of shape [B,3,3]
    returns: tensor of shape [B, C, 3, N]
    '''
    return torch.einsum('bctn,btj->bcjn', x, rot)


def split_heads(x, num_heads):
    '''
    x: tensor of shape [B,C,3,N]
    return: tensor of shape [B,H,C/H,3,N]
    '''
    B,C,_,N = x.shape
    assert C % num_heads == 0
    return x.view(B, num_heads, C//num_heads, 2, N)