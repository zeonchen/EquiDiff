import torch.nn as nn
import math
from model.gat_layer import DenseGATConv
from model.layers import TransformerBlock


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ContextEncoder(torch.nn.Module):
    def __init__(self, context_dim, input_size=2, hidden_size=16):
        super(ContextEncoder, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.l = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, context_dim))

        self.gnn_l1 = DenseGATConv(in_channels=context_dim, out_channels=context_dim, heads=4, concat=False)
        self.gnn_l2 = DenseGATConv(in_channels=context_dim, out_channels=context_dim, heads=4, concat=False)
        self.gnn_l3 = DenseGATConv(in_channels=context_dim, out_channels=context_dim, heads=4, concat=False)

    def forward(self, x, mask=None):
        batch_size, node_num, length, feat_dim = x.shape

        post_x = torch.zeros_like(x)
        post_x[:, :, :-1, :] = x[:, :, 1:, :]
        velocity = (post_x - x).pow(2).sum(-1, keepdim=True).sqrt()
        velocity = velocity[:, :,0 :-1, :]

        pre_velocity = torch.zeros_like(velocity)
        pre_velocity[:, :, 1:] = velocity[:, :, :-1]
        pre_velocity[:, :, 0] = velocity[:, :, 0]
        EPS = 1e-6
        vel_cosangle = torch.sum(pre_velocity * velocity, dim=-1) / (
                    (torch.norm(pre_velocity, dim=-1) + EPS) * (torch.norm(velocity, dim=-1) + EPS))

        vel_angle = torch.acos(torch.clamp(vel_cosangle, -1, 1)).unsqueeze(-1)

        x = torch.cat([velocity, vel_angle], dim=-1).view(batch_size*node_num, length-1, -1)

        # x = x.view(batch_size*node_num, length, -1)
        _, hidden = self.gru(x)
        # x = self.l(hidden[-1, :, :]).view(batch_size, node_num, -1)

        x = hidden[-1, :, :].view(batch_size, node_num, -1)
        adj = torch.ones(batch_size, node_num, node_num).to(x.device)
        x = self.gnn_l1(x, adj, mask)
        # x = self.gnn_l2(x, adj, mask)

        return x[:, 0, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class EquiDiffPlus(nn.Module):
    def __init__(self, context_dim, T):
        super().__init__()
        self.context = ContextEncoder(context_dim, hidden_size=context_dim)
        self.time_embedding = TimeEmbedding(T, context_dim // 4, context_dim)

        self.pos_emb = PositionalEncoding(d_model=2, dropout=0.1, max_len=25)

        self.init_l = nn.Linear(25, context_dim)
        self.end_l = nn.Linear(context_dim, 25)

        self.l1 = nn.Sequential(nn.Linear(context_dim*2, context_dim),
                                nn.ReLU(),
                                nn.Linear(context_dim, 2))
        self.l2 = nn.Sequential(nn.Linear(context_dim*2, context_dim),
                                nn.ReLU(),
                                nn.Linear(context_dim, 2))
        self.l3 = nn.Sequential(nn.Linear(context_dim*2, context_dim),
                                nn.ReLU(),
                                nn.Linear(context_dim, 2))
        self.l4 = nn.Sequential(nn.Linear(context_dim*2, context_dim),
                                nn.ReLU(),
                                nn.Linear(context_dim, 2))

        self.trans_layer1 = TransformerBlock(f_dim=context_dim, num_heads=4, bias_eps=1e-6)
        self.trans_layer2 = TransformerBlock(f_dim=context_dim, num_heads=4, bias_eps=1e-6)
        self.trans_layer3 = TransformerBlock(f_dim=context_dim, num_heads=4, bias_eps=1e-6)
        self.trans_layer4 = TransformerBlock(f_dim=context_dim, num_heads=4, bias_eps=1e-6)

    def forward(self, t, x, context, node_mask):
        batch_size, length, _ = x.shape
        context = context.view(batch_size, 1, -1)

        time_emb = self.time_embedding(t).unsqueeze(1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)

        x = self.l1(ctx_emb) * x
        x = self.init_l(x.transpose(1, 2)).transpose(1, 2)

        trans = self.trans_layer1(x.unsqueeze(-1))
        trans = self.trans_layer2(trans)
        trans = self.trans_layer3(trans)
        trans = self.trans_layer4(trans)
        node_loc = trans.squeeze(-1)
        node_loc = self.end_l(node_loc.transpose(1, 2)).transpose(1, 2)

        node_loc = self.l2(ctx_emb) * node_loc
        node_loc = self.l3(ctx_emb) * node_loc
        node_loc = self.l4(ctx_emb) * node_loc

        return node_loc
