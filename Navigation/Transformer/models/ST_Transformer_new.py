import torch
import torch.nn as nn
from .GCN_models import GCN
from .transformer import husformerEncoder
import numpy as np
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, n_mask = None):
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        if n_mask is not None:
            n_mask = n_mask.to(scores.device)
            n_mask = n_mask.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            scores.masked_fill_(n_mask, -1e9)
        attn = F.softmax(scores, dim = -1)
        context = torch.matmul(attn, V)
        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V, spatial_n_mask = None):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)

        context = ScaledDotProductAttention()(Q, K, V, spatial_n_mask)
        context = context.permute(0, 3, 2, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)

        output = self.fc_out(context)
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V, temporal_n_mask = None):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)



        context = ScaledDotProductAttention()(Q, K, V, temporal_n_mask)
        context = context.permute(0, 2, 3, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)

        output = self.fc_out(context)
        return output


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, cheb_K, dropout, forward_expansion, device):
        super(STransformer, self).__init__()

        self.device = device
        self.embed_size = embed_size
        self.input_fc = nn.Linear(self.embed_size, self.embed_size)
        self.embed_liner = nn.Linear(5, embed_size)
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
                            nn.Linear(embed_size, forward_expansion * embed_size),
                            nn.ReLU(),
                            nn.Linear(forward_expansion * embed_size, embed_size),
                            )
        
        
        self.gcn = GCN(embed_size, embed_size*2, embed_size, cheb_K, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)


    def forward(self, value, key, query, n_mask = None, adj = None):

        self.D_S = adj.to(self.device)

        B, N, T, C = query.shape
        D_S = self.embed_liner(self.D_S)
        D_S = D_S.permute(0, 2, 1, 3)
        X_G = torch.Tensor(B, N, T, C).to(self.device)
        X_G_3d = torch.Tensor(N, 0, C).to(self.device)

        
        for k in range(query.shape[0]):
            for t in range(query.shape[2]):
                o = self.gcn(query[k, :, t, :].unsqueeze(0), adj[k, t])
                o = o.permute(1, 0, 2)

                X_G[k] = torch.cat((X_G_3d, o), dim = 1)

        query = query + D_S
        attention = self.attention(query, query, query, n_mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) +  self.fg(X_G))
        out = g*U_S + (1-g)*X_G

        return out


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion, device):
        super(TTransformer, self).__init__()

        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size)
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
                            nn.Linear(embed_size, forward_expansion * embed_size),
                            nn.ReLU(),
                            nn.Linear(forward_expansion * embed_size, embed_size)
                            )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, value, key, query, temporal_n_mask = None):
        B, N, T, C = query.shape

        D_T = self.temporal_embedding(torch.arange(0, T).to(self.device))
        D_T = D_T.expand(B, N, T, C)

        query = query + D_T
        attention = self.attention(query, query, query, temporal_n_mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, time_num, cheb_K, dropout, forward_expansion, device):
        super(STTransformerBlock, self).__init__()

        self.STransformer = STransformer(embed_size, heads, cheb_K, dropout, forward_expansion, device)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion, device)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, spatial_n_mask = None, temporal_n_mask = None, adj = None):
        # Add skip connection,run through normalization and finally dropout
        x1 = self.dropout(self.norm1(self.STransformer(value, key, query, spatial_n_mask, adj) + query))
        x2 = self.dropout(self.norm2(self.TTransformer(value, key, query, temporal_n_mask) + query))
        return x1, x2


class Encoder(nn.Module):
    def __init__(self, embed_size, heads, time_num, forward_expansion, cheb_K, dropout, device):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.STEncoder = STTransformerBlock(embed_size, heads, time_num, cheb_K, dropout, forward_expansion, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_n_mask = None, temporal_n_mask = None, adj = None):
        out = self.dropout(x)
        out1, out2 = self.STEncoder(out, out, out, spatial_n_mask, temporal_n_mask, adj)

        return out1, out2


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, forward_expansion, cheb_K, dropout, device):
        super(Transformer, self).__init__()

        self.encoder = Encoder(embed_size, heads, time_num, forward_expansion, cheb_K, dropout, device)

    def forward(self, src, spatial_n_mask = None, temporal_n_mask = None, adj = None):
        enc_src1, enc_src2 = self.encoder(src, spatial_n_mask, temporal_n_mask, adj)
        return enc_src1, enc_src2


class STTransformer(nn.Module):
    def __init__(self, args, device):
        super(STTransformer, self).__init__()

        self.device = device

        self.args = args
        self.heads = self.args.heads
        self.time_num = self.args.time_num
        self.in_channels = self.args.in_channels
        self.forward_expansion = self.args.forward_expansion
        self.cheb_K = self.args.cheb_K
        self.dropout = self.args.dropout

        self.embed_size = self.args.embed_size
        self.husformer_embed = self.args.husformer_embed
        self.husformer_heads = self.args.husformer_heads
        self.husformer_layers = self.args.husformer_layers
        self.attn_dropout = self.args.attn_dropout
        self.relu_dropout = self.args.relu_dropout
        self.res_dropout = self.args.res_dropout
        self.embed_dropout = self.args.embed_dropout
        self.attn_mask = self.args.attn_mask

        self.linear1 = nn.Linear(self.args.in_channels, self.args.embed_size)
        self.linear2 = nn.Linear(self.args.embed_size, self.args.in_channels)
        self.conv1 = nn.Conv2d(2, 1, 1)

        self.Transformer = Transformer(
                           self.embed_size,
                           self.heads,
                           self.time_num,
                           self.forward_expansion,
                           self.cheb_K,
                           self.dropout,
                           device = self.device
                           )

        self.ST_cross_attention = husformerEncoder(
                                  embed_dim = self.husformer_embed,
                                  num_heads = self.husformer_heads,
                                  layers = self.husformer_layers,
                                  attn_dropout = self.attn_dropout,
                                  relu_dropout = self.attn_dropout,
                                  res_dropout = self.res_dropout,
                                  embed_dropout = self.embed_dropout,
                                  attn_mask = self.attn_mask
                                  )

        self.TS_cross_attention = husformerEncoder(
                                  embed_dim = self.husformer_embed,
                                  num_heads = self.husformer_heads,
                                  layers = self.husformer_layers,
                                  attn_dropout = self.attn_dropout,
                                  relu_dropout = self.attn_dropout,
                                  res_dropout = self.res_dropout,
                                  embed_dropout = self.embed_dropout,
                                  attn_mask = self.attn_mask
                                  )

        self.self_trans = husformerEncoder(
                                  embed_dim = self.husformer_embed,
                                  num_heads = self.husformer_heads,
                                  layers = self.husformer_layers,
                                  attn_dropout = self.attn_dropout,
                                  relu_dropout = self.attn_dropout,
                                  res_dropout = self.res_dropout,
                                  embed_dropout = self.embed_dropout,
                                  attn_mask = self.attn_mask
                                  )

    def set_device(self, device):
        self.device = device
        self.to(device)

    def get_spatial_mask(self, x):
        B, N, T, C = x.shape
        mask = torch.zeros((B, T, N, N))
        for i in range(B):
            for j in range(T):
                mask[i, j, :, :] = torch.eye(N)
                traj = x[i, :, j, :].cpu()
                index = np.where(traj[:, 0] == 0)[0]
                mask[i, j, index, :] = 1
                mask[i, j, :, index] = 1

        return mask.bool()

    def get_temporal_mask(self, x):
        B, N, T, C = x.shape
        mask = torch.zeros((B, N, T, T))
        for i in range(B):
            for j in range(N):
                mask[i, j, :, :] = torch.eye(T)
                traj = x[i, j, :, :].cpu()
                index = np.where(traj[:, 0] == 0)[0]
                mask[i, j, index, :] = 1
                mask[i, j, :, index] = 1

        return mask.bool()

    def get_adj(self, x):
        B, N, T, C = x.shape
        adj = torch.zeros((B, T, N, N)).to(x.device)
        for i in range(B):
            for j in range(T):
                traj = x[i, :, j, :]
                dis = self.calc_dis(traj)
                adj[i, j] = dis

        return adj

    def calc_dis(self, traj):
        adj = torch.ones((len(traj), len(traj))).to(traj.device)
        for i in range(len(traj)):
            agent = traj[i]
            for j in range(len(traj)):
                if j == i:
                    continue
                neighbor = traj[j]
                dis = float((torch.sum((agent - neighbor) ** 2)) ** 0.5)
                adj[i, j] = dis
                adj[j, i] = dis

        return adj


    def forward(self, state, past_states = None, adj = None):

        if past_states:
            state = past_states.append(state)
            state = torch.stack(state, dim = 0).permute(1, 2, 0, 3) # [B, N, T, C]

        else:
            state = state.unsqueeze(0).permute(1, 2, 0, 3)

        st_feat = state.copy()          # [B, N, T, C]
        traj = st_feat[:, :, :2]        # [B, N, T, 2]
        if adj is None:
            adj_ = self.get_adj(traj)      # [B, T, N, N]

        else:
            adj_ = adj

        spatial_n_mask = self.get_spatial_mask(traj)    # [B, T, N, N]
        temporal_n_mask = self.get_temporal_mask(traj)  # [B, N, T, T]

        input_transformer = self.linear1(st_feat)  # [B, N, T, C]   C: 8 -> 32
        spatial_feat, temporal_feat = self.Transformer(input_transformer, spatial_n_mask, temporal_n_mask, adj_)  # [B, N, T, C]

        B, N, T, C = spatial_feat.shape

        spatial_feat = spatial_feat.permute(2, 0, 1, 3).reshape((1, -1, C))        # [B, N, 1, C] -> [1, B, N, C] -> [1, B * N, C]
        temporal_feat = temporal_feat.permute(2, 0, 1, 3).reshape((1, -1, C))      # [B, N, 1, C] -> [1, B, N, C] -> [1, B * N, C]
        spatial_temporal_feat = torch.cat((spatial_feat, temporal_feat), dim = 0)  # [2, B * N, C]

        spatial_feat2 = self.ST_cross_attention(spatial_feat, spatial_temporal_feat, spatial_temporal_feat)    # [1, B * N, C]
        temporal_feat2 = self.TS_cross_attention(temporal_feat, spatial_temporal_feat, spatial_temporal_feat)  # [1, B * N, C]
        cross_feat = torch.cat((spatial_feat2, temporal_feat2), dim=0)                                         # [2, B * N, C]

        final_feat = self.self_trans(cross_feat)                         # [2, B * N, C]
        final_feat = final_feat.reshape(2, B, N, C).permute(1, 0, 2, 3)  # [2, B * N, C] -> [2, B, N, C] -> [B, 2, N, C]
        final_feat = self.conv1(final_feat)                              # [B, 1, N, C]
        final_feat = final_feat.permute(0, 2, 1, 3)                      # [B, N, 1, C]

        out = self.linear2(final_feat)  # [B, N, 1, C]   C:32 -> 8
        out = out.squeeze(2)            # [B, N, C]

        return out
