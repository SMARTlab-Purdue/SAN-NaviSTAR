import torch
import torch.nn as nn
import copy
import math
import numpy as np


def _reshape(seq_len, env_num, x):
    if len(x.shape) == 3:
        return x.reshape(seq_len, env_num, *x.shape[1:]).permute(1, 0, 2, 3)
    elif len(x.shape) == 2:
        return x.reshape(seq_len, env_num, *x.shape[1:]).permute(1, 0, 2)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

def get_masks(masks, device='cpu'):
    # [B, T, 1] -> [B, T, T]
    batch_size, time_len = masks.shape[0], masks.shape[1]
    zeros_masks = torch.zeros((batch_size, time_len, time_len)).to(device)
    attn_mask = torch.tril(torch.ones((batch_size, time_len, time_len)).to(device))
    masks = masks.squeeze(-1).permute(1, 0) # [T, B * N]
    zero_index = (masks[1:] == 0).any(-1).nonzero().squeeze().cpu()
    if zero_index.dim() == 0:
        # Deal with scalar
        zero_index = [zero_index.item() + 1]
    else:
        zero_index = (zero_index + 1).numpy().tolist()

    # add t=0 and t=T to the list
    zero_index = [0] + zero_index + [time_len]
    for i in range(len(zero_index) - 1):
        zeros_masks[:, zero_index[i]:zero_index[i + 1], zero_index[i]:zero_index[i + 1]] = 1

    # limit the temporal attention length
    attention_num_limit = 5
    if time_len > attention_num_limit:
        limit_attention_num_mask = torch.triu(
            torch.ones((batch_size, time_len - attention_num_limit, time_len - attention_num_limit)).to(device))
        attn_mask[:, attention_num_limit:, :time_len - attention_num_limit] *= limit_attention_num_mask

    attn_mask = (attn_mask * zeros_masks).to(device)
    return attn_mask

def get_adj_matrix(x, device='cpu'):
    # [B, N, 2]
    B, N, C = x.shape
    x1 = torch.repeat_interleave(x, N, dim=1) # [B, N * N, C]
    x2 = x.repeat(1, N, 1) # [B, N * N, C]
    dis = torch.norm(x1 - x2, dim=-1) # [B, N * N]
    dis = dis.reshape(B, N, N) # [B, N, N]
    dis = dis.detach()

    return dis


class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, x, graph):
        '''
        :param x: [B * T, N, C]
        :param graph: [B * T, N, N]
        :return: [B * T, N, C]
        '''
        B, N = x.shape[0], x.shape[1]
        graph = GCN.process_graph(graph)

        output_1 = self.linear_1(x)  # [B, N, C]
        output_1 = self.act(torch.matmul(graph, output_1))  # [B, N, N], [B, N, C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph, output_2))  # [B, N, C]

        return output_2

    @staticmethod
    def process_graph(graph):
        '''
        :param graph: [B * T, N, N]
        :return: [B * T, N, N]
        '''
        B, N = graph.shape[0], graph.shape[1]
        for i in range(B):
            g = graph[i]
            matrix_i = torch.eye(N, dtype=g.dtype, device=g.device)
            g += matrix_i  # A~ [N, N]

            degree_matrix = torch.sum(g, dim=-1, keepdim=False)  # [N]
            degree_matrix = degree_matrix.pow(-1)
            degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

            degree_matrix = torch.diag(degree_matrix)  # [N, N]

            g = torch.mm(degree_matrix, g)  # D^(-1) * A = \hat(A)

            graph[i] = g

        return graph

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # [B, H, T, C]
        batch_size, head, length, d_tensor = k.size()
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, head, 1, 1)

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e7)

        score = self.softmax(score)

        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, n_head, qkv_same_dim=True, q_dim=None, k_dim=None, v_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        if qkv_same_dim:
            self.w_q = nn.Linear(embed_size, embed_size)
            self.w_k = nn.Linear(embed_size, embed_size)
            self.w_v = nn.Linear(embed_size, embed_size)
            self.w_concat = nn.Linear(embed_size, embed_size)
        else:
            self.w_q = nn.Linear(embed_size, q_dim)
            self.w_k = nn.Linear(embed_size, k_dim)
            self.w_v = nn.Linear(embed_size, v_dim)
            self.w_concat = nn.Linear(v_dim, v_dim)

    def split(self, tensor):
        batch_size, length, embed_size = tensor.shape

        d_tensor = embed_size // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) # [B, H, T, C]

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.shape
        embed_size = head * d_tensor

        tensor = tensor.transpose(1, 2).reshape(batch_size, length, embed_size)
        return tensor

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        if self.n_head > 1:
            out = self.w_concat(out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, hidden_size, n_head, device):
        super(SpatialAttention, self).__init__()
        self.device = device

        self.attention = MultiHeadAttention(hidden_size, n_head)

    def forward(self, q, k, v, masks=None):
        x = self.attention(q, k, v, masks)

        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, forward_size, drop, n_head, device):
        super(Attention, self).__init__()
        self.device = device

        self.attention = MultiHeadAttention(hidden_size, n_head)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(drop)

        self.ffn = PositionwiseFeedForward(hidden_size, forward_size, drop)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, q, k, v, masks=None):
        res = q

        x = self.attention(q, k, v, masks)

        x = self.dropout1(x)
        x = self.norm1(x + res)

        res = x

        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + res)

        return x

class GCNTransformer(nn.Module):
    def __init__(self, hidden_size, forward_size, drop, n_head, device):
        super(GCNTransformer, self).__init__()
        self.device = device
        self.attention = Attention(hidden_size, forward_size, drop, n_head, device)
        self.gcn = GCN(hidden_size, 2 * hidden_size, hidden_size)

        self.fs = nn.Linear(hidden_size, hidden_size)
        self.fg = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(drop)

        self.ffn = PositionwiseFeedForward(hidden_size, forward_size, drop)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, q, k, v, graph, masks=None):
        # x [B, T, N, C]
        res = q

        x = self.attention(q, k, v, masks)

        x = self.dropout1(x)
        x = self.norm1(x + res)

        res = x

        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + res)

        gcn_feat = self.gcn(q, graph)

        g = torch.sigmoid(self.fs(x) + self.fg(gcn_feat))

        out = g * x + (1 - g) * gcn_feat

        return out

class Transformer(nn.Module):
    def __init__(self, hidden_size, forward_size, n_layers, drop, n_head, device):
        super(Transformer, self).__init__()
        self.device = device
        self.blocks = nn.ModuleList([Attention(hidden_size, forward_size, drop, n_head, device) for _ in range(n_layers)])
    def forward(self, q, k, v, masks=None):
        for block in self.blocks:
            q = block(q, k, v, masks)

        return q


class STAR(nn.Module):
    def __init__(self, obs_space, config, output_size=None, device='cpu'):
        super(STAR, self).__init__()
        self.config = config
        self.device = device
        self.human_num = self.config.sim.human_num
        self.hidden_size = self.config.trans.hidden_size
        self.forward_size = self.config.trans.forward_size
        self.n_layers = self.config.trans.n_layers
        self.n_head = self.config.trans.n_head
        self.dropout = self.config.trans.dropout
        self.output_size = self.hidden_size if output_size is None else output_size
        self.is_recurrent = True

        self.Temporal_Transformer = Transformer(self.hidden_size,
                                             self.forward_size,
                                             self.n_layers,
                                             self.dropout,
                                             self.n_head,
                                             self.device)

        self.gcn_trans = GCNTransformer(self.hidden_size,
                                        self.forward_size,
                                        self.dropout,
                                        self.n_head,
                                        self.device)

        self.Spatial_Transformer = Transformer(self.hidden_size,
                                                   self.forward_size,
                                                   self.n_layers,
                                                   self.dropout,
                                                   self.n_head,
                                                   self.device)

        self.Cross_Transformer = SpatialAttention(self.hidden_size,
                                                  1,
                                                  self.device)

        self.Fusion_Transformer = Transformer(2 * self.hidden_size,
                                                   self.forward_size,
                                                   self.n_layers,
                                                   self.dropout,
                                                   self.n_head,
                                                   self.device)

        if self.config.trans.activation == 'Tanh':
            self.activation = nn.Tanh()
        elif self.config.trans.activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif self.config.trans.activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        self.robot_node_embed = nn.Sequential(nn.Linear(7, 3),
                                              self.activation,
                                              nn.Linear(3, self.hidden_size))
        self.spatial_embed = nn.Linear(4, self.hidden_size)
        self.temporal_embed = nn.Linear(2, self.hidden_size)
        self.gcn_layer = nn.Linear(4, self.hidden_size)
        self.cross_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.final_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.actor = nn.Sequential(nn.Linear(self.hidden_size, 8 * self.hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(8 * self.hidden_size, self.output_size),
                                   nn.Tanh())

        self.critic = nn.Sequential(nn.Linear(self.hidden_size, 8 * self.hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(8 * self.hidden_size, self.hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_size, 1))

        self._init_params()

    def _init_params(self):
        initialize_weights(self.modules())

    def set_device(self, device=None):
        if device is not None:
            self.to(device)
        else:
            self.to(self.device)

    def forward(self, obs, rnn_hxs=None, masks=None, infer=False):
        # masks [T * B, 1]
        # rnn_hxs is useless like many other parameters here, just for compatibility.

        robot_node = obs['robot_node'] # [T * B(env), 1, 7]
        spatial_edges_transformer = obs['spatial_edges_transformer'] # [T * B(env), N(agent_num), 4]
        visible_masks = obs['visible_masks']  # [T * B(env), N(agent_num), 1]
        robot_pos = obs['robot_pos'] # [T * B(env), 1, 4]

        batch, human_num = spatial_edges_transformer.shape[0], spatial_edges_transformer.shape[1] - 1
        if infer:
            seq_len = 1
            env_num = batch // seq_len
        else:
            seq_len = self.config.ppo.num_steps
            env_num = batch // seq_len

        if masks is None:
            masks = torch.ones(seq_len * env_num, 1).to(robot_node.device)

        robot_node = _reshape(seq_len, env_num, robot_node)  # [B, T, 1, 5]
        spatial = _reshape(seq_len, env_num, spatial_edges_transformer)  # [B, T, N, 4]
        visible_masks = _reshape(seq_len, env_num, visible_masks)  # [B, T, N]
        robot_pos = _reshape(seq_len, env_num, robot_pos)  # [B, T, 1, 4]
        masks = _reshape(seq_len, env_num, masks)  # [B, T, 1]
        temporal_masks = get_masks(masks, device=masks.device) # [B, T, T]


        robot_node_embedding = self.activation(self.robot_node_embed(robot_node))  # [B, T, 1, C]
        robot_node_embedding = robot_node_embedding.reshape(-1, 1, self.hidden_size)  # [B * T, 1, C]

        all_agent = torch.cat((robot_pos, spatial), dim=2).reshape(env_num * seq_len, self.human_num + 2, -1) # [B * T, N, 4]
        all_pos = all_agent[..., :2] # [B * T, N, 2]
        graph = get_adj_matrix(all_pos, device=all_pos.device) # [B * T, N, N]
        all_agent = self.activation(self.gcn_layer(all_agent)) # [B * T, N, C]
        gcn_feat = self.gcn_trans(all_agent, all_agent, all_agent, graph) # [B * T, N, C]
        gcn_feat = gcn_feat[:, [0], :] # [B * T, 1, C]

        spatial = spatial.permute(0, 2, 1, 3).reshape(env_num * (human_num + 1), seq_len, -1) # [B * N, T, 4]
        temporal_masks_for_spatial = temporal_masks.reshape(env_num, 1, seq_len, -1) # [B, 1, T, T]


        spatial_embedding = self.activation(self.spatial_embed(spatial))  # [B, T, C]
        temporal_masks_for_spatial = torch.repeat_interleave(temporal_masks_for_spatial,
                                                             human_num + 1,
                                                             dim=1).reshape(env_num * (human_num + 1), seq_len, -1)  # [B * N, T, T]


        spatial_feat = self.Spatial_Transformer(spatial_embedding,
                                                spatial_embedding,
                                                spatial_embedding,
                                                masks=temporal_masks_for_spatial) # [B * N, T, C]

        cross_masks = visible_masks.reshape(env_num * seq_len, 1, -1) # [B * T, 1, N]
        spatial_feat = spatial_feat.reshape(env_num, human_num + 1, seq_len, -1).permute(0, 2, 1, 3) # [B, T, N, C]
        spatial_feat = spatial_feat.reshape(env_num * seq_len, human_num + 1, -1)  # [B * T, N, C]

        cross_feat = self.Cross_Transformer(gcn_feat,
                                            spatial_feat,
                                            spatial_feat,
                                            masks=cross_masks) # [B * T, 1, C]

        cross_feat = torch.cat((gcn_feat, cross_feat), dim=-1) # [B * T, 1, 2 * C]
        cross_feat = self.activation(self.cross_layer(cross_feat)) # [B * T, 1, C]

        fusion = torch.cat((robot_node_embedding, cross_feat), dim=-1) # [B * T, 1, 2 * C]
        fusion = fusion.reshape(env_num, seq_len, -1) # [B, T, 2 * C]

        output_feat = self.Fusion_Transformer(fusion,
                                             fusion,
                                             fusion,
                                             masks=temporal_masks) # [B, T, 2 * C]

        output_feat = self.final_layer(output_feat)  # [B * T, C]
        output_feat = output_feat.reshape(env_num, seq_len, -1).permute(1, 0, 2) # [T, B, C]

        actor_feat = self.actor(output_feat).reshape(-1, self.output_size) # [T * B, C]
        critic_feat = self.critic(output_feat).reshape(-1, 1) # [T * B, 1]

        return critic_feat, actor_feat, rnn_hxs







