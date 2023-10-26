import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Conv1D(nn.Module):
    def __init__(self, cnn_method: str, in_channels: int, cnn_kernel_num: int, cnn_window_size: int):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=cnn_window_size, padding=(cnn_window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=5, padding=2)
        else:
            assert cnn_kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=2, padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=4, padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 5, kernel_size=5, padding=2)
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return F.relu(self.conv(feature)) # [batch_size, cnn_kernel_num, length]
        elif self.cnn_method == 'group3':
            return F.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return F.relu(torch.cat([self.conv1(feature), \
                                     self.conv2(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv3(feature), \
                                     self.conv4(torch.cat([feature, padding_zeros], dim=1)), \
                                     self.conv5(feature)], dim=1))


class Conv2D_Pool(nn.Module):
    def __init__(self, cnn_method: str, in_channels: int, cnn_kernel_num: int, cnn_window_size: int, last_channel_num: int):
        super(Conv2D_Pool, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group4']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        self.last_channel_num = last_channel_num
        self.cnn_window_size = cnn_window_size
        if self.cnn_method == 'naive':
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num, kernel_size=[cnn_window_size, last_channel_num], padding=[(cnn_window_size - 1) // 2, 0])
        elif self.cnn_method == 'group3':
            assert cnn_kernel_num % 3 == 0
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[1, last_channel_num], padding=[0, 0])
            self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[2, last_channel_num], padding=[0, 0])
            self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 3, kernel_size=[3, last_channel_num], padding=[1, 0])
        else:
            assert cnn_kernel_num % 4 == 0
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[1, last_channel_num], padding=[0, 0])
            self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[2, last_channel_num], padding=[0, 0])
            self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[3, last_channel_num], padding=[1, 0])
            self.conv4 = nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_kernel_num // 4, kernel_size=[4, last_channel_num], padding=[1, 0])
        self.device = torch.device('cuda')

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, cnn_kernel_num]
    def forward(self, feature):
        length = feature.size(2)
        if self.cnn_method == 'naive':
            conv_relu = F.relu(self.conv(feature), inplace=True)                                                     # [batch_size, cnn_kernel_num, length]
            conv_relu_pool, _ = torch.max(conv_relu[:, :, :length - self.cnn_window_size + 1], dim=2, keepdim=False) # [batch_size, cnn_kernel_num]
            return conv_relu_pool
        elif self.cnn_method == 'group3':
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1, self.last_channel_num], device=self.device)
            conv1_relu = F.relu(self.conv1(feature), inplace=True)
            conv1_relu_pool, _ = torch.max(conv1_relu, dim=2, keepdim=False)
            conv2_relu = F.relu(self.conv2(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv2_relu_pool, _ = torch.max(conv2_relu[:, :, :length - 1], dim=2, keepdim=False)
            conv3_relu = F.relu(self.conv3(feature), inplace=True)
            conv3_relu_pool, _ = torch.max(conv3_relu[:, :, :length - 2], dim=2, keepdim=False)
            return torch.cat([conv1_relu_pool, conv2_relu_pool, conv3_relu_pool], dim=1)
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1, self.last_channel_num], device=self.device)
            conv1_relu = F.relu(self.conv1(feature), inplace=True)
            conv1_relu_pool, _ = torch.max(conv1_relu, dim=2, keepdim=False)
            conv2_relu = F.relu(self.conv2(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv2_relu_pool, _ = torch.max(conv2_relu[:, :, :length - 1], dim=2, keepdim=False)
            conv3_relu = F.relu(self.conv3(feature), inplace=True)
            conv3_relu_pool, _ = torch.max(conv3_relu[:, :, :length - 2], dim=2, keepdim=False)
            conv4_relu = F.relu(self.conv4(torch.cat([feature, padding_zeros], dim=2)), inplace=True)
            conv4_relu_pool, _ = torch.max(conv4_relu[:, :, :length - 3], dim=2, keepdim=False)
            return torch.cat([conv1_relu_pool, conv2_relu_pool, conv3_relu_pool, conv4_relu_pool], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_Q = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_K = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_V = nn.Linear(d_model, self.h*self.d_v, bias=True)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_k]
    # Output
    # out  : [batch_size, len_q, h * d_v]
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])                                           # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])                                           # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])                                           # [batch_size, len_k, h, d_v]
        Q = Q.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_q, self.d_k])                   # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_k])                   # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_v])                   # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar                                  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.h]).view([batch_size * self.h, 1, self.len_k]).repeat([1, self.len_q, 1]) # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(_mask == 0, -1e9), dim=2)                                              # [batch_size * h, len_q, len_k]
        else:
            alpha = F.softmax(A, dim=2)                                                                            # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])                                 # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view([batch_size, self.len_q, self.out_dim])                  # [batch_size, len_q, h * d_v]
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    # Input
    # feature : [batch_size, length, feature_dim]
    # mask    : [batch_size, length]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out




class ScaledDotProduct_CandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(ScaledDotProduct_CandidateAttention, self).__init__()
        self.K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.Q = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_scalar = math.sqrt(float(attention_dim))

    def initialize(self):
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.zeros_(self.Q.bias)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = torch.bmm(self.K(feature), self.Q(query).unsqueeze(dim=2)).squeeze(dim=2) / self.attention_scalar # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                          # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                       # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                       # [batch_size, feature_dim]
        return out


class CandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(CandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(feature_dim, attention_dim, bias=False)
        self.query_affine = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_affine = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, feature_dim]
    def forward(self, feature, query, mask=None):
        a = self.attention_affine(torch.tanh(self.feature_affine(feature) + self.query_affine(query).unsqueeze(dim=1))).squeeze(dim=2) # [batch_size, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1)                                                                   # [batch_size, feature_num]
        else:
            alpha = F.softmax(a, dim=1)                                                                                                # [batch_size, feature_num]
        out = torch.bmm(alpha.unsqueeze(dim=1), feature).squeeze(dim=1)                                                                # [batch_size, feature_dim]
        return out


class MultipleCandidateAttention(nn.Module):
    def __init__(self, feature_dim: int, query_dim: int, attention_dim: int):
        super(MultipleCandidateAttention, self).__init__()
        self.feature_affine = nn.Linear(feature_dim, attention_dim, bias=False)
        self.query_affine = nn.Linear(query_dim, attention_dim, bias=True)
        self.attention_affine = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.feature_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.query_affine.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.query_affine.bias)
        nn.init.xavier_uniform_(self.attention_affine.weight)

    # Input
    # feature : [batch_size, feature_num, feature_dim]
    # query   : [batch_size, query_num, query_dim]
    # mask    : [batch_size, feature_num]
    # Output
    # out     : [batch_size, query_num, feature_dim]
    def forward(self, feature, query, mask=None):
        query_num = query.size(1)
        a = self.attention_affine(torch.tanh(self.feature_affine(feature).unsqueeze(dim=1) + self.query_affine(query).unsqueeze(dim=2))).squeeze(dim=3) # [batch_size, query_num, feature_num]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask.unsqueeze(dim=1).expand(-1, query_num, -1) == 0, -1e9), dim=2)                                         # [batch_size, query_num, feature_num]
        else:
            alpha = F.softmax(a, dim=2)                                                                                                                 # [batch_size, query_num, feature_num]
        out = torch.bmm(alpha, feature)                                                                                                                 # [batch_size, query_num, feature_dim]
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False, layer_norm=False):
        super(GCNLayer, self).__init__()
        self.residual = residual
        self.layer_norm = layer_norm
        if self.residual and in_dim != out_dim:
            raise Exception('To facilitate residual connection, in_dim must equal to out_dim')
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        if self.layer_norm:
            self.layer_normalization = nn.LayerNorm(normalized_shape=[out_dim])

    def initialize(self):
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W.bias)

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = self.W(torch.bmm(graph, feature)) # [batch_size, node_num, feature_num]
        if self.layer_norm:
            out = self.layer_normalization(out) # [batch_size, node_num, feature_num]
        out = F.relu(out)                       # [batch_size, node_num, feature_num]
        if self.residual:
            out = out + feature                 # [batch_size, node_num, feature_num]
        return out

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, num_layers=1, dropout=0.1, residual=False, layer_norm=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = []
        if self.num_layers == 1:
            self.gcn_layers.append(GCNLayer(in_dim, out_dim, residual=residual, layer_norm=layer_norm))
        else:
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.gcn_layers.append(GCNLayer(in_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            for i in range(1, self.num_layers - 1):
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, residual=residual, layer_norm=layer_norm))
            self.gcn_layers.append(GCNLayer(hidden_dim, out_dim, residual=residual, layer_norm=layer_norm))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

    def initialize(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.initialize()

    # Input
    # feature : [batch_size, node_num, feature_dim]
    # graph   : [batch_size, node_num, node_num]
    # Output
    # out     : [batch_size, node_num, feature_dim]
    def forward(self, feature, graph):
        out = feature
        for i in range(self.num_layers - 1):
            out = self.dropout(self.gcn_layers[i](out, graph))
        out = self.gcn_layers[self.num_layers - 1](out, graph)
        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V,attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, user_ID,Q, K, V=None):
        if V is None:
            V = K
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)


        #context= ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s,attn_mask)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_k)
        #scores = torch.exp(scores)
        #attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        t=0.01
        attn = F.softmax(scores/t, dim=-1)
        #print(attn)
        sparse_map = torch.full_like(scores, 0.019)  # 0.09
        mask_sparse = torch.lt(attn, sparse_map)
        scores = scores.masked_fill(mask_sparse, -1e9)

        attn= F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v_s)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context


class Sparse_En_MHAtt(nn.Module):#对候选新闻处理
    def __init__(self, d_model, num_attention_heads):
        super(Sparse_En_MHAtt, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = 0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
    def forward(self, q, k, v):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_v
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_k
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_k
        ).transpose(1, 2)

        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.d_model
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        att_map = F.softmax(scores, dim=-1)

# Sparse
        sparse_map = torch.full_like(scores, 0.09)

        mask_sparse = torch.lt(att_map, sparse_map)
        scores = scores.masked_fill(mask_sparse, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class Sparse_De_MHAtt(nn.Module):#对历史新闻处理
    def __init__(self, d_model, num_attention_heads):
        super(Sparse_De_MHAtt, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, q, k, v):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_v
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_k
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_k
        ).transpose(1, 2)

        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.d_model
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        t=0.01
        att_map = F.softmax(scores/t, dim=-1)
        #print(att_map)
        # Sparse
        sparse_map = torch.full_like(scores, 0.019)#0.09

        mask_sparse = torch.lt(att_map, sparse_map)
        scores = scores.masked_fill(mask_sparse, -1e9)
        att_map = F.softmax(scores, dim=-1)
        #att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))
class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=400,
            mid_size=200,
            out_size=400,
            dropout_r=0.2,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ScaledDotProductAttention1(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention1, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)

        Md = torch.empty(20, 20)  # 从1到5(左右都包含),等距取3个数
        for i in range(20):
            for j in range(20):
                if i < j:
                    Md[i][j] = i - j
                else:
                    Md[i][j] = -(i - j)
                j = j + 1
            i = i + 1
        Md=Md.unsqueeze(dim=0).unsqueeze(dim=1).repeat(V.shape[0], V.shape[1],1, 1)
        deAtt=context+torch.matmul(Md.to(device), V.to(device))#[120, 200, 20, 2]
        return deAtt, attn


class decay_Attention(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(decay_Attention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(
                batch_size, maxlen) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention1(self.d_k)(q_s, k_s, v_s,attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context



class Sparse_MHAtt(nn.Module):#对历史新闻处理
    def __init__(self, d_model, num_attention_heads):
        super(Sparse_MHAtt, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, q, k, v):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_v
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_k
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_attention_heads,
            self.d_k
        ).transpose(1, 2)

        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.d_model
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)



class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

    def split_heads(self, x, batch_size):
        # 将输入张量分割成多个头
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()

        # 使用线性层进行投影
        Q_proj = self.W_q(Q)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        # 将每个头分开
        Q_proj = self.split_heads(Q_proj, batch_size)
        K_proj = self.split_heads(K_proj, batch_size)
        V_proj = self.split_heads(V_proj, batch_size)

        # 计算注意力分数
        scores = torch.matmul(Q_proj, K_proj.permute(0, 1, 3, 2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # 计算注意力权重
        t=0.001
        att_map = torch.nn.functional.softmax(scores/t, dim=-1)

        sparse_map = torch.full_like(scores, 0.019)  # 0.09

        mask_sparse = torch.lt(att_map, sparse_map)
        scores = scores.masked_fill(mask_sparse, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重加权求和得到注意力输出
        attn_output = torch.matmul(attn_weights, V_proj)

        # 合并多个头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        # 通过线性层得到最终输出
        output = self.W_o(attn_output)

        return output

