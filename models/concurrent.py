# -*-coding:utf-8 -*-  
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
Tensor = torch.Tensor
device = torch.device("mps" if torch.has_mps else "cpu")

class Concurrent(nn.Module):
    def __init__(self, dim, small_win, dim_in, dim_out):
        super(Concurrent, self).__init__()
        self.num_node = dim
        self.weights_pool_x = nn.Parameter(torch.FloatTensor(small_win, dim_in, dim_out))
        self.bias_pool_x = nn.Parameter(torch.FloatTensor(small_win, dim_out))
        self.weights_pool_pos = nn.Parameter(torch.FloatTensor(small_win, dim_in, dim_out))
        self.bias_pool_pos = nn.Parameter(torch.FloatTensor(small_win, dim_out))

        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt = torch.clamp(d_inv_sqrt, min=1e-10)  # 防止除零错误
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.matmul(d_mat_inv_sqrt).T.matmul(d_mat_inv_sqrt)

    def graph_max_min(self, batch_matrix):
        flat = batch_matrix.reshape(batch_matrix.shape[0], -1)
        tensormax = flat.max(-1, keepdim=True)[0]
        tensormin = flat.min(-1, keepdim=True)[0]
        maxmin_norm = (flat - tensormin) / (tensormax - tensormin)
        matrix_norm = maxmin_norm.reshape(batch_matrix.shape)
        self.cosine_graph_max_min_weight = matrix_norm
        return matrix_norm

    def forward(self, res_x, origin_x):
        # 获取输入数据的设备（确保设备一致性）
        device = res_x.device

        # 计算学习的图
        learned_graph = torch.einsum("bmc,bcn->bmn", res_x, res_x.transpose(1, 2))

        # 计算归一化的图
        norm = torch.norm(res_x, p=2, dim=-1, keepdim=True)
        norm = torch.einsum("bmc,bcn->bmn", norm, norm.transpose(1, 2))
        norm_learned_graph  = learned_graph / norm

        # 创建 mask，并确保它在正确的设备上
        mask = torch.unsqueeze(torch.eye(self.num_node, self.num_node).bool(), dim=0).to(device)
        norm_learned_graph.masked_fill_(mask, 0)

        # 计算最大最小归一化
        self.norm_mask_graph = norm_learned_graph
        masked_max_min_norm = self.graph_max_min(norm_learned_graph)

        # 计算图卷积支持矩阵
        batch_supports_x = []
        for i, adj in enumerate(masked_max_min_norm):
            supports = self.normalize_adj(adj.to(device))  # 确保 supports 在正确设备上
            batch_supports_x.append(supports)

        supports_x = torch.stack(batch_supports_x, dim=0).to(device)  # 确保支持矩阵在正确设备上
        x_g1 = torch.einsum("bnm,bmc->bnc", supports_x, origin_x)
        weights_x = torch.einsum('bnd,dio->bnio', origin_x, self.weights_pool_x)
        bias_x = torch.matmul(origin_x, self.bias_pool_x)
        x_gconv1 = torch.tanh(torch.einsum('bni,bnio->bno', x_g1, weights_x) + bias_x)
        x_gconv = x_gconv1.transpose(1, 2)

        # Optionally apply dropout to the output to reduce overfitting
        x_gconv = self.dropout(x_gconv)

        return x_gconv