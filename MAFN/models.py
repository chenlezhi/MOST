import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sympy.tensor import tensor
from torch.nn import Linear
from torch.nn.parameter import Parameter

from layers import GraphConvolution
import torch
import sympy


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class decoder(torch.nn.Module):
    def __init__(self, nfeat,  nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

class AttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        """
        Apply attention mechanism between two embeddings (emb1 and emb2) from each modality.
        """
        # Concatenate the two embeddings along the sequence dimension
        emb = torch.cat((torch.unsqueeze(emb1, dim=1), torch.unsqueeze(emb2, dim=1)), dim=1)

        # Apply linear transformation and tanh activation
        v = F.tanh(torch.matmul(emb, self.w_omega))  # Shape: [batch, 2, out_feat]

        # Compute attention scores
        vu = torch.matmul(v, self.u_omega)  # Shape: [batch, 2, 1]

        # Get attention weights using softmax
        alpha = F.softmax(vu.squeeze(), dim=-1)  # Shape: [batch, 2]

        # Apply attention weights to combine embeddings
        emb_combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))
        # emb_combined = F.relu(emb_combined)  # Optional activation

        return emb_combined.squeeze()  # , alpha

class MAFN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(MAFN, self).__init__()
        
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.dropout = dropout
        self.alpha = 0.9
        self.meta = nn.Parameter(torch.Tensor([0.1]))
        self.meta.data.clamp_(0, 1)
        self.attention = AttentionLayer(nhid2, nhid2)

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN

        emb = emb1 * self.alpha + emb2 * (1 - self.alpha)

        [pi, disp, mean] = self.ZINB(emb)
        return emb, pi, disp, mean, emb1, emb2

