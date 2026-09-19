"""
model.py — GNN encoder, link predictor, and the Temporal ARSS model.

No blocking bugs live here — B1/B2/B4 were upstream (data/sampler), and once
those are fixed this code is architecturally sound.

One thing to decide, not silently fix (report issue R4): the report's
Eq. 3.6 says h_i should be MEAN POOLING over the whole subgraph; the code
takes the ROOT NODE'S OWN embedding after message passing. Both are
defensible designs, but they are different models, and the report currently
describes the one that wasn't run. `readout` below lets you run both and
report whichever you choose to write up — or both, as an ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Batch

NUM_ROLES = 5


class SubgraphGNN(nn.Module):
    def __init__(self, in_ch=NUM_ROLES, hidden=128, out=64, layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        dims = [in_ch] + [hidden] * (layers - 1) + [out]
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(layers)])

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim=64, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_i, h_j, h_common):
        hadamard = h_i * h_j                         # Eq. 8
        z_ij = torch.cat([hadamard, h_common], dim=-1)
        return self.mlp(z_ij).squeeze(-1)             # Eq. 9


class TemporalARSSModel(nn.Module):
    def __init__(self, hidden=128, emb_dim=64, dropout=0.3, readout="root"):
        """readout: 'root' (code-original) or 'mean' (report Eq. 3.6). See
        module docstring re: R4."""
        super().__init__()
        assert readout in ("root", "mean")
        self.readout = readout
        self.emb_dim = emb_dim
        self.gnn = SubgraphGNN(NUM_ROLES, hidden, emb_dim, 2, dropout)
        self.predictor = LinkPredictor(emb_dim, hidden)

    def forward(self, subgraphs):
        B = len(subgraphs)
        device = next(self.parameters()).device

        batch = Batch.from_data_list(subgraphs).to(device)
        all_embs = self.gnn(batch.x, batch.edge_index)

        starts = batch.ptr[:-1]

        if self.readout == "root":
            h_i = all_embs[starts]
            h_j = all_embs[starts + 1]
        else:  # "mean": Eq. 3.6, mean pooling over the whole subgraph
            h_i = torch.zeros(B, self.emb_dim, device=device)
            h_j = h_i.clone()
            for idx in range(B):
                s, e = batch.ptr[idx].item(), batch.ptr[idx + 1].item()
                pooled = all_embs[s:e].mean(dim=0)
                h_i[idx] = pooled
                h_j[idx] = pooled  # both roots draw on the same subgraph pool

        h_common = torch.zeros(B, self.emb_dim, device=device)
        for idx, sg in enumerate(subgraphs):
            cm = sg.common_mask
            if cm.any():
                start = batch.ptr[idx].item()
                sg_embs = all_embs[start: start + sg.num_nodes]
                h_common[idx] = sg_embs[cm.to(device)].sum(dim=0)

        return self.predictor(h_i, h_j, h_common)
