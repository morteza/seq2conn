import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Normal, kl_divergence
from .graph_decoder import GraphDecoder
from .graph_encoder import GraphEncoder
from torch_geometric.nn.models import VGAE


class TS2ConnModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, edge_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim)
        self.n_nodes_fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        self.edge_prob_fc = nn.Linear(hidden_dim, edge_dim)
        self.graph_vae = VGAE(hidden_dim, hidden_dim)
        self.graph_encoder = GraphEncoder(hidden_dim, edge_dim)
        self.graph_decoder = GraphDecoder(hidden_dim, edge_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.edge_dim = edge_dim

    def to_dense_adj(self, edges):
        n_nodes = edges.shape[1]
        adj = torch.zeros((edges.shape[0], n_nodes, n_nodes), dtype=torch.float32)
        for i in range(n_nodes):
            for j in range(n_nodes):
                adj[:, i, j] = edges[:, i * n_nodes + j]
        return adj

    def add_node_actor(self, G_embedding):
        add_node_prob = self.add_node_fc(G_embedding)

        add_node = Bernoulli(probs=add_node_prob).sample()

        new_node_embedding = None  # defaults to None (if add_node == False)

        if add_node:
            new_node_embedding = ...

        return add_node, new_node_embedding

    def add_edge_actor(self, G_embedding, src_node, tgt_node):

        src_embedding = G_embedding[:, src_node, :]
        tgt_embedding = G_embedding[:, tgt_node, :]

        similarities = torch.cosine_similarity(src_embedding, tgt_embedding, dim=1)

        add_edge_dist = Bernoulli(probs=similarities)

        return add_edge_dist.sample()

    def forward(self, x):
        # Encode the time series
        h = self.encoder(x)

        # Graph Generator (Encoder)

        actions = []
        while True:
            add_node, new_node = self.add_node_actor(h)
            actions.append((add_node, new_node))
            if add_node:
                for tgt_node in self.nodes:
                    add_edge, edge_weight = self.add_edge_actor(new_node, tgt_node)
                    actions.append((add_edge, edge_weight))
                    if add_edge:
                        self.edges.append((new_node, tgt_node, edge_weight))
            else:  # stop the loop if add_node == False
                break

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, z_mean, z_std = self(x)

        # Compute the KL divergence the embedding and the normal distribution
        kl_div = kl_divergence(
            Normal(z_mean, z_std),
            Normal(torch.zeros_like(z_mean), torch.ones_like(z_std)))

        loss_kl = kl_div.sum(dim=-1).mean()
        loss_recon = F.mse_loss(x_recon, x)

        loss = loss_recon + loss_kl

        self.log('train/loss_kl', loss_kl)
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, z_mean, z_std = self(x)

        # Compute the KL divergence the embedding and the normal distribution
        kl_div = kl_divergence(
            Normal(z_mean, z_std),
            Normal(torch.zeros_like(z_mean), torch.ones_like(z_std)))

        loss_kl = kl_div.sum(dim=-1).mean()
        loss_recon = F.mse_loss(x_recon, x)

        loss = loss_recon + loss_kl

        self.log('val/loss_kl', loss_kl)
        self.log('val/loss_recon', loss_recon)
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
