import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, Normal, kl_divergence
from .graph_decoder import GraphDecoder
from .graph_encoder import GraphEncoder
from .graph_vae import GraphVAE


class TS2ConnModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, edge_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim)
        self.n_nodes_fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        self.edge_prob_fc = nn.Linear(hidden_dim, edge_dim)
        self.graph_vae = GraphVAE(hidden_dim, hidden_dim)
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

    def forward(self, x):
        # Encode the time series
        h = self.encoder(x)

        # determine the number of nodes from the input data
        n_nodes = self.n_nodes_fc(h).squeeze().round().long().item()

        # Learn the edge probabilities from the input data
        edge_probs = torch.sigmoid(self.edge_prob_fc(h))
        edge_probs = edge_probs.view(-1, self.edge_dim)

        # Generate a probabilistic graph embedding
        z, z_mean, z_std = self.graph_vae(h)

        # Generate edges from the graph embedding
        edge_logits = self.graph_encoder(z, None)
        edge_probs = edge_probs.repeat(1, n_nodes * n_nodes).view(-1, self.edge_dim)
        edge_probs = torch.sigmoid(edge_probs + edge_logits)
        edge_probs = edge_probs.view(-1, n_nodes * n_nodes, self.edge_dim)

        # Sample edges using the Bernoulli distribution
        edge_dist = Bernoulli(probs=edge_probs)
        edges = edge_dist.sample()

        # Construct the adjacency matrix from the sampled edges
        adj = self.to_dense_adj(edges)

        # Reconstruct the graph embedding from the adjacency matrix
        adj = adj.view(-1, n_nodes, n_nodes)
        adj = adj + adj.transpose(-2, -1)  # Ensure that the adjacency matrix is symmetric
        adj = adj.view(-1, n_nodes * n_nodes)
        z_recon = self.graph_decoder(adj, None)

        # Reconstruct the time series from the graph embedding
        x_recon = self.decoder(z_recon)

        return x_recon, z_mean, z_std

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
