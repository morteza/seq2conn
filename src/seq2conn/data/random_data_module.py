import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch


class RandomDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/',
                 batch_size: int = 32,
                 n_timesteps: int = 10,
                 input_dim: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_data = torch.randn(128, n_timesteps, input_dim)
        self.test_data = torch.randn(128, n_timesteps, input_dim)
        self.val_data = torch.randn(128, n_timesteps, input_dim)

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)

    def teardown(self, stage: str):
        pass
