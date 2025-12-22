"""Tests for Lightning TEMDataModule batch shapes.

These tests ensure that the DataModule yields full-walk, time-major tensors
with walk length coming solely from DataModuleConfig.sequence_length.
"""

from torch_tem.config import DataModuleConfig, EnvironmentConfig
from torch_tem.data.datamodule import TEMDataModule


def test_tem_datamodule_yields_time_major_full_walks() -> None:
    """DataModule yields (observations, actions, locations) with [T,B,...] shapes."""
    cfg = DataModuleConfig(
        environment=EnvironmentConfig(width=4, height=4, observation_mode="unique"),
        batch_size=3,
        sequence_length=7,
        n_train_batches=2,
        n_val_batches=0,
        n_test_batches=0,
        num_workers=0,
    )

    dm = TEMDataModule(cfg)
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    observations, actions, locations = batch

    assert observations.shape[0] == 7
    assert observations.shape[1] == 3
    assert actions.shape == (7, 3)
    assert locations.shape == (7, 3)
