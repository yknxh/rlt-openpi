"""Data loader for RLT training, delegating to OpenPI's pipeline.

Reuses OpenPI's full transform chain so that normalization, action
chunking, and camera layout exactly match the pretrained VLA config.

Custom hardware setups can override the default data transforms by
passing a ``data_transforms`` :class:`~openpi.transforms.Group`.  See
``rlt_openpi/policies/franka/`` for a concrete example.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import typing

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from openpi.models.model import Observation
from openpi.training.config import get_config
from openpi.training.data_loader import (
    create_torch_dataset,
    transform_dataset,
)
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def _collate_fn(items):
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _patch_repack_action_key(data_config, action_key: str):
    """Rewrite the repack transform so `"actions"` reads from *action_key*."""
    new_inputs = []
    for t in data_config.repack_transforms.inputs:
        if isinstance(t, _transforms.RepackTransform) and "actions" in t.structure:
            patched = dict(t.structure)
            patched["actions"] = action_key
            t = _transforms.RepackTransform(patched)
        new_inputs.append(t)
    repack = _transforms.Group(inputs=new_inputs)
    return dataclasses.replace(data_config, repack_transforms=repack)


def _drop_missing_repack_keys(data_config, available_keys: set[str]):
    """Remove repack entries whose source key is absent from the dataset."""
    new_inputs = []
    for t in data_config.repack_transforms.inputs:
        if isinstance(t, _transforms.RepackTransform):
            pruned = {k: v for k, v in t.structure.items() if v in available_keys}
            dropped = set(t.structure) - set(pruned)
            if dropped:
                logger.info("Dropping repack keys missing from dataset: %s", sorted(dropped))
            t = _transforms.RepackTransform(pruned)
        new_inputs.append(t)
    repack = _transforms.Group(inputs=new_inputs)
    return dataclasses.replace(data_config, repack_transforms=repack)


def build_data_loader(
    openpi_config_name: str,
    repo_id: str,
    batch_size: int,
    *,
    num_workers: int = 2,
    shuffle: bool = True,
    data_transforms: _transforms.Group | None = None,
):
    """Build a PyTorch data loader using OpenPI's full pipeline.

    Works for any robot/model registered in OpenPI (DROID, ALOHA, Libero,
    etc.).  Returns an infinite iterator yielding
    ``(Observation, actions)`` tuples with correctly normalised,
    tokenised, and padded tensors.

    Args:
        openpi_config_name: Registered OpenPI config name
            (e.g. ``"pi05_droid_finetune"``).
        repo_id: LeRobot dataset repo ID (e.g. ``"local/stack_the_blocks"``).
        batch_size: Global batch size.
        num_workers: DataLoader workers.
        shuffle: Whether to shuffle.
        data_transforms: Optional override for the config's default
            ``data_transforms``.  If ``None``, the transforms from the
            OpenPI config are used as-is.  Provide a custom
            :class:`~openpi.transforms.Group` to change observation /
            action processing (e.g. different camera layouts).

    Yields:
        ``(observation, actions)`` –
        ``observation`` is an :class:`Observation` with torch tensors;
        ``actions`` is ``[B, action_horizon, action_dim]`` float32.
    """
    config = get_config(openpi_config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_config = dataclasses.replace(data_config, repo_id=repo_id)

    # Auto-detect the action column name in the LeRobot dataset.
    # Standard LeRobot datasets use "action" (singular), while OpenPI's
    # DROID conversion produces "actions" (plural).  Patch both
    # action_sequence_keys and the repack transform so users don't have
    # to rename anything.
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    if "action" in meta.features and "actions" not in meta.features:
        data_config = dataclasses.replace(
            data_config, action_sequence_keys=("action",)
        )
        data_config = _patch_repack_action_key(data_config, "action")

    available_keys = set(meta.features) | {"prompt"}
    data_config = _drop_missing_repack_keys(data_config, available_keys)

    if data_transforms is not None:
        logger.info("Overriding data_transforms with custom Group")
        data_config = dataclasses.replace(data_config, data_transforms=data_transforms)

    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = transform_dataset(dataset, data_config)

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_fn,
        drop_last=True,
    )

    return _InfiniteLoader(torch_loader)


class _InfiniteLoader:
    """Wraps a torch DataLoader as an infinite iterator of ``(Observation, actions)``."""

    def __init__(self, loader):
        self._loader = loader

    @staticmethod
    def _to_float32(x):
        t = torch.as_tensor(x)
        if t.is_floating_point():
            t = t.float()
        return t

    def __iter__(self):
        while True:
            for batch in self._loader:
                batch = jax.tree.map(self._to_float32, batch)
                yield Observation.from_dict(batch), batch["actions"]
