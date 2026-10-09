"""M4Raw training data: fully sampled k-space plus its cached ESPIRiT maps.

Each item is the raw k-space and the coil maps for one slice, not an image.
Undersampling, normalization and the reconstruction target are all derived in
``FlexMRI.training_losses``, so training exercises exactly the operators
sampling uses, and a change to the forward model cannot drift out of sync
between them.
"""

import numpy as np
from torch.utils.data import DataLoader, Dataset

import read_data


def load_data(*, data_dir, batch_size, image_size, num_workers=4, **data_kwargs):
    """
    Create an infinite generator over (kspace, kwargs) pairs.

    Each kspace is an ``[B, 2C, H, W]`` float tensor holding the real parts of
    the coils followed by the imaginary parts; the kwargs dict carries the
    coil maps the same way under ``"coil_map"``.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: matrix size the model is built for; volumes that do not
        match it are skipped.
    :param num_workers: DataLoader workers; slices are read from disk on
        access, so a value above 0 is recommended.
    :param data_kwargs: forwarded to ``read_data.TrainSlices`` (phase,
        contrast, max_volumes, slices_per_volume, coil_map_cache).
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    slices = read_data.TrainSlices(data_dir, image_size=image_size, **data_kwargs)

    loader = DataLoader(
        KspaceDataset(slices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    while True:
        yield from loader


def _to_real(arr):
    """Complex ``[C, H, W]`` to float32 ``[2C, H, W]``: real coils, then imaginary."""
    arr = np.asarray(arr)
    return np.concatenate([arr.real, arr.imag], axis=0).astype(np.float32)


class KspaceDataset(Dataset):
    """One item per slice, read lazily: a training split does not fit in RAM."""

    def __init__(self, slices):
        super().__init__()
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        kspace, coil_map = self.slices[i]
        return _to_real(kspace), {"coil_map": _to_real(coil_map)}
