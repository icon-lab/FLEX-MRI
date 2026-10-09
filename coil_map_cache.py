"""Per-volume disk cache for ESPIRiT coil sensitivity maps.

The maps depend only on the fully sampled k-space of a volume, so they are the
same for every acceleration rate ``R``. Sampling sweeps such as ``test.sh``
(several R values x several contrasts) would otherwise re-run ESPIRiT once per
combination over the same data. This module stores one ``.npy`` per volume and
reuses it.

Layout::

    <cache_root>/<espirit signature>/multicoil_<phase>/<volume stem>.npy

The signature encodes the ESPIRiT parameters, so changing them writes to a new
directory instead of silently reusing stale maps.

Cache root resolution order: explicit ``cache_root`` argument, then the
``FLEX_MRI_COIL_MAP_CACHE`` environment variable, then
``<dataset folder>/coil_maps_cache``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from espirit import (
    DEFAULT_ACS_CENTER_FRACTION,
    DEFAULT_ESPIRIT_CROP,
    DEFAULT_ESPIRIT_KERNEL_WIDTH,
    DEFAULT_ESPIRIT_THRESH,
    espirit_coil_maps_from_kspace,
)

DEFAULT_CACHE_DIRNAME = "coil_maps_cache"
CACHE_ENV_VAR = "FLEX_MRI_COIL_MAP_CACHE"


def resolve_cache_root(data_folder: str, cache_root: Optional[str] = None) -> str:
    """Cache root for a dataset folder (the one holding the ``multicoil_*`` dirs)."""
    if cache_root is None:
        cache_root = os.environ.get(CACHE_ENV_VAR) or None
    if cache_root is None:
        cache_root = os.path.join(
            os.path.expanduser(str(data_folder)).rstrip("/"), DEFAULT_CACHE_DIRNAME
        )
    return cache_root


def espirit_signature(
    center_fraction: float = DEFAULT_ACS_CENTER_FRACTION,
    kernel_width: int = DEFAULT_ESPIRIT_KERNEL_WIDTH,
    thresh: float = DEFAULT_ESPIRIT_THRESH,
    crop: float = DEFAULT_ESPIRIT_CROP,
) -> str:
    """Directory name encoding the ESPIRiT parameters the maps were made with."""
    return f"espirit_cf{center_fraction:g}_kw{kernel_width:g}_th{thresh:g}_cr{crop:g}"


def cache_path(
    filename: str,
    phase: str,
    cache_root: str,
    center_fraction: float = DEFAULT_ACS_CENTER_FRACTION,
    kernel_width: int = DEFAULT_ESPIRIT_KERNEL_WIDTH,
    thresh: float = DEFAULT_ESPIRIT_THRESH,
    crop: float = DEFAULT_ESPIRIT_CROP,
) -> str:
    """Path of the cached coil maps for one volume file."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    signature = espirit_signature(center_fraction, kernel_width, thresh, crop)
    return os.path.join(cache_root, signature, f"multicoil_{phase}", stem + ".npy")


def load_cached(path: str, expected_shape=None, mmap: bool = False):
    """Return cached maps, or ``None`` if absent / unreadable / wrong shape."""
    if not os.path.isfile(path):
        return None
    try:
        maps = np.load(path, mmap_mode="r" if mmap else None)
    except Exception as exc:  # a truncated or corrupt file is treated as a cache miss
        print(f"Ignoring unreadable coil map cache {path} ({exc})")
        return None
    if expected_shape is not None and tuple(maps.shape) != tuple(expected_shape):
        print(
            f"Ignoring coil map cache {path}: shape {tuple(maps.shape)} "
            f"!= expected {tuple(expected_shape)}"
        )
        return None
    return maps


def save_cached(path: str, maps: np.ndarray) -> None:
    """Write maps atomically (tmp file + rename) so a partial file is never read."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        # Written through a file handle: np.save would append .npy to the temporary name.
        with open(tmp, "wb") as handle:
            np.save(handle, np.asarray(maps, dtype=np.complex64))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def get_volume_coil_maps(
    kspace: np.ndarray,
    filename: str,
    phase: str,
    cache_root: str,
    center_fraction: float = DEFAULT_ACS_CENTER_FRACTION,
    kernel_width: int = DEFAULT_ESPIRIT_KERNEL_WIDTH,
    thresh: float = DEFAULT_ESPIRIT_THRESH,
    crop: float = DEFAULT_ESPIRIT_CROP,
    device=None,
    use_cache: bool = True,
    overwrite: bool = False,
    show_progress: bool = False,
) -> np.ndarray:
    """Coil maps for one volume's fully sampled k-space, cached on disk.

    ``kspace`` is ``(N, C, H, W)``; the returned maps have the same shape.
    """
    kspace = np.asarray(kspace, dtype=np.complex64)
    path = cache_path(
        filename, phase, cache_root, center_fraction, kernel_width, thresh, crop
    )

    if use_cache and not overwrite:
        cached = load_cached(path, expected_shape=kspace.shape)
        if cached is not None:
            return np.asarray(cached, dtype=np.complex64)

    maps = espirit_coil_maps_from_kspace(
        kspace,
        center_fraction=center_fraction,
        kernel_width=kernel_width,
        thresh=thresh,
        crop=crop,
        device=device,
        show_progress=show_progress,
    )

    if use_cache:
        try:
            save_cached(path, maps)
        except OSError as exc:  # e.g. a read-only cache directory or a full disk
            print(f"Could not write coil map cache {path} ({exc})")

    return maps
