"""M4Raw data loading for FLEX-MRI.

Layout: ``<data_dir>/multicoil_<phase>/<volume>.h5`` raw files holding
``kspace`` and ``reconstruction_rss``. Undersampling masks are generated here
and coil maps are estimated with ESPIRiT (cached per volume on disk, see
``coil_map_cache.py`` / ``precompute_coil_maps.py``).

Two entry points::

    TrainSlices(data_dir, ...)   -> per-slice (kspace, coil_map), read lazily
    get_test_data(data_dir, ...) -> (kspace, masks, coil_maps)
"""

from __future__ import annotations

import os

import h5py
import numpy as np
from tqdm import tqdm

from coil_map_cache import cache_path, get_volume_coil_maps, resolve_cache_root
from espirit import DEFAULT_ACS_CENTER_FRACTION

VOLUME_EXT = ".h5"
PHASE_PREFIX = "multicoil_"
DEFAULT_MASK_SEED = 0


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------


def phase_dir(data_dir, phase):
    """Directory holding the volumes of one split."""
    return os.path.join(os.path.expanduser(str(data_dir)), f"{PHASE_PREFIX}{phase}")


def volume_contrast(filename):
    """Contrast label of a volume, e.g. ``2022061001_T101.h5`` -> ``T101``."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_", 1)[1] if "_" in stem else stem


def list_volumes(folder, contrast="all", max_volumes=None):
    """Sorted volume filenames in ``folder``, optionally filtered by contrast.

    ``contrast`` is ``'all'`` or a comma separated list such as ``T1,T2``.
    """
    names = sorted(name for name in os.listdir(folder) if name.endswith(VOLUME_EXT))
    if contrast and str(contrast).strip().lower() != "all":
        wanted = [c.strip() for c in str(contrast).split(",") if c.strip()]
        names = [
            name for name in names
            if any(volume_contrast(name).startswith(c) for c in wanted)
        ]
    if max_volumes is not None:
        names = names[: int(max_volumes)]
    return names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainSlices:
    """Training slices: fully sampled k-space and its ESPIRiT coil maps.

    ``slices[i]`` is ``(kspace, coil_map)``, both complex ``[C, H, W]`` for one
    slice. Volumes are opened on access -- the training split does not fit in
    RAM. Undersampling, normalization and the reconstruction target are all
    derived downstream in ``FlexMRI.training_losses``, so training exercises
    exactly the operators sampling uses; nothing is normalized here.

    Volumes whose coil maps are not cached are skipped and counted; run
    ``precompute_coil_maps.py --phase train`` to add them.
    """

    def __init__(self, data_dir, phase="train", contrast="all", image_size=None,
                 max_volumes=None, slices_per_volume=None, coil_map_cache=None):
        self.folder = phase_dir(data_dir, phase)
        cache_root = resolve_cache_root(data_dir, coil_map_cache)

        volumes = []
        self.map_paths = {}
        missing_maps = 0
        wrong_size = 0

        for name in list_volumes(self.folder, contrast):
            maps = cache_path(name, phase, cache_root)
            if not os.path.isfile(maps):
                missing_maps += 1
                continue
            with h5py.File(os.path.join(self.folder, name), "r") as handle:
                shape = handle["kspace"].shape
            if image_size is not None and (
                shape[-2] != image_size or shape[-1] != image_size
            ):
                wrong_size += 1
                continue
            self.map_paths[name] = maps
            volumes.append((name, shape[0]))

        if not volumes:
            raise RuntimeError(
                f"no usable volumes in {self.folder}: {missing_maps} without cached "
                f"coil maps, {wrong_size} at the wrong resolution. Run "
                f"precompute_coil_maps.py --phase {phase} first."
            )

        if max_volumes is not None:
            volumes = volumes[: int(max_volumes)]
        self.index = []
        for name, num_slices in volumes:
            if slices_per_volume is not None:
                num_slices = min(num_slices, int(slices_per_volume))
            self.index.extend((name, s) for s in range(num_slices))

        print(
            f"{phase}: {len(self.index)} slices from {len(volumes)} volumes "
            f"({missing_maps} skipped for missing coil maps, {wrong_size} for size)"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        name, s = self.index[i]
        with h5py.File(os.path.join(self.folder, name), "r") as handle:
            kspace = handle["kspace"][s]
        coil_map = np.load(self.map_paths[name], mmap_mode="r")[s]
        return np.asarray(kspace), np.asarray(coil_map)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def get_test_data(
    data_dir,
    phase="test",
    contrast="T1",
    R=4,
    max_volumes=None,
    slices_per_volume=None,
    coil_map_cache=None,
    use_coil_map_cache=True,
    mask_seed=DEFAULT_MASK_SEED,
):
    """Reconstruction inputs: ``(kspace, masks, coil_maps)``.

    ``kspace`` is the undersampled multicoil k-space ``[N, C, H, W]``,
    ``masks`` is ``[N, 1, H, W]`` and ``coil_maps`` is ``[N, C, H, W]``.
    """
    folder = phase_dir(data_dir, phase)
    filenames = list_volumes(folder, contrast, max_volumes)

    # Coil maps depend only on the fully sampled k-space, so they are cached
    # per volume and reused across acceleration rates (see coil_map_cache.py).
    cache_root = resolve_cache_root(data_dir, coil_map_cache)

    print("Reading data from", folder)
    print("Dataset:", phase)
    print("Contrast:", contrast)
    print(f"Volumes: {len(filenames)} file(s)")
    print("Acceleration =", R)
    print("Coil map cache:", cache_root if use_coil_map_cache else "disabled")

    kspace_list = []
    coil_map_list = []
    for filename in tqdm(filenames):
        with h5py.File(os.path.join(folder, filename), "r") as f:
            kspace = f["kspace"][:]

        # Cache whole volumes; truncate afterwards so a run with
        # slices_per_volume set still shares the cache with a full run.
        volume_maps = get_volume_coil_maps(
            kspace,
            filename,
            phase,
            cache_root,
            use_cache=use_coil_map_cache,
        )
        if slices_per_volume is not None:
            volume_maps = volume_maps[: int(slices_per_volume)]
            kspace = kspace[: int(slices_per_volume)]

        coil_map_list.append(volume_maps)
        kspace_list.append(kspace)

    kspace = np.concatenate(kspace_list, axis=0)
    coil_maps = np.concatenate(coil_map_list, axis=0)

    print("Getting Undersampling Masks")
    masks = get_masks(kspace, R, seed=mask_seed)
    print("masks:", masks.shape, masks.dtype, np.mean(masks))
    print("coil maps:", coil_maps.shape, coil_maps.dtype)

    print("Undersampling data")
    kspace_und = kspace * masks
    print("kspace:", kspace_und.shape, kspace_und.dtype)

    return kspace_und, masks, coil_maps


# ---------------------------------------------------------------------------
# Undersampling masks
# ---------------------------------------------------------------------------


def get_acs_region_bounds(height, width, center_fraction=DEFAULT_ACS_CENTER_FRACTION):
    """Center ACS window bounds (matches ``get_masks`` center block)."""
    center_h = int(height * center_fraction)
    center_w = int(width * center_fraction)
    start_h = height // 2 - center_h // 2
    start_w = width // 2 - center_w // 2
    return start_h, start_w, center_h, center_w


def make_mask(height, width, R=4.0, seed=None):
    """Gaussian variable-density undersampling mask ``[H, W]`` with a fully
    sampled ACS block in the center; exactly ``int(H * W / R)`` samples.

    When ``seed`` is set, RNG state is saved and restored so callers can draw
    reproducible, independent masks with ``seed=base_seed + slice_idx``.
    """
    total_pixels = height * width
    if R <= 1.0:
        return np.ones((height, width), dtype=np.float32)

    num_sampled = int(total_pixels / R)
    center_fraction = DEFAULT_ACS_CENTER_FRACTION
    start_h, start_w, center_h, center_w = get_acs_region_bounds(
        height, width, center_fraction=center_fraction
    )

    if (center_h * center_w) > num_sampled:
        center_h = int(np.sqrt(num_sampled))
        center_w = int(np.sqrt(num_sampled))
        start_h = height // 2 - center_h // 2
        start_w = width // 2 - center_w // 2

    y, x = np.indices((height, width))
    y = y - height // 2
    x = x - width // 2
    std_scale = 0.5 / np.sqrt(R)
    std_y, std_x = height * std_scale, width * std_scale
    pdf = np.exp(-(x**2 / (2 * std_x**2) + y**2 / (2 * std_y**2))) + 1e-8
    pdf[start_h : start_h + center_h, start_w : start_w + center_w] = 0.0
    pdf_flat = pdf.flatten()
    pdf_flat /= pdf_flat.sum()
    num_remaining = num_sampled - (center_h * center_w)

    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(int(seed))

    slice_mask_flat = np.zeros(total_pixels, dtype=np.float32)
    if num_remaining > 0:
        sampled_indices = np.random.choice(
            total_pixels,
            size=num_remaining,
            replace=False,
            p=pdf_flat,
        )
        slice_mask_flat[sampled_indices] = 1.0
    slice_mask = slice_mask_flat.reshape(height, width)
    slice_mask[start_h : start_h + center_h, start_w : start_w + center_w] = 1.0

    if rng_state is not None:
        np.random.set_state(rng_state)

    return slice_mask.astype(np.float32, copy=False)


def get_masks(kspace_data, R=4.0, seed=DEFAULT_MASK_SEED):
    """
    Gaussian undersampling masks with exact acceleration rate R.

    Parameters:
    -----------
    kspace_data : np.ndarray
        Fully sampled k-space data of shape (num_slices, num_coils, height, width)
        and dtype complex64.
    R : float
        Acceleration rate. The total number of 1s in the mask for each slice
        will be exactly equal to int((height * width) / R).
    seed : int
        Base mask RNG seed. Slice ``i`` uses ``seed + i``.

    Returns:
    --------
    masks : np.ndarray
        Binary undersampling masks of shape (num_slices, 1, height, width).
    """
    num_slices, _num_coils, height, width = kspace_data.shape
    masks = np.zeros((num_slices, height, width), dtype=np.float32)
    for i in tqdm(range(num_slices)):
        masks[i] = make_mask(height, width, R, seed=int(seed) + i)
    return masks[:, np.newaxis, :, :]


