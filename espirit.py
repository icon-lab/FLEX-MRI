"""ESPIRiT coil sensitivity estimation (sigpy)."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np

DEFAULT_ACS_CENTER_FRACTION = 0.08
DEFAULT_ESPIRIT_CROP = 0.8
DEFAULT_ESPIRIT_KERNEL_WIDTH = 6
DEFAULT_ESPIRIT_THRESH = 0.02
#: Device used for ESPIRiT; override with FLEX_MRI_ESPIRIT_DEVICE (auto, cpu, cuda:N).
ESPIRIT_DEVICE_ENV_VAR = "FLEX_MRI_ESPIRIT_DEVICE"
DEFAULT_ESPIRIT_DEVICE = os.environ.get(ESPIRIT_DEVICE_ENV_VAR, "auto")


def resolve_espirit_device(
    device: Optional[Union[str, int]] = None,
) -> Tuple[object, str]:
    """Resolve sigpy device for ESPIRiT (GPU via cupy when available)."""
    import sigpy as sp

    if device is None:
        device = DEFAULT_ESPIRIT_DEVICE

    if isinstance(device, str):
        name = device.strip().lower()
        if name in {"cpu", "none", "false"}:
            return sp.cpu_device, "cpu"
        if name in {"auto", ""}:
            try:
                import cupy

                if cupy.cuda.runtime.getDeviceCount() > 0:
                    return sp.Device(0), "cuda:0"
            except Exception:
                pass
            return sp.cpu_device, "cpu"
        if name.startswith("cuda:"):
            name = name.split(":", 1)[1]
        if name.isdigit():
            device = int(name)
        else:
            raise ValueError(
                f"Unsupported ESPIRiT device '{device}'. "
                "Use auto, cpu, 0, or cuda:N."
            )

    if isinstance(device, int):
        try:
            return sp.Device(int(device)), f"cuda:{int(device)}"
        except ValueError as exc:
            raise ValueError(
                f"ESPIRiT GPU device {device} unavailable. "
                "Install cupy for your CUDA version (e.g. cupy-cuda12x)."
            ) from exc

    raise TypeError(f"Unsupported ESPIRiT device type: {type(device)}")


def espirit_calib_width(
    height: int,
    width: int,
    center_fraction: float = DEFAULT_ACS_CENTER_FRACTION,
) -> int:
    """ESPIRiT ``calib_width`` matched to the Gaussian mask ACS center (~8%)."""
    center_h = int(height * center_fraction)
    center_w = int(width * center_fraction)
    return int(max(center_h, center_w))


def espirit_coil_maps_from_kspace(
    kspace: np.ndarray,
    *,
    center_fraction: float = DEFAULT_ACS_CENTER_FRACTION,
    kernel_width: int = DEFAULT_ESPIRIT_KERNEL_WIDTH,
    thresh: float = DEFAULT_ESPIRIT_THRESH,
    crop: float = DEFAULT_ESPIRIT_CROP,
    device: Optional[Union[str, int]] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Estimate coil sensitivity maps with sigpy ESPIRiT.

    Parameters
    ----------
    kspace
        ``(C, H, W)`` or ``(N, C, H, W)`` multicoil k-space (fully sampled).
    device
        ``auto`` (default), ``cpu``, GPU index, or ``cuda:N``. GPU requires cupy.

    Returns
    -------
    np.ndarray
        ESPIRiT maps with the same shape as ``kspace`` (NumPy on CPU).
    """
    try:
        import sigpy as sp
        import sigpy.mri as mr
    except ImportError as exc:
        raise ImportError(
            "sigpy is required for ESPIRiT coil maps. Install with: pip install sigpy"
        ) from exc

    from tqdm import tqdm

    sigpy_device, _device_name = resolve_espirit_device(device)

    def _run_slice(kspace_slice: np.ndarray) -> np.ndarray:
        kspace_slice = np.asarray(kspace_slice, dtype=np.complex64)
        calib_width = espirit_calib_width(
            int(kspace_slice.shape[-2]),
            int(kspace_slice.shape[-1]),
            center_fraction=center_fraction,
        )
        k_dev = sp.to_device(kspace_slice, sigpy_device)
        app = mr.app.EspiritCalib(
            k_dev,
            calib_width=calib_width,
            kernel_width=int(kernel_width),
            thresh=float(thresh),
            crop=float(crop),
            show_pbar=False,
            device=sigpy_device,
        )
        maps = sp.to_device(app.run(), sp.cpu_device)
        return np.asarray(maps, dtype=np.complex64)

    kspace = np.asarray(kspace, dtype=np.complex64)
    if kspace.ndim == 3:
        return _run_slice(kspace)

    if kspace.ndim == 4:
        maps = np.empty_like(kspace)
        slice_iter = range(kspace.shape[0])
        if show_progress:
            slice_iter = tqdm(slice_iter, desc="ESPIRiT coil maps")
        for slice_idx in slice_iter:
            maps[slice_idx] = _run_slice(kspace[slice_idx])
        return maps

    raise ValueError(
        f"Expected k-space shape (C,H,W) or (N,C,H,W), got {kspace.shape}"
    )


