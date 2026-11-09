"""Score reconstructions with the fastMRI evaluation protocol.

The magnitude of the reconstructed coil-combined (SENSE) image,
``sense<i>.npy`` from ``sample.py``, is compared with the magnitude of the
coil-combined fully sampled image ``S^H F^-1 k`` (the same ESPIRiT maps the
reconstruction used):

  * the reconstruction is zeroed wherever the reference is zero (outside the
    ESPIRiT support, i.e. the background),
  * both images are divided by their mean,
  * PSNR and SSIM use the reference's data range.

Zero-filling, ``S^H F^-1 (M k)``, is scored alongside as a baseline. Its
masks are regenerated with the seeds ``sample.py`` used: slice ``i`` of the
concatenated test set of one contrast uses ``seed + i``. Pass the same
``--max_volumes`` / ``--slices_per_volume`` as the reconstruction run so the
indices line up.

    python evaluate.py --data_path /path/to/M4Raw/ --results results/exp
"""

import argparse
import os

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import read_data
from coil_map_cache import get_volume_coil_maps, resolve_cache_root


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))


def sense_magnitude(coil_images, coil_map):
    """``|S^H x|`` for coil images ``[C, H, W]``."""
    return np.abs(np.sum(np.conj(coil_map) * coil_images, axis=0))


def metrics(ref, rec):
    """PSNR / SSIM with the fastMRI protocol's masking and mean normalization."""
    ref = np.abs(ref).astype(np.float64)
    rec = np.abs(rec).astype(np.float64)
    background = ref == 0
    rec[background] = 0
    ref = ref / ref.mean()
    rec = rec / rec.mean()
    rec[background] = 0
    data_range = ref.max() - ref.min()
    return psnr(ref, rec, data_range=data_range), ssim(ref, rec, data_range=data_range)


def evaluate(args, R, contrast):
    folder = read_data.phase_dir(args.data_path, args.phase)
    filenames = read_data.list_volumes(folder, contrast, optional_int(args.max_volumes))
    slices_per_volume = optional_int(args.slices_per_volume)
    cache_root = resolve_cache_root(args.data_path, args.coil_map_cache or None)
    imgs_dir = os.path.join(args.results, f"{R}x_{contrast}")

    scores = {"psnr": [], "ssim": [], "psnr_zf": [], "ssim_zf": []}
    if not os.path.isdir(imgs_dir):
        return scores, 0
    missing = 0
    index = 0

    for filename in filenames:
        with h5py.File(os.path.join(folder, filename), "r") as f:
            kspace = f["kspace"][:]
        coil_maps = get_volume_coil_maps(kspace, filename, args.phase, cache_root)
        num_slices = kspace.shape[0]
        if slices_per_volume is not None:
            num_slices = min(num_slices, slices_per_volume)

        for s in range(num_slices):
            idx = index
            index += 1
            if idx % args.interval:
                continue

            path = os.path.join(imgs_dir, f"sense{idx}.npy")
            if not os.path.isfile(path):
                missing += 1
                continue

            height, width = kspace.shape[-2:]
            mask = read_data.make_mask(height, width, R, seed=args.mask_seed + idx)[np.newaxis]

            reference = sense_magnitude(ifft2c(kspace[s]), coil_maps[s])
            zero_filled = sense_magnitude(ifft2c(kspace[s] * mask), coil_maps[s])
            sense = np.load(path)
            recon = np.abs(sense[0] + 1j * sense[1])

            p, q = metrics(reference, recon)
            scores["psnr"].append(p)
            scores["ssim"].append(q)
            p, q = metrics(reference, zero_filled)
            scores["psnr_zf"].append(p)
            scores["ssim_zf"].append(q)

    return scores, missing


def optional_int(value):
    """Treat 0 / negative values as 'no limit'."""
    value = int(value)
    return value if value > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_path", required=True,
                        help="Dataset folder containing the multicoil_* directories.")
    parser.add_argument("--results", required=True,
                        help="Folder holding the <R>x_<contrast> directories from test.sh.")
    parser.add_argument("--R_values", default="2,4,6,8")
    parser.add_argument("--contrasts", default="T1,T2,FLAIR")
    parser.add_argument("--phase", default="test")
    parser.add_argument("--interval", type=int, default=1,
                        help="Score every n-th slice (1 = all).")
    parser.add_argument("--max_volumes", type=int, default=0, help="0 = every volume.")
    parser.add_argument("--slices_per_volume", type=int, default=0, help="0 = every slice.")
    parser.add_argument("--mask_seed", type=int, default=read_data.DEFAULT_MASK_SEED)
    parser.add_argument("--coil_map_cache", default="",
                        help="Empty = $FLEX_MRI_COIL_MAP_CACHE or <data_path>/coil_maps_cache.")
    args = parser.parse_args()

    scored = 0
    for R in [int(r) for r in args.R_values.split(",") if r.strip()]:
        for contrast in [c.strip() for c in args.contrasts.split(",") if c.strip()]:
            scores, missing = evaluate(args, R, contrast)
            print(R, contrast)
            if not scores["psnr"]:
                print(f"no reconstructions found in {args.results}/{R}x_{contrast}\n")
                continue
            scored += len(scores["psnr"])
            print(f"PSNR = {np.nanmean(scores['psnr']):.3f} +- {np.nanstd(scores['psnr']):.3f}"
                  f"   (zero-filled {np.nanmean(scores['psnr_zf']):.3f})")
            print(f"SSIM = {np.nanmean(scores['ssim']):.3f} +- {np.nanstd(scores['ssim']):.3f}"
                  f"   (zero-filled {np.nanmean(scores['ssim_zf']):.3f})")
            if missing:
                print(f"({missing} slice(s) without a reconstruction were skipped)")
            print()

    if not scored:
        raise SystemExit(f"nothing to score under {args.results}")


if __name__ == "__main__":
    main()
