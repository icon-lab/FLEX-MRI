"""Score reconstructions against the M4Raw ``reconstruction_rss`` ground truth.

``reconstruction_rss`` is an RSS magnitude image, so reconstructions are
scored as RSS images as well: ``sample.py`` saves ``rss<i>.npy`` computed from
the sampler's own coil images. The ``sqrt(H*W)`` factor converts units -- the transforms are
unnormalized FFTs while ``reconstruction_rss`` is orthonormal.

Zero-filling is scored alongside as a baseline. Its masks are regenerated with
the seeds ``sample.py`` used: slice ``i`` of the concatenated test set of one
contrast uses ``seed + i``. Pass the same ``--max_volumes`` /
``--slices_per_volume`` as the reconstruction run so the indices line up.

    python evaluate.py --data_path /path/to/M4Raw/ --results results/exp
"""

import argparse
import os

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import read_data


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))


def metrics(rec, ref):
    ref = np.clip(ref, a_min=0, a_max=None).astype(np.float64)
    rec = rec.astype(np.float64)
    data_range = ref.max() - ref.min()
    return psnr(ref, rec, data_range=data_range), ssim(ref, rec, data_range=data_range)


def evaluate(args, R, contrast):
    folder = read_data.phase_dir(args.data_path, args.phase)
    filenames = read_data.list_volumes(folder, contrast, optional_int(args.max_volumes))
    slices_per_volume = optional_int(args.slices_per_volume)
    imgs_dir = os.path.join(args.results, f"{R}x_{contrast}")

    scores = {"psnr": [], "ssim": [], "psnr_zf": [], "ssim_zf": []}
    if not os.path.isdir(imgs_dir):
        return scores, 0
    missing = 0
    index = 0

    for filename in filenames:
        with h5py.File(os.path.join(folder, filename), "r") as f:
            kspace = f["kspace"][:]
            reference = f["reconstruction_rss"][:]
        num_slices = kspace.shape[0]
        if slices_per_volume is not None:
            num_slices = min(num_slices, slices_per_volume)

        for s in range(num_slices):
            idx = index
            index += 1
            if idx % args.interval:
                continue

            path = os.path.join(imgs_dir, f"rss{idx}.npy")
            if not os.path.isfile(path):
                missing += 1
                continue

            height, width = kspace.shape[-2:]
            scale = np.sqrt(height * width)

            mask = read_data.make_mask(height, width, R, seed=args.mask_seed + idx)[np.newaxis]
            kspace_und = kspace[s] * mask

            recon = np.load(path) * scale
            zero_filled = np.sqrt((np.abs(ifft2c(kspace_und)) ** 2).sum(axis=0)) * scale

            p, q = metrics(recon, reference[s])
            scores["psnr"].append(p)
            scores["ssim"].append(q)
            p, q = metrics(zero_filled, reference[s])
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
