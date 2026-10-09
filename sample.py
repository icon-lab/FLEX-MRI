"""Reconstruct undersampled M4Raw data with a trained FLEX-MRI model."""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as th

import read_data
from utils import dist_util, logger
from utils.script_util import (
    model_and_transition_defaults,
    create_model_and_transition,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    for name in ("model_path", "data_path", "save_path"):
        if not getattr(args, name):
            raise SystemExit(f"--{name} is required")

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    kspaces, masks, coil_maps = read_data.get_test_data(
        args.data_path,
        phase=args.phase,
        contrast=args.contrast,
        R=args.R,
        max_volumes=optional_int(args.max_volumes),
        slices_per_volume=optional_int(args.slices_per_volume),
        use_coil_map_cache=args.use_coil_map_cache,
        coil_map_cache=args.coil_map_cache or None,
    )

    logger.configure(dir=args.save_path)
    logger.log("creating model and transition process...")

    model, transition = create_model_and_transition(
        **args_to_dict(args, model_and_transition_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    with th.no_grad():
        for index in range(0, len(kspaces)):

            mask = th.from_numpy(np.asarray(masks[index])).to(device).unsqueeze(0)
            kspace = th.from_numpy(np.asarray(kspaces[index])).to(device).unsqueeze(0)
            coil_map = th.from_numpy(np.asarray(coil_maps[index])).to(device).unsqueeze(0)

            print(f"Index: {index}")
            start_time = time.time()

            sense_img, rss_img = transition.reconstruct(model, kspace, mask, coil_map)

            total_time = time.time() - start_time
            print(f"total time: {total_time:.4f}")

            # RSS is the convention reconstruction_rss uses, so it is what
            # evaluate.py scores. The complex SENSE image (sense*.npy,
            # [2, H, W] real/imag, ~15 GB per full sweep) is for comparisons
            # that need phase; --save_sense False skips it.
            rss_np = rss_img[0].contiguous().cpu().numpy()
            np.save(os.path.join(args.save_path, f"rss{index}.npy"), rss_np)
            if args.save_sense:
                np.save(os.path.join(args.save_path, f"sense{index}.npy"),
                        sense_img[0].float().cpu().numpy())

            plt.imsave(
                os.path.join(args.save_path, f"image_{index}.png"),
                rss_np / max(rss_np.max(), 1e-12),
                cmap='gray',
            )

            print(f"Saved to: {args.save_path}")


def optional_int(value):
    """Treat 0 / negative command line values as 'no limit'."""
    value = int(value)
    return value if value > 0 else None


def create_argparser():
    defaults = dict(
        model_path="",
        data_path="",
        save_path="",
        phase="test",
        R=4,
        contrast="T1",
        max_volumes=0,  # 0 = every volume
        slices_per_volume=0,  # 0 = every slice
        coil_map_cache="",  # empty = $FLEX_MRI_COIL_MAP_CACHE or <data_path>/coil_maps_cache
        use_coil_map_cache=True,
        save_sense=True,  # also save the complex SENSE image
    )
    defaults.update(model_and_transition_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
