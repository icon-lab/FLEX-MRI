"""Precompute and cache ESPIRiT coil sensitivity maps for an M4Raw split.

Run it once per split; afterwards both training and reconstruction load the
maps from disk instead of re-running ESPIRiT for every acceleration rate.
Training *requires* it -- volumes without cached maps are skipped by the data
loader.

Example::

    python precompute_coil_maps.py --data_path /path/to/M4Raw/ \\
        --phase train --workers 16

Existing cache entries are skipped unless ``--overwrite`` is given, so the job
can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True,
                        help="Dataset folder containing the multicoil_* directories.")
    parser.add_argument("--phase", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--contrast", default="all",
                        help="all, or a comma separated list such as T1,T2.")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache root (default: $FLEX_MRI_COIL_MAP_CACHE or "
                             "<data_path>/coil_maps_cache).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of volumes processed in parallel.")
    parser.add_argument("--threads_per_worker", type=int, default=2,
                        help="BLAS/OMP threads inside each worker.")
    parser.add_argument("--device", default=None,
                        help="ESPIRiT device: auto, cpu, or cuda:N (GPU needs cupy).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute volumes that are already cached.")
    return parser.parse_args()


def _set_thread_env(threads):
    """Cap BLAS/OMP threads.

    Must run before numpy is imported: the libraries read these variables at
    import time. With the fork start method, workers would inherit an
    already-initialised BLAS and each start a full thread pool, oversubscribing
    the machine by workers x cores. Workers are therefore started with the
    spawn method, and the limits are set in the parent before numpy is
    imported.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(threads)


def _init_worker(threads):
    _set_thread_env(threads)


def _process_volume(job):
    filename, folder, phase, cache_root, device, overwrite = job

    import h5py
    import numpy as np
    from coil_map_cache import cache_path, get_volume_coil_maps, load_cached

    path = cache_path(filename, phase, cache_root)

    with h5py.File(os.path.join(folder, filename), "r") as handle:
        shape = handle["kspace"].shape
        if not overwrite and load_cached(path, expected_shape=shape, mmap=True) is not None:
            return filename, shape[0], "cached"
        kspace = handle["kspace"][:]

    start = time.time()
    maps = get_volume_coil_maps(
        np.asarray(kspace, dtype=np.complex64),
        filename,
        phase,
        cache_root,
        device=device,
        overwrite=overwrite,
    )
    return filename, maps.shape[0], f"computed in {time.time() - start:.1f}s"


def main():
    args = parse_args()

    # Set the thread caps before the first numpy import (see _set_thread_env).
    _set_thread_env(args.threads_per_worker)

    # Import after arg parsing so --help works from any directory.
    from coil_map_cache import espirit_signature, resolve_cache_root

    import read_data

    folder = read_data.phase_dir(args.data_path, args.phase)
    if not os.path.isdir(folder):
        raise SystemExit(f"No such split directory: {folder}")

    cache_root = resolve_cache_root(args.data_path, args.cache_dir)
    contrast = args.contrast
    filenames = read_data.list_volumes(folder, contrast)
    if not filenames:
        raise SystemExit(f"No volumes matching contrast '{contrast}' in {folder}")

    print(f"Split:      {folder}")
    print(f"Contrast:   {contrast} ({len(filenames)} volumes)")
    print(f"Cache root: {cache_root}")
    print(f"Signature:  {espirit_signature()}")
    print(f"Workers:    {args.workers} x {args.threads_per_worker} threads")

    jobs = [
        (name, folder, args.phase, cache_root, args.device, args.overwrite)
        for name in filenames
    ]

    start = time.time()
    done = skipped = 0
    if args.workers <= 1:
        for job in jobs:
            name, slices, status = _process_volume(job)
            done += 1
            skipped += status == "cached"
            print(f"[{done}/{len(jobs)}] {name}: {slices} slices, {status}", flush=True)
    else:
        # The spawn start method makes each worker import numpy under the thread caps.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(args.threads_per_worker,),
        ) as pool:
            futures = {pool.submit(_process_volume, job): job[0] for job in jobs}
            for future in as_completed(futures):
                name, slices, status = future.result()
                done += 1
                skipped += status == "cached"
                print(f"[{done}/{len(jobs)}] {name}: {slices} slices, {status}",
                      flush=True)

    elapsed = time.time() - start
    print(f"\nDone: {done} volumes ({skipped} already cached) in {elapsed / 60:.1f} min")
    print(f"Cache: {os.path.join(cache_root, espirit_signature(), f'multicoil_{args.phase}')}")


if __name__ == "__main__":
    main()
