"""Train FLEX-MRI on M4Raw."""

import argparse

from utils import dist_util, logger
from utils.kspace_dataset import load_data
from utils.script_util import (
    model_and_transition_defaults,
    create_model_and_transition,
    args_to_dict,
    add_dict_to_argparser,
)
from utils.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(args, '\n')

    logger.configure(args.save_dir)

    logger.log("creating model and transition process...")
    model, transition = create_model_and_transition(
        **args_to_dict(args, model_and_transition_defaults().keys())
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        phase=args.phase,
        contrast=args.contrast,
        max_volumes=optional_int(args.max_volumes),
        slices_per_volume=optional_int(args.slices_per_volume),
        num_workers=args.num_workers,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        transition=transition,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir,
        accumulation_steps=args.accumulation_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
    ).run_loop()


def optional_int(value):
    """Treat 0 / negative command line values as 'no limit'."""
    value = int(value)
    return value if value > 0 else None


def create_argparser():
    defaults = dict(
        data_dir="",
        phase="train",
        contrast="all",  # 'all', or a comma separated list such as T1,T2
        max_volumes=0,  # 0 = every volume
        slices_per_volume=0,  # 0 = every slice
        num_workers=8,  # slices are read from disk on access
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",  # continue a run: weights, EMA, optimizer and step count
        accumulation_steps=8,  # iterations per optimizer step
        warmup_steps=5000,  # linear LR warmup, then cosine decay to min_lr
        min_lr=1e-6,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_dir='logs',
    )
    defaults.update(model_and_transition_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
