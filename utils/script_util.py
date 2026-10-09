import argparse

from . import finite_transition
from .unet import UNetModel


def model_and_transition_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=256,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="32,16",
        dropout=0.0,
        transition_steps=10,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        r_min=2.0,
        r_max=8.0,
    )


def create_model_and_transition(
    image_size,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    transition_steps,
    use_checkpoint,
    use_scale_shift_norm,
    r_min,
    r_max,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    transition = create_finite_transition(
        transition_steps,
        image_size,
        r_min,
        r_max,
    )
    return model, transition


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=2,
        model_channels=num_channels,
        out_channels=2,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_finite_transition(
    steps,
    image_size,
    r_min,
    r_max,
):
    return finite_transition.FlexMRI(
        steps=steps,
        image_size=image_size,
        r_min=r_min,
        r_max=r_max,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Parse a boolean command line value (yes/no, true/false, t/f, y/n, 1/0).
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
