import copy
import os
import math

import numpy as np
import torch as th
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import UniformSampler

# Initial log2 loss scale for fp16 training (the improved-diffusion default).
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        transition,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        save_dir='',
        accumulation_steps=8,
        warmup_steps=5000,
        min_lr=1e-6,
    ):
        self.model = model
        self.transition = transition
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(transition)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.save_dir = save_dir

        # Iterations per optimizer step, and the LR schedule's warmup length and
        # floor (all in iterations).
        self.accumulation_steps = accumulation_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        self.step = 0
        self.resume_step = 0

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self._load_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        zero_grad(self.model_params)

    def _load_parameters(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    self.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        ema_checkpoint = find_ema_checkpoint(self.resume_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)
        else:
            logger.warn(f"no EMA checkpoint for rate {rate}; restarting it from the weights")

        return ema_params

    def _load_optimizer_state(self):
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            logger.warn(f"no optimizer state at {opt_checkpoint}; AdamW restarts from zero")

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (self.step + self.resume_step < self.lr_anneal_steps):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step != 0:
                self.save()
            self.step += 1
        # Save the last checkpoint if the final iteration did not already.
        last = self.step - 1
        if self.step > 0 and not (last != 0 and last % self.save_interval == 0):
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        # Accumulate gradients over `accumulation_steps` iterations, then step.
        if (self.step + 1) % self.accumulation_steps == 0:
            if self.use_fp16:
                self.optimize_fp16()
            else:
                self.optimize_normal()
            zero_grad(self.model_params)

        self.log_step()

    def forward_backward(self, batch, cond):
        for i in range(0, batch.shape[0], self.microbatch):
            # `micro` is raw k-space, not an image: training_losses derives the
            # images, the mask and the per-slice scale, using the same code
            # sampling does.
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            losses = self.transition.training_losses(
                self.model,
                micro,
                t,
                self.step,
                self.save_dir,
                model_kwargs=micro_cond,
            )

            # Weight each microbatch by its share of the batch, so splitting a
            # batch into microbatches leaves the gradient unchanged.
            share = micro.shape[0] / batch.shape[0]
            loss = (losses["loss"] * weights).mean() * share / self.accumulation_steps
            log_loss_dict(
                self.transition, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        th.nn.utils.clip_grad_norm_(self.master_params, 1.0)
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return

        current_step = self.step + self.resume_step
        min_lr = self.min_lr

        if current_step < self.warmup_steps:
            # Linear Warmup
            lr = min_lr + (self.lr - min_lr) * (current_step / self.warmup_steps)
        else:
            # Cosine Decay
            frac_done = (current_step - self.warmup_steps) / (self.lr_anneal_steps - self.warmup_steps)
            frac_done = min(max(frac_done, 0.0), 1.0)
            lr = min_lr + 0.5 * (self.lr - min_lr) * (1 + math.cos(math.pi * frac_done))

        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        step = self.step + self.resume_step

        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{step:06d}.pt"
            else:
                filename = f"ema_{rate}_{step:06d}.pt"
            th.save(state_dict, os.path.join(logger.get_dir(), filename))

        # Save the raw weights as well as the EMA: at ema_rate 0.9999 the
        # average lags about 7k iterations behind the weights.
        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        # The optimizer state, so --resume_checkpoint continues AdamW's moments.
        th.save(self.opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = os.path.basename(filename).split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_ema_checkpoint(main_checkpoint, step, rate):
    if not main_checkpoint:
        return None
    path = os.path.join(os.path.dirname(main_checkpoint), f"ema_{rate}_{step:06d}.pt")
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(transition, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / transition.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
