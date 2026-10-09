import numpy as np
import torch as th

import read_data
from . import logger

# --- FLEX-MRI: FINITE-TRANSITION RECOVERY IN K-SPACE ---


class FlexMRI:
    """
    FLEX-MRI: physics-guided finite-transition MRI reconstruction.

    Representation
    --------------
    Training and sampling both work on the SENSE coil-combined complex image,
    carried as two real channels, in per-slice normalized units (the 99th
    percentile of the zero-filled magnitude). The scale is derived the same
    way in both places, and only from data that exists at test time, so the
    network is never shown a scale it was not trained on.

    Recovery trajectory
    -------------------
    ``x_zf`` is the data-consistent zero-filled reconstruction -- the starting
    point available at inference -- and ``x_0`` is the SENSE combination of
    the fully sampled k-space. The intermediate states are the task-aligned
    states of the paper: they interpolate between the two and are re-projected
    onto the measurements::

        alpha_t = (t + 1) / T
        x_t     = DC( alpha_t * x_zf + (1 - alpha_t) * x_0 )

    The finite-transition predictor predicts the whole remaining jump
    ``x_0 - x_t`` from ``x_t``; sampling walks the same trajectory with ``x_0``
    replaced by the running estimate. ``t = T-1`` is therefore the zero-filled
    state and ``t = 0`` is the final refinement step.

    Why the target is the fully sampled image
    -----------------------------------------
    Regressing onto the difference between two randomly masked versions of an
    image would not be learnable: which coefficients get dropped is randomness
    the input does not determine, so the L1-optimal prediction is zero.
    Regressing the jump onto the fully sampled image makes the target a
    deterministic function of the input.

    Training masks come from the same ``read_data.make_mask`` used at test
    time, drawn over a range of accelerations, so together with the shared
    representation and normalization above, the network sees the same kind of
    input in training and in sampling.

    Stochastic exploration
    ----------------------
    ``drop_rate`` perturbs a random subset of the estimated k-space
    coefficients before the predictor sees them. It is applied during training
    only and defaults to 0.0; ``recovery_trajectory`` is deterministic.

    Example:
        flex = FlexMRI(steps=10, image_size=256, r_min=2.0, r_max=8.0)
        image, rss = flex.reconstruct(model, kspace, mask, coil_map)
    """

    def __init__(
        self,
        steps,
        image_size,
        r_min=2.0,
        r_max=8.0,
        drop_rate=0.0,
        dc_weight=1.0,
    ):
        self.num_timesteps = steps
        self.image_size = image_size
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.drop_rate = float(drop_rate)
        self.dc_weight = float(dc_weight)

        logger.log(
            f"\n\n Transitions = {self.num_timesteps}\n Image Size = {self.image_size}\n "
            f"Acceleration Range = [{self.r_min}, {self.r_max}]\n "
            f"Train Dropout = {self.drop_rate}\n DC Weight = {self.dc_weight}\n\n"
        )

    # --- TRANSFORMS ---

    def fft2c(self, x):
        return th.fft.fftshift(th.fft.fft2(th.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))

    def ifft2c(self, x, dim=((-2, -1)), img_shape=None):
        return th.fft.fftshift(th.fft.ifft2(th.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim=dim)

    @staticmethod
    def to_complex(x):
        """Real-stacked coils ``[B, 2C, H, W]`` back to complex ``[B, C, H, W]``."""
        channels = x.shape[1] // 2
        return th.complex(x[:, :channels], x[:, channels:]).contiguous()

    @staticmethod
    def as_complex_image(img):
        """Two-channel image ``[B, 2, H, W]`` to complex ``[B, 1, H, W]``."""
        return img[:, [0]] + 1j * img[:, [1]]

    def sense_combine(self, coil_images, coil_map):
        """``S^H``: coil images to one complex image, as two real channels."""
        combined = th.sum(th.conj(coil_map) * coil_images, dim=1).unsqueeze(1)
        return th.cat([combined.real, combined.imag], 1).to(th.float32)

    # --- PHYSICS ---

    def data_consistency(self, img, kspace, mask, coil_map):
        """Re-impose the acquired frequencies on ``S * img``, then recombine.

        ``dc_weight`` < 1 keeps part of the network's own estimate on measured
        lines, which trades fidelity for noise suppression; 1.0 is hard DC.
        """
        coil_est = self.fft2c(self.as_complex_image(img) * coil_map)
        w = self.dc_weight
        merged = mask * (w * kspace + (1.0 - w) * coil_est) + (1.0 - mask) * coil_est
        return self.sense_combine(self.ifft2c(merged), coil_map)

    def zero_filled(self, kspace, mask, coil_map):
        """Data-consistent zero-filled SENSE reconstruction: the source state.

        The adjoint ``S^H`` is used rather than a plain coil mean, so the
        starting point already accounts for the sensitivities.
        """
        adjoint = self.sense_combine(self.ifft2c(kspace), coil_map)
        return self.data_consistency(adjoint, kspace, mask, coil_map)

    def normalization(self, kspace, coil_map):
        """Per-slice scale: 99th percentile of the zero-filled magnitude.

        Depends only on the undersampled measurements, so training and
        sampling compute it identically.
        """
        zf = self.sense_combine(self.ifft2c(kspace), coil_map)
        magnitude = th.sqrt(zf[:, 0] ** 2 + zf[:, 1] ** 2 + 1e-12).flatten(1)
        scale = th.quantile(magnitude, 0.99, dim=1).clamp(min=1e-12)
        return scale.view(-1, 1, 1, 1)

    def sample_masks(self, batch_size, height, width, device):
        """Gaussian-density masks at random acceleration, as generated at test time."""
        masks = np.empty((batch_size, 1, height, width), dtype=np.float32)
        for i in range(batch_size):
            R = float(np.random.uniform(self.r_min, self.r_max))
            masks[i, 0] = read_data.make_mask(height, width, R)
        return th.from_numpy(masks).to(device)

    def interpolate(self, x_zf, x_target, t):
        """State the schedule assigns to ``t``: zero-filled at ``T-1``, target at ``-1``."""
        alpha = ((t.float() + 1.0) / self.num_timesteps).view(-1, 1, 1, 1)
        return alpha * x_zf + (1.0 - alpha) * x_target

    def _stochastic_frequency_dropout(self, x, drop_rate):
        """I.i.d. Bernoulli dropout in k-space; used only as training augmentation."""
        B, _, H, W = x.shape
        if drop_rate <= 0:
            return x
        x_kspace = self.fft2c(self.as_complex_image(x)[:, 0])
        keep = (th.rand((B, H, W), device=x.device) > drop_rate).float()
        dropped = self.ifft2c(x_kspace * keep)
        return th.stack([dropped.real, dropped.imag], dim=1).to(th.float32)

    # --- NETWORK ---

    def p_forward(self, model, x, t, model_kwargs=None):
        return model(x, t, **(model_kwargs or {}))

    def p_sample(self, model, x, t, model_kwargs=None):
        """Predicted fully sampled image: the state plus the predicted jump."""
        mu_diff = self.p_forward(model, x, t, model_kwargs)
        pred_xstart = x + mu_diff
        return {"sample": pred_xstart, "pred_xstart": pred_xstart, "model_output": mu_diff}

    def training_losses(self, model, kspace_ri, t, step, save_dir, model_kwargs=None, noise=None):
        """L1 between the predicted jump and ``x_0 - x_t``.

        ``kspace_ri`` is the fully sampled k-space as real-stacked coils;
        ``model_kwargs["coil_map"]`` carries the ESPIRiT maps the same way.
        Both are consumed here rather than forwarded to the network.
        """
        model_kwargs = dict(model_kwargs or {})
        coil_map = self.to_complex(model_kwargs.pop("coil_map"))
        kspace_full = self.to_complex(kspace_ri)
        B, _, H, W = kspace_full.shape

        mask = self.sample_masks(B, H, W, kspace_full.device)
        kspace_und = kspace_full * mask

        scale = self.normalization(kspace_und, coil_map)
        kspace_und = kspace_und / scale
        kspace_full = kspace_full / scale

        x_target = self.sense_combine(self.ifft2c(kspace_full), coil_map)
        x_zf = self.zero_filled(kspace_und, mask, coil_map)
        x_t = self.data_consistency(
            self.interpolate(x_zf, x_target, t), kspace_und, mask, coil_map
        )
        x_t = self._stochastic_frequency_dropout(x_t, self.drop_rate)

        mu_diff_pred = self.p_forward(model, x_t, t, model_kwargs)
        true_diff = x_target - x_t

        # L1 rather than MSE: sparser error distribution and better
        # high-frequency behaviour. Scaled by 10 to keep gradients in the same
        # regime the learning rate was tuned for.
        loss = th.nn.functional.l1_loss(mu_diff_pred, true_diff) * 10.0

        if step % 100 == 0:
            with th.no_grad():
                baseline = true_diff.abs().mean().item()
                residual = (mu_diff_pred - true_diff).abs().mean().item()
            # A ratio below 1 means the prediction improves on predicting zero.
            logger.log(
                f"Step: {step} | Total Loss: {loss.item():.4f} | "
                f"residual/zero-pred: {residual / max(baseline, 1e-12):.3f}"
            )

        return {"loss": loss}

    # --- LOOPS ---

    def recovery_trajectory(self, model, shape, kspace, mask, coil_map):
        """Walk the training trajectory, substituting the estimate for ``x_0``.

        Yields the reconstruction after each finite-transition, denormalized
        back into the units of the supplied k-space.
        """
        scale = self.normalization(kspace, coil_map)
        kspace = kspace / scale

        x_zf = self.zero_filled(kspace, mask, coil_map)
        x = x_zf

        for i in list(range(self.num_timesteps))[::-1]:
            t = th.full((shape[0],), i, device=x.device, dtype=th.long)
            with th.no_grad():
                pred_xstart = self.p_sample(model, x, t)["pred_xstart"]
            # Step to the state the schedule assigns to t-1.
            alpha_next = float(i) / self.num_timesteps
            x = self.data_consistency(
                alpha_next * x_zf + (1.0 - alpha_next) * pred_xstart, kspace, mask, coil_map
            )
            yield x * scale

    def reconstruct(self, model, kspace, mask, coil_map):
        """Run the full trajectory.

        Returns the SENSE image ``[B, 2, H, W]`` and the RSS image
        ``[B, H, W]``, both in the units of the supplied k-space. RSS is the
        convention of ``reconstruction_rss``, so it is the output to score
        against the M4Raw ground truth; the SENSE image is a different quantity
        and is not directly comparable to an RSS reference.
        """
        image = None
        for image in self.recovery_trajectory(
            model, (kspace.shape[0],), kspace, mask, coil_map
        ):
            pass

        coil_images = self.ifft2c(
            mask * kspace + (1.0 - mask) * self.fft2c(self.as_complex_image(image) * coil_map)
        )
        rss = th.sqrt((th.abs(coil_images) ** 2).sum(dim=1))
        return image, rss
