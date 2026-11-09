import numpy as np
import torch as th

import read_data
from . import logger

# --- FLEX-MRI: FINITE-TRANSITION LEARNING WITH K-SPACE SUPPORT STATES ---


class FlexMRI:
    """
    FLEX-MRI: stochastic finite-transition learning and exploration
    (Sec. III of the paper; equation names below refer to it).

    Retained states (Eqs. forward_mask, forward_state_random)
    ---------------------------------------------------------
    ``x_0`` is the fully sampled image. ``Omega`` is the support of the
    undersampled endpoint ``x_T``: a variable-density mask at
    ``endpoint_R`` (6 in the paper) from the same generator used at test time.
    Every coefficient outside ``Omega`` gets a priority ``rho_i ~ U(0, 1)``,
    and state ``t`` keeps it while ``rho_i >= nu_t``, with equispaced
    thresholds ``nu_t = t / T``::

        m_t = Omega  +  { i not in Omega : rho_i >= t / T }
        x_t = F^-1[ m_t * F(x_0) ]

    so ``x_0`` is fully sampled, ``x_T`` is the endpoint, and the masks are
    nested (``m_t`` is contained in ``m_{t-1}``). The states are binary
    changes in k-space support, not interpolations.

    Finite-transition target and objective (Eqs. forward_transition_increment,
    finite_transition_objective)
    ---------------------------------------------------------------------
    Transition ``j`` goes from ``x_j`` to ``x_{j-1}``. Its target is the
    individual finite increment

        Delta_j = x_{j-1} - x_j = F^-1[ eta_j * F(x_0) ],
        eta_j   = m_{j-1} - m_j,

    and the predictor ``mu_theta(x_j, j - 1)`` is trained with the
    support-relaxed objective ``10 * || mu_theta - Delta_j ||_1`` (mean
    absolute error; the transition mask is not applied to the prediction
    error). Each transition is
    therefore learned at the scale it is executed at inference; nothing is
    regressed onto ``x_0`` directly. ``objective="restricted"`` evaluates the
    error only on ``eta_j`` (the "Restricted" ablation), and
    ``train_perturbation > 0`` perturbs the training inputs with the same
    schedule as inference (the "Perturbed" ablation); the proposed setting
    uses neither.

    Stochastic recovery (Eqs. stochastic_perturb, stochastic_prediction;
    Algorithm 1)
    ---------------------------------------------------------------------
    Starting from the zero-filled endpoint, for ``j = J, ..., 1``:

        gamma_j = gamma * rho_hat * (j - 1) / J
        z_j[i] ~ Bernoulli(1 - gamma_j)                    (all of k-space)
        x~     = F^-1[ z_j * F(x_j) ]                      perturbation
        x_pred = x~ + mu_theta(x~, j - 1)                  finite transition
        x_{j-1} = DC(x_pred)                               hard data consistency

    ``rho_hat`` is the unacquired fraction of k-space, so the perturbation
    rate is largest at the first transition and decays linearly to zero at
    the last; ``gamma = 0.1`` is the base rate (0 gives the deterministic
    ablation). Data consistency re-imposes the acquired coefficients only;
    the unacquired ones come from the prediction.

    Multi-coil data
    ---------------
    The theory suppresses coil encoding. Here images are SENSE
    coil-combined complex images (``S^H F^-1 k`` with ESPIRiT maps), carried
    as two real channels, and data consistency uses the coil maps. Each slice
    is divided by the 99th percentile of its zero-filled magnitude, which only
    needs the undersampled data, so training and reconstruction share it. At
    training the scale is computed under a test-style mask at
    ``R ~ U[r_min, r_max]`` so the network sees the range of scales the test
    sweep produces.

    Example:
        flex = FlexMRI(steps=10, image_size=256)
        image, rss = flex.reconstruct(model, kspace, mask, coil_map)
    """

    def __init__(
        self,
        steps,
        image_size,
        endpoint_R=6.0,
        gamma=0.1,
        objective="relaxed",
        train_perturbation=0.0,
        r_min=2.0,
        r_max=8.0,
    ):
        if objective not in ("relaxed", "restricted"):
            raise ValueError(f"objective must be 'relaxed' or 'restricted', got {objective!r}")
        self.num_timesteps = steps
        self.image_size = image_size
        self.endpoint_R = float(endpoint_R)
        self.gamma = float(gamma)
        self.objective = objective
        self.train_perturbation = float(train_perturbation)
        self.r_min = float(r_min)
        self.r_max = float(r_max)

        logger.log(
            f"\n\n Transitions J = {self.num_timesteps}\n Image Size = {self.image_size}\n "
            f"Endpoint R = {self.endpoint_R}\n Base perturbation rate gamma = {self.gamma}\n "
            f"Objective = {self.objective}\n Training-input perturbation = {self.train_perturbation}\n "
            f"Normalization R range = [{self.r_min}, {self.r_max}]\n\n"
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

    @staticmethod
    def as_two_channel(img):
        """Complex ``[B, 1, H, W]`` (or ``[B, H, W]``) to two real channels."""
        if img.dim() == 3:
            img = img.unsqueeze(1)
        return th.cat([img.real, img.imag], 1).to(th.float32)

    def sense_combine(self, coil_images, coil_map):
        """``S^H``: coil images to one complex image, as two real channels."""
        combined = th.sum(th.conj(coil_map) * coil_images, dim=1).unsqueeze(1)
        return self.as_two_channel(combined)

    def apply_kspace_mask(self, img, mask):
        """``F^-1[ mask * F(img) ]`` for a two-channel image."""
        return self.as_two_channel(self.ifft2c(mask * self.fft2c(self.as_complex_image(img))))

    # --- PHYSICS ---

    def data_consistency(self, img, kspace, mask, coil_map):
        """Hard DC: replace the acquired coefficients of ``S * img`` by the
        measurements, keep the estimate elsewhere, then recombine with ``S^H``."""
        coil_est = self.fft2c(self.as_complex_image(img) * coil_map)
        merged = mask * kspace + (1.0 - mask) * coil_est
        return self.sense_combine(self.ifft2c(merged), coil_map)

    def zero_filled(self, kspace, mask, coil_map):
        """The endpoint ``x_T`` at inference: the data-consistent adjoint
        ``S^H F^-1 y`` of the acquired k-space."""
        adjoint = self.sense_combine(self.ifft2c(kspace), coil_map)
        return self.data_consistency(adjoint, kspace, mask, coil_map)

    def normalization(self, kspace, coil_map):
        """Per-slice scale: 99th percentile of the zero-filled magnitude.

        Depends only on the undersampled measurements, so training and
        reconstruction compute it identically.
        """
        zf = self.sense_combine(self.ifft2c(kspace), coil_map)
        magnitude = th.sqrt(zf[:, 0] ** 2 + zf[:, 1] ** 2 + 1e-12).flatten(1)
        scale = th.quantile(magnitude, 0.99, dim=1).clamp(min=1e-12)
        return scale.view(-1, 1, 1, 1)

    @staticmethod
    def _variable_density_masks(batch_size, height, width, R, device):
        """Test-time mask generator (``read_data.make_mask``); ``R`` is a
        float or a ``(low, high)`` range drawn per sample."""
        masks = np.empty((batch_size, 1, height, width), dtype=np.float32)
        for i in range(batch_size):
            r = float(np.random.uniform(*R)) if isinstance(R, tuple) else float(R)
            masks[i, 0] = read_data.make_mask(height, width, r)
        return th.from_numpy(masks).to(device)

    # --- FORWARD PROCESS ---

    def state_masks(self, omega, t):
        """Nested support masks ``(m_{t+1}, m_t)`` for transition ``t+1 -> t``.

        ``omega`` is the endpoint support ``[B, 1, H, W]``; ``t`` in
        ``[0, T-1]`` is the index of the transition's destination state.
        """
        rho = th.rand_like(omega)
        nu_src = ((t.float() + 1.0) / self.num_timesteps).view(-1, 1, 1, 1)
        nu_dst = (t.float() / self.num_timesteps).view(-1, 1, 1, 1)
        unacquired = 1.0 - omega
        m_src = omega + unacquired * (rho >= nu_src).float()
        m_dst = omega + unacquired * (rho >= nu_dst).float()
        return m_src, m_dst

    def perturb(self, x, rate):
        """``P_gamma``: drop each k-space coefficient of ``x`` independently
        with probability ``rate`` (Eqs. stochastic_perturb, stochastic_mask).
        ``rate`` is a float or a per-sample ``[B]`` tensor."""
        B, _, H, W = x.shape
        rate = th.as_tensor(rate, device=x.device, dtype=th.float32)
        if bool((rate <= 0).all()):
            return x
        rate = rate.view(-1, 1, 1, 1)
        z = (th.rand((B, 1, H, W), device=x.device) >= rate).float()
        return self.apply_kspace_mask(x, z)

    # --- NETWORK ---

    def p_forward(self, model, x, t, model_kwargs=None):
        return model(x, t, **(model_kwargs or {}))

    def training_losses(self, model, kspace_ri, t, step, save_dir, model_kwargs=None, noise=None):
        """Support-relaxed finite-transition loss for transitions ``t+1 -> t``.

        ``kspace_ri`` is the fully sampled k-space as real-stacked coils;
        ``model_kwargs["coil_map"]`` carries the ESPIRiT maps the same way.
        Both are consumed here rather than forwarded to the network.
        ``t`` is drawn uniformly from ``[0, T-1]`` and is also the network's
        time input, i.e. ``mu_theta(x_{t+1}, t)``.
        """
        model_kwargs = dict(model_kwargs or {})
        coil_map = self.to_complex(model_kwargs.pop("coil_map"))
        kspace_full = self.to_complex(kspace_ri)
        B, _, H, W = kspace_full.shape
        device = kspace_full.device

        norm_mask = self._variable_density_masks(B, H, W, (self.r_min, self.r_max), device)
        scale = self.normalization(kspace_full * norm_mask, coil_map)
        x_0 = self.sense_combine(self.ifft2c(kspace_full / scale), coil_map)

        omega = self._variable_density_masks(B, H, W, self.endpoint_R, device)
        m_src, m_dst = self.state_masks(omega, t)
        x_src = self.apply_kspace_mask(x_0, m_src)
        x_dst = self.apply_kspace_mask(x_0, m_dst)

        # Ablation only (0 by default): perturb the input on the inference
        # schedule, with the endpoint's unacquired fraction as rho_hat.
        rho_hat = 1.0 - 1.0 / self.endpoint_R
        x_in = self.perturb(
            x_src, self.train_perturbation * rho_hat * t.float() / self.num_timesteps
        )
        target = x_dst - x_in

        pred = self.p_forward(model, x_in, t, model_kwargs)
        if self.objective == "restricted":
            # Error evaluated only on the coefficients the transition adds.
            eta = m_dst - m_src
            pred = self.apply_kspace_mask(pred, eta)
            target = self.apply_kspace_mask(target, eta)
        # L1, scaled by 10 to keep gradients in the regime the learning
        # rate was tuned for.
        loss = th.nn.functional.l1_loss(pred, target) * 10.0

        if step % 100 == 0:
            with th.no_grad():
                baseline = target.abs().mean().item() * 10.0
            # A ratio below 1 means the prediction improves on predicting zero.
            logger.log(
                f"Step: {step} | Loss: {loss.item():.4f} | "
                f"loss/zero-pred: {loss.item() / max(baseline, 1e-12):.3f}"
            )

        return {"loss": loss}

    # --- RECOVERY (Algorithm 1) ---

    def recovery_steps(self, model, kspace, mask, coil_map):
        """Run the ``J`` stochastic finite transitions from the endpoint.

        Yields the estimate after each transition, in the units of the
        supplied k-space.
        """
        scale = self.normalization(kspace, coil_map)
        kspace = kspace / scale

        x = self.zero_filled(kspace, mask, coil_map)
        rho_hat = 1.0 - mask.float().flatten(1).mean(1)  # unacquired fraction, [B]
        for j in range(self.num_timesteps, 0, -1):
            t = th.full((x.shape[0],), j - 1, device=x.device, dtype=th.long)
            gamma_j = self.gamma * rho_hat * (j - 1) / self.num_timesteps
            x_tilde = self.perturb(x, gamma_j)
            with th.no_grad():
                x_pred = x_tilde + self.p_forward(model, x_tilde, t)
            x = self.data_consistency(x_pred, kspace, mask, coil_map)
            yield x * scale

    def reconstruct(self, model, kspace, mask, coil_map):
        """Run all ``J`` transitions.

        Returns the SENSE image ``[B, 2, H, W]`` and the RSS image
        ``[B, H, W]``, both in the units of the supplied k-space. RSS is the
        convention of ``reconstruction_rss``, so it is the output to score
        against the M4Raw ground truth.
        """
        image = None
        for image in self.recovery_steps(model, kspace, mask, coil_map):
            pass

        coil_images = self.ifft2c(
            mask * kspace + (1.0 - mask) * self.fft2c(self.as_complex_image(image) * coil_map)
        )
        rss = th.sqrt((th.abs(coil_images) ** 2).sum(dim=1))
        return image, rss
