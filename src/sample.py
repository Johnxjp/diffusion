import torch
from tqdm import tqdm

from src.diffusion_model import DiffusionUNet


def ddim_sampler(
    model: DiffusionUNet,
    noise_rates: torch.Tensor,
    signal_rates: torch.Tensor,
    timesteps: int,
    step_size: int = 1,
    batch_size: int = 1,
    device: str = "cpu",
) -> torch.Tensor:

    model.eval()
    with torch.no_grad():
        x = torch.randn(
            batch_size,
            model.in_channels,
            model.image_resolution,
            model.image_resolution,
            device=device,
        )
        for i in tqdm(reversed(range(0, timesteps, step_size)), desc="Sampling"):
            t_batch = torch.tensor([i], device=device)
            noise_level = noise_rates[t_batch].repeat(batch_size).view(-1, 1, 1, 1).to(device)
            signal_level = signal_rates[t_batch].repeat(batch_size).view(-1, 1, 1, 1).to(device)
            noise_variance = (noise_level) ** 2
            predicted_noise = model(x, noise_variance.squeeze())
            pred_images = (x - noise_level * predicted_noise) / signal_level

            # When we're training we're always starting from
            # a normalised image and working backwards.
            pred_images = torch.clamp(pred_images, 0, 1)

            if i > 0:
                next_signal_rate = signal_rates[i - step_size]
                next_noise_rate = noise_rates[i - step_size]
                x = next_signal_rate * pred_images + next_noise_rate * predicted_noise
            else:
                print("Final step, no noise added")
                x = pred_images

        # Normalize to [0, 1] range for display
        x = x.clamp(0, 1)

    return x


def ddpm_sampler(
    model: DiffusionUNet,
    signal_rates: torch.Tensor,
    noise_rates: torch.Tensor,
    timesteps: int,
    device: str,
    batch_size: int = 1,
) -> torch.Tensor:
    """DDPM sampling with variance schedule"""
    model.eval()

    with torch.no_grad():
        x = torch.randn(
            batch_size,
            model.in_channels,
            model.image_resolution,
            model.image_resolution,
            device=device,
        )

        for t in tqdm(reversed(range(timesteps)), desc="DDPM Sampling"):
            # Predict noise
            alpha_bar_t = signal_rates[t] ** 2
            alpha_bar_prev = signal_rates[t - 1] ** 2 if t > 0 else torch.tensor(1.0, device=device)
            alpha_t = alpha_bar_t / alpha_bar_prev if t > 0 else alpha_bar_t
            noise_variance = (noise_rates[t] ** 2).reshape(batch_size)
            predicted_noise = model(x, noise_variance)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, 0, 1)

            if t > 0:
                # DDPM reverse process
                mu = (
                    torch.sqrt(alpha_t) * x
                    + (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
                )
                mu = (1 / torch.sqrt(alpha_t)) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
                )

                # Add noise
                sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_t))
                noise = torch.randn_like(x)
                x = mu + sigma * noise
            else:
                x = pred_x0

    return x.clamp(0, 1)


