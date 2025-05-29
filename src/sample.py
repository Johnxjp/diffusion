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

        # Generate initial noise
        x = torch.randn(
            batch_size,
            model.in_channels,
            model.image_resolution,
            model.image_resolution,
            device=device,
        )

        for i in tqdm(reversed(range(0, timesteps, step_size)), desc="Sampling"):
            t_batch = torch.tensor([i], device=device)
            noise_level = noise_rates[t_batch].view(-1, 1, 1, 1).to(device)
            signal_level = signal_rates[t_batch].view(-1, 1, 1, 1).to(device)

            noise_variance = ((noise_level) ** 2).reshape([1])
            predicted_noise = model(x, noise_variance)
            pred_images = (x - noise_level * predicted_noise) / signal_level

            # When we're training we're always starting from
            # a normalised image and working backwards.
            pred_images = torch.clamp(pred_images, 0, 1)

            if i > 0:
                next_signal_rate = signal_rates[i - step_size]
                next_noise_rate = noise_rates[i - step_size]
                x = next_signal_rate * pred_images + next_noise_rate * predicted_noise
            else:
                x = pred_images

        # Normalize to [0, 1] range for display
        x = x.clamp(0, 1)
        return x
