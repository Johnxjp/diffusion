from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    signal_rates: torch.Tensor,
    noise_rates: torch.Tensor,
    timesteps: int,
    valid_dataloader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 10,
    batch_size: int = 1,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 1,
    early_stopping: bool = False,
):
    # Simple training loop one image
    # Move everything to GPU upfront
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    signal_rates = signal_rates.to(device)
    noise_rates = noise_rates.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 100  # Arbitrary number of batches per epoch

        with tqdm(train_dataloader) as pbar:
            for batch_idx, (batch_images, _) in enumerate(pbar):
                optimizer.zero_grad()

                # Sample timesteps and normalize
                sampled_timesteps = torch.randint(0, timesteps, (batch_size,), device=device)
                normalized_timesteps = sampled_timesteps.float() / (timesteps - 1)

                # Get noise schedules
                timesteps_signal_rate = signal_rates[sampled_timesteps].view(-1, 1, 1, 1)
                timestep_noise_rate = noise_rates[sampled_timesteps].view(-1, 1, 1, 1)

                # Add noise
                noise = torch.randn_like(batch_images, device=device)
                noisy_images = timesteps_signal_rate * batch_images + timestep_noise_rate * noise

                noise_variance = (timestep_noise_rate.squeeze()) ** 2
                # Forward pass
                predicted_noise = model(noisy_images, noise_variance)

                # Loss and backprop
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()

                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)

                if total_norm > 10.0:  # Flag large gradients
                    print(f"WARNING: Large gradient norm: {total_norm:.4f}")

                if torch.isnan(loss):
                    print("NaN loss detected!")
                    break

                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: Average Loss {avg_loss:.5f}")

    torch.save(model.state_dict(), "model_weights.pth")
