from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import save_checkpoint
from src.diffusion_model import DiffusionUNet


def train(
    model: DiffusionUNet,
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
    patience: int = 5,
    device: str = "cpu",
) -> nn.Module:

    model = model.to(device)
    signal_rates = signal_rates.to(device)
    noise_rates = noise_rates.to(device)

    model.train()
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    for epoch in range(epochs):
        total_loss = 0

        with tqdm(train_dataloader) as pbar:
            for batch_images, _ in pbar:
                optimizer.zero_grad()

                # Sample timesteps and normalize
                sampled_timesteps = torch.randint(0, timesteps, (batch_size,), device=device)

                # Get noise schedules
                timesteps_signal_rate = signal_rates[sampled_timesteps].view(-1, 1, 1, 1)
                timestep_noise_rate = noise_rates[sampled_timesteps].view(-1, 1, 1, 1)

                # Add noise
                noise = torch.randn_like(batch_images, device=device)
                noisy_images = timesteps_signal_rate * batch_images + timestep_noise_rate * noise

                # Change to shape (B,)
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

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Average Loss {avg_loss:.5f}")

        if valid_dataloader is not None:
            with torch.no_grad():
                with tqdm(valid_dataloader) as pbar:
                    val_loss = 0
                    for batch_images, _ in pbar:
                        batch_images = batch_images.to(device)
                        sampled_timesteps = torch.randint(
                            0, timesteps, (batch_size,), device=device
                        )

                        timesteps_signal_rate = signal_rates[sampled_timesteps].view(-1, 1, 1, 1)
                        timestep_noise_rate = noise_rates[sampled_timesteps].view(-1, 1, 1, 1)

                        noise = torch.randn_like(batch_images, device=device)
                        noisy_images = (
                            timesteps_signal_rate * batch_images + timestep_noise_rate * noise
                        )

                        noise_variance = (timestep_noise_rate.squeeze()) ** 2
                        predicted_noise = model(noisy_images, noise_variance)

                        loss = F.mse_loss(predicted_noise, noise)
                        val_loss += loss.item()

                    avg_val_loss = val_loss / len(valid_dataloader)
                    print(f"Validation Loss: {avg_val_loss:.5f}")

            if checkpoint_path and (epoch + 1) % checkpoint_interval == 0:
                print(f"Saving checkpoint for epoch {epoch + 1}...")
                save_checkpoint(
                    checkpoint_path,
                    model,
                    model.config(),
                    optimizer,
                    epoch,
                    train_loss=avg_loss,
                    val_loss=avg_val_loss,
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_loss_epoch = epoch
                print(f"New best validation loss: {best_val_loss:.5f}")
                if checkpoint_path:
                    print(f"Saving best model to {checkpoint_path}...")
                    torch.save(model.state_dict(), checkpoint_path)

            if (
                early_stopping
                and avg_val_loss > best_val_loss
                and (epoch - best_val_loss_epoch) >= patience
            ):
                print(
                    f"Early stopping at epoch {epoch + 1}. Best validation loss was at epoch {best_val_loss_epoch + 1}."
                )
                print("Early stopping triggered. Stopping training.")
                break

    return model
