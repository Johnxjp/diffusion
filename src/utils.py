from PIL import Image
import torch
import numpy as np


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_config": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "image_resolution": 64,
            "num_res_blocks": 2,
            "channel_multipliers": [1, 2, 4, 8],
            "attention_resolutions": [16],
            "dropout": 0.1,
            "time_emb_dim": 128,
        },
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.

    Parameters:
        tensor (torch.Tensor): The input tensor. Can be:
            - 3D tensor of shape [C, H, W] (C=3 for RGB, C=1 for grayscale)
            - 4D tensor of shape [1, C, H, W] (with batch dimension)

    Returns:
        PIL.Image: The converted PIL Image
    """
    # If tensor has 4 dimensions with batch size 1, remove batch dimension
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Ensure tensor is on CPU and detached from computation graph
    tensor = tensor.cpu().detach()

    # Clamp values to be between 0 and 1
    tensor = torch.clamp(tensor, 0, 1)

    # For 3-channel RGB image
    if tensor.size(0) == 3:
        # Convert from [C, H, W] to [H, W, C] format and then to numpy array
        tensor = tensor.permute(1, 2, 0).numpy()

        # Convert from float values in range [0, 1] to uint8 in range [0, 255]
        img_array = (tensor * 255).astype(np.uint8)

        # Create PIL image
        pil_image = Image.fromarray(img_array)

    # For single-channel grayscale image
    elif tensor.size(0) == 1:
        # Remove channel dimension
        tensor = tensor.squeeze(0).numpy()

        # Convert from float values in range [0, 1] to uint8 in range [0, 255]
        img_array = (tensor * 255).astype(np.uint8)

        # Create PIL image
        pil_image = Image.fromarray(img_array, mode="L")

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected 3 or 1 channels.")

    return pil_image
