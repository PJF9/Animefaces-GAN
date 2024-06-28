from src import config
from src.utils.data import denorm
from src.utils.log import configure_logger

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import os
from typing import List, Tuple, Optional

# Get the logger for this module
logger = configure_logger(__name__)


def show_grid(images_tensor: torch.Tensor, nmax: int=64) -> None:
    '''
    Visuzalize the given tensor images in a grid.

    Args:
        images_tensor (torch.Tensor): The tensor containing the images to be visualize
        nmax (int): The maximum number of those images to be visualized 
    '''
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images_tensor.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def plot_losses(
        discr_loss: List[float],
        genr_loss: List[float],
        fig_size: Tuple[int, int] = (6, 4),
        font_size: int = 11,
        save: bool=False
    ) -> None:

    plt.figure(figsize=fig_size)

    plt.plot(range(len(discr_loss)), discr_loss, label="Discriminator Loss")
    plt.plot(range(len(genr_loss)), genr_loss, label="Generator Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves', fontsize=14)
    plt.legend(fontsize=font_size)

    if save:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(config.PLOTS_PATH), exist_ok=True)
        plt.savefig(os.path.join(config.PLOTS_PATH, 'losses.jpg'))
        logger.info(f"Plot saved to `{config.PLOTS_PATH}`.")

    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.show()
