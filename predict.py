from src import config
from src.models import Generator
from src.utils import load_model, get_device, denorm, configure_logger

import torch
from torchvision.utils import save_image

import os
from PIL import Image


# Get the logger for this module
logger = configure_logger(__name__)

# Get the default device
device = get_device()


def main() -> None:
    # Load the generator model
    generator = load_model(Generator, 
        model_path=os.path.join(config.MODELS_PATH, 'Generator_best.pth'),
        device=device,
        latent_size=config.LATENT_SIZE,
        **config.GENR_PARAMS
    ).to(device)

    logger.info(f'Generator model created succesfully and placed on device: {device}.')

    # Initialize the latent space
    latent = torch.randn(1, config.LATENT_SIZE, 1, 1, device=device)

    # Generate image
    with torch.inference_mode():
        img_tensor = generator(latent).squeeze(dim=0)
        logger.info(f'Generator created image succesfully.')
    
    # Save image
    image_path = os.path.join(config.IMAGES_PATH, 'prediction.jpg')
    save_image(denorm(img_tensor, 0.5, 0.5), image_path, nrow=1)
    logger.info(f'Image saved sucesfully on `{image_path}.`')

    # SHow the image
    image = Image.open(image_path)
    image.show()


if __name__ == '__main__':
    main()
