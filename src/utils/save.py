from src import config
from src.utils.log import configure_logger
from src.utils.data import denorm

import torch
from torch import nn
from torchvision.utils import save_image

import os
from pathlib import Path


# Get the logger for this module
logger = configure_logger(__name__)


def save_model(model: nn.Module, path: str, stops=False) -> None:
    '''
    Save a PyTorch model to a specified path.

    Args:
        model (torch.Module): The PyTorch model to be saved.
        path (str): The path where the model will be saved.
        stops (bool, optional): If True, stops the function execution if the model file already exists at the given path. Defaults to False.

    Raises:
        AssertionError: If the file extension of the specified path is not `.pt` or `.pth`.

    Returns:
        None
    '''
    target_path = Path('/'.join(path.split('/')[:-1]))
    model_name = path.split('/')[-1]

    if not (model_name.endswith('.pth') or model_name.endswith('.pt')):
        logger.error('Wrong extension: Expecting `.pt` or `.pth`.')
        return
    
    # Creating the directory that the model is going to be saved if not exists
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    # If path already exists
    if Path(path).is_file():
        logger.info(f'Model `{model_name}` already exists on `{target_path}`.')
        if stops:
            return
        logger.warning(f'Deleting `{path}`.')
        os.remove(path)

    # Saving the Model to the given path
    logger.info(f'Saving Model `{model_name}` to `{target_path}`.')
    torch.save(obj=model.state_dict(), f=path)

    logger.info(f'Model Successfully Saved to `{path}`.')


def load_model(model_class: nn.Module, model_path: str, device: torch.device=torch.device('cpu'), **kargs) -> nn.Module:
    '''
    Loads a PyTorch model from a specified file.
    
    Parameters:
        model_path (str): Path to the saved model file (e.g., 'model.pth').
        model_class (nn.Module): The class of the model to be loaded.
        device (torch.device): The device that the model will be load on. Default is CPU.
        **kwargs: Additional arguments required to initialize the model class.

    Returns:
        The loaded model.
    '''
    # Initialize the model
    model = model_class(**kargs)

    # Load the state dict (parameters)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    
    # Load the parameters into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()

    logger.info('Model succesfully loaded.')
    
    return model


def save_samples(
        generator: nn.Module,
        device: torch.device = torch.device('cpu')
    ) -> None:
    '''
    Generate and save samples from the generator model.

    Args:
        generator (nn.Module): The generator model to use for generating images.
        device (torch.device, optional): The device to run the generator on. Defaults to CPU.

    Returns:
        None

    This function generates a batch of images using the provided generator model and saves them as a grid image.
    The generated images are denormalized before saving. The images are saved in the directory specified by `config.IMAGES_PATH`.
    The filename of the saved image includes an index that is incremented with each call to the function.
    '''
    latent = torch.randn(config.BATCH_SIZE, config.LATENT_SIZE, 1, 1, device=device)

    generator.eval()

    with torch.inference_mode():
        # Generating Images
        generated_images = generator(latent.to(device))
        logger.info('Images generated succesfully.')

    image_grid_name = f'generated-images-{save_samples.index:0=4d}.jpg'

    # Create the directory if it doesn't exists
    os.makedirs(config.IMAGES_PATH, exist_ok=True)

    save_image(denorm(generated_images, 0.5, 0.5), os.path.join(config.IMAGES_PATH, image_grid_name), nrow=8)
    logger.info(f'Image succesfully saved to {os.path.join(config.IMAGES_PATH, image_grid_name)}.')

    # Increament the index of the images
    save_samples.index += 1

# Initialze the index of the images (static variable)
save_samples.index = 0
