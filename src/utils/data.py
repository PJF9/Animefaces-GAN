from src.utils.log import configure_logger

import torch

from pathlib import Path
from shutil import rmtree
from os import remove
import zipfile


# Get the logger for this module
logger = configure_logger(__name__)


def unzipping_dataset(
        source: str,
        dest: str = './',
        stop_if_exists: bool=False,
        del_zip: bool=False
    ) -> Path:
    '''
    Unzip a dataset from a given source zip file to a destination directory.

    :param source: Path to the source zip file.
    :param dest: Path to the destination directory. Default is the current directory.
    :param stop_if_exists: If True, stop_if_exists the function if the destination directory already exists. Default is False.
    :param del_zip: If True, deletes the source zip file after extraction. Default is False.

    :return: Path to the destination directory.
    '''

    zip_path = Path(source)
    file_name = zip_path.stem
    dest_path = Path(dest).joinpath(file_name)

    if dest_path.is_dir():
        logger.info(f'Directory `{dest_path}` already exists.')
        if stop_if_exists:
            return dest_path
        rmtree(dest_path)
        logger.info(f'Directory `{dest_path}` deleted succesfully.')

    # Creating the dataset directory
    dest_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'Directory `{dest_path}` created succesfully.')

    # Unziping the Dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        logger.info(f'Unzipping file `{zip_path}` to `{dest_path}`.')
        zip_ref.extractall(dest_path)
    
    # Deleting the zip file
    if del_zip:
        remove(zip_path)
        logger.info(f'File `{zip_path}` deleted succesfully.')

    logger.info(f'Dataset extracted succesfully to `{dest_path}`.')

    return dest_path


def denorm(img_tesnor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    '''
    Denormalize the image tensor.

    :param img_tensor: The normalized tensor that will be scaled.
    :param mean: The mean of the normalized tensor.
    :param std: The std of the normalized tensor.

    :return: The original tensor (before the transformations)
    '''
    return (img_tesnor * std) + mean # Back to [0, 1]
