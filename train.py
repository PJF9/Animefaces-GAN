from src import config
from src.models import Discriminator, Generator
from src.trainers import Trainer
from src.utils import unzipping_dataset, get_device, configure_logger, plot_losses

from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder


## Default Device
device = get_device()

# Get the logger for this module
logger = configure_logger(__name__)


def get_dataset() -> ImageFolder:
    '''
    Unzip the dataset, apply transformations and return the ImageFolder object.
    '''
    # Unzip the dataset
    dataset_path = unzipping_dataset(source=config.ZIP_PATH, dest=config.DATASET_DIR, stop_if_exists=True)

    # Define the Transformation of the dataset
    image_transforms = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #  This will ensure that pixel values are in the range (-1, 1),
        #    which is more convenient for training the discriminator.
    ])

    # Create the Dataset object
    dataset = ImageFolder(
        root = dataset_path,
        transform = image_transforms
    )

    return dataset


def main() -> None:
    # Initialize the DataLoader
    dataset = get_dataset()

    # Initialize the models
    discriminator = Discriminator(**config.DISCR_PARAMS).to(device)
    logger.info(f'Discriminator is created and placed on the device: {device.type}')

    generator = Generator(config.LATENT_SIZE, **config.GENR_PARAMS).to(device)
    logger.info(f'Generator is created and placed on the device: {device.type}')

    # Initialize loss function and optimizers
    loss_fn = nn.BCELoss()
    opt_discr = optim.Adam(discriminator.parameters(), lr=config.DICR_LR, betas=config.DICR_BETTAS)
    opt_genr = optim.Adam(generator.parameters(), lr=config.GENR_LR, betas=config.GENR_BETTAS)

    # Initialize the Trainer
    trainer = Trainer(
        discr_model=discriminator,
        genr_model=generator,
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        latent_size=config.LATENT_SIZE,
        loss_fn=loss_fn,
        opt_discr=opt_discr,
        opt_genr=opt_genr,
        device=device
    )

    logger.info('The trainer is created.')

    # Train the Models
    train_res = trainer.fit(
        epochs=config.EPOCHS,
        save_per=config.EPOCHS,
        save_best_genr=True
    )

    logger.info(f'Training Results: {train_res}')

    # Plot the losses and save them
    plot_losses(train_res['discr_losses'], train_res['genr_losses'], save=True)


if __name__ == '__main__':
    main()
