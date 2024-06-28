from src import config
from src.models import Discriminator, Generator
from src.utils import configure_logger, save_model, save_samples

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from multiprocessing import cpu_count

from tqdm import tqdm
from timeit import default_timer as timer
from typing import Tuple, Dict, List, Any


class Trainer:
    '''
    A class to handle the training of a PyTorch GAN modeling.
    '''

    # Initialize the logger as a class attribute
    logger = configure_logger(__name__)

    def __init__(self,
            discr_model: Discriminator,
            genr_model: Generator,
            dataset: ImageFolder,
            batch_size: int,
            latent_size: int,
            loss_fn: nn.Module,
            opt_discr: optim,
            opt_genr: optim,
            train_prop: float=0.8,
            device = torch.device('cpu')
        ) -> None:
        '''
        Initialize the Trainer with given models, dataset, and configurations.

        Args:
            discr_model (Discriminator): The discriminator model.
            genr_model (Generator): The generator model.
            dataset (ImageFolder): The dataset for training.
            batch_size (int): The size of the batch.
            latent_size (int): The size of the latent vector.
            loss_fn (nn.Module): The loss function.
            opt_discr (optim): The optimizer for the discriminator.
            opt_genr (optim): The optimizer for the generator.
            device (torch.device): The device to run the training on.
        '''
        self.discr_model = discr_model
        self.genr_model = genr_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.loss_fn = loss_fn
        self.opt_discr = opt_discr
        self.opt_genr = opt_genr
        self.train_prop = train_prop
        self.device = device

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        '''
        Creates a training and validation DataLoaders

        Returns:
            DataLoader: DataLoader for the dataset.
        '''
        train_ds, valid_ds = random_split(self.dataset, [self.train_prop, 1 - self.train_prop])

        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True, drop_last=True)
        valid_dl = DataLoader(valid_ds, self.batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True, drop_last=True)

        return train_dl, valid_dl

    def _discr_step(self, x_batch: torch.Tensor) -> Tuple[float, float, float]:
        '''
        Perform a single training step for the discriminator.

        Args:
            x_batch (torch.Tensor): A batch of real images.

        Returns:
            Tuple[float, float, float]: The discriminator loss, score on real images, and score on fake images.
        '''
        ### 1. Getting Discriminator's Loss on Real Images ###
        logits_on_real = self.discr_model(x_batch) # Pass real images through discriminator
        labels_on_real = torch.ones(self.batch_size, 1, device=self.device)

        loss_on_real = self.loss_fn(logits_on_real, labels_on_real)
        score_on_real = torch.mean(logits_on_real).item() # Average Score per Batch (As close to 1 as possible)

        ### 2. Getting Discriminator's Loss on Generated Images ###
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        generated_images = self.genr_model(latent.to(self.device))

        logits_on_fake = self.discr_model(generated_images)
        labels_on_fake = torch.zeros(generated_images.shape[0], 1, device=self.device) # shape: (`batch_size`, 1)

        loss_on_fake = self.loss_fn(logits_on_fake, labels_on_fake)
        score_on_fake = torch.mean(logits_on_fake).item() # (As close to 0 as possible)

        ### 3. Updating Discriminator's Weights ###
        loss = loss_on_real + loss_on_fake

        if self.discr_model.training:
            self.opt_discr.zero_grad()
            loss.backward()
            self.opt_discr.step()

        return loss.item(), score_on_real, score_on_fake

    def _genr_step(self) -> float:
        '''
        Perform a single training step for the generator.

        Returns:
            float: The generator loss.
        '''
        ### 1. Generating Fake Images ###
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.genr_model(latent.to(self.device)) # Generate fake images

        ### 2. Passing the Fake Images into the Dicriminator (Trying to Fool it) ###
        logits = self.discr_model(fake_images.to(self.device))
        labels = torch.ones(self.batch_size, 1, device=self.device)

        loss = self.loss_fn(logits, labels)

        ### 3. Updating Discriminator's Weights ###
        if self.genr_model.training:
            self.opt_genr.zero_grad()
            self.opt_discr.zero_grad()
            loss.backward()
            self.opt_genr.step()

        return loss.item()
    
    def _process_loader(self, dl: DataLoader) -> Tuple[float, float, float, float]:
        '''
        Process a DataLoader for training or validation.

        This method processes a DataLoader by passing each batch through the discriminator and generator
        models, accumulating the losses and scores, and returning the average values across the entire DataLoader.

        Args:
            dl (DataLoader): The DataLoader to process.

        Returns:
            Tuple[float, float, float, float]: A tuple containing:
                - Average discriminator loss (float)
                - Average generator loss (float)
                - Average score on real images (float)
                - Average score on fake images (float)
        '''
        total_discr_loss, total_genr_loss = 0.0, 0.0
        total_score_on_real, total_score_on_fake = 0.0, 0.0

        phase = 'Training Step' if self.genr_model.training else 'Validation Step'

        for x_batch, _ in tqdm(dl, ascii=True, desc=' '*12 + phase):
            x_batch = x_batch.to(self.device, non_blocking=True)

            # Passing a batch through the Discriminator
            _discr_loss, _score_on_real, _score_on_fake = self._discr_step(x_batch)
            total_discr_loss += _discr_loss
            total_score_on_real += _score_on_real
            total_score_on_fake += _score_on_fake

            # Passing a batch through the Generator
            total_genr_loss += self._genr_step()
            
        # Calculate averages
        total_discr_loss /= len(dl)
        total_genr_loss /= len(dl)
        total_score_on_real /= len(dl)
        total_score_on_fake /= len(dl)
        
        return total_discr_loss, total_genr_loss, total_score_on_real, total_score_on_fake
    
    def _training_step(self, train_dl: DataLoader) -> Dict[str, List[float]]:
        '''
        Perform a single training step.

        This method sets the discriminator and generator models to training mode, processes the training DataLoader,
        and then sets the models back to evaluation mode. It returns a dictionary containing the losses and scores
        from the training step.

        Args:
            train_dl (DataLoader): The training DataLoader to process.

        Returns:
            Dict[str, List[float]]: A dictionary containing:
                - 'discr_losses' (float): Average discriminator loss over the training DataLoader.
                - 'genr_losses' (float): Average generator loss over the training DataLoader.
                - 'scores_on_real' (float): Average score on real images over the training DataLoader.
                - 'scores_on_fake' (float): Average score on fake images over the training DataLoader.
        '''
        # Set models to training mode
        self.discr_model.train()
        self.genr_model.train()

        # Process the training DataLoader
        discr_loss, genr_loss, score_on_real, score_on_fake = self._process_loader(train_dl)

        # Set models back to evaluation mode
        self.discr_model.eval()
        self.genr_model.eval()

        return {
            'discr_losses': discr_loss,
            'genr_losses': genr_loss,
            'scores_on_real': score_on_real,
            'scores_on_fake': score_on_fake
        }

    def _validation_step(self, valid_dl: DataLoader) -> Dict[str, List[float]]:
        '''
        Perform a single validation step.

        This method sets the discriminator and generator models to evaluation mode, processes the validation DataLoader
        without updating the model weights, and returns a dictionary containing the losses and scores from the validation step.

        Args:
            valid_dl (DataLoader): The validation DataLoader to process.

        Returns:
            Dict[str, List[float]]: A dictionary containing:
                - 'discr_losses' (float): Average discriminator loss over the validation DataLoader.
                - 'genr_losses' (float): Average generator loss over the validation DataLoader.
                - 'scores_on_real' (float): Average score on real images over the validation DataLoader.
                - 'scores_on_fake' (float): Average score on fake images over the validation DataLoader.
        '''
        # Set models to evaluation mode
        self.discr_model.eval()
        self.genr_model.eval()

        # Process the validation DataLoader without updating model weights
        with torch.inference_mode():
            discr_loss, genr_loss, score_on_real, score_on_fake = self._process_loader(valid_dl)

        return {
            'discr_losses': discr_loss,
            'genr_losses': genr_loss,
            'scores_on_real': score_on_real,
            'scores_on_fake': score_on_fake
        }

    def fit(self,
            epochs: int,
            save_per: int,
            save_best_genr: bool=False
        ) -> Dict[str, Any]:
        '''
        Train the GAN models for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            save_per (int): Save models after every 'save_per' epochs.
            save_best_genr (bool): Save the best generator model based on loss.

        Returns:
            dict: A dictionary with training statistics:
                - train_discr_losses (list of float): List of training discriminator losses per epoch.
                - train_genr_losses (list of float): List of training generator losses per epoch.
                - train_scores_on_real (list of float): List of average training discriminator scores on real images per epoch.
                - train_scores_on_fake (list of float): List of average training discriminator scores on fake images per epoch.
                - valid_discr_losses (list of float): List of validation discriminator losses per epoch.
                - valid_genr_losses (list of float): List of validation generator losses per epoch.
                - valid_scores_on_real (list of float): List of average validation discriminator scores on real images per epoch.
                - valid_scores_on_fake (list of float): List of average validation discriminator scores on fake images per epoch.
                - loss_fn (str): Name of the loss function used.
                - discr_opt (str): Name of the optimizer used for the discriminator.
                - genr_opt (str): Name of the optimizer used for the generator.
                - device (str): Device type used for training (e.g., 'cpu' or 'cuda').
                - epochs (int): Total number of epochs trained.
                - total_time (float): Total training time in seconds.
        '''
        start_time = timer()
        train_discr_losses, train_genr_losses, train_scores_on_real, train_scores_on_fake = [], [], [], []
        valid_discr_losses, valid_genr_losses, valid_scores_on_real, valid_scores_on_fake = [], [], [], []

        Trainer.logger.info('Start Training Process.')

        # Initialize the training DataLoader
        train_dl, valid_dl = self._get_loaders()
        Trainer.logger.info('Training Dataloader created succesfully.')

        best_genr_loss = float('inf')

        for epoch in range(1, epochs + 1):
            Trainer.logger.info(f'-> Epoch: {epoch}/{epochs}')

            _train_res = self._training_step(train_dl)
            _valid_res = self._validation_step(valid_dl)

            # Log the results
            Trainer.logger.info('')
            Trainer.logger.info(f'Results')
            Trainer.logger.info(f'Discr Loss:     (train: {_train_res["discr_losses"]:.4f} | valid: {_valid_res["discr_losses"]:.4f})')
            Trainer.logger.info(f'Genr Loss:      (train: {_train_res["genr_losses"]:.4f} | valid: {_valid_res["genr_losses"]:.4f})')
            Trainer.logger.info(f'Scores on Real: (train: {_train_res["scores_on_real"]:.4f} | valid: {_valid_res["scores_on_real"]:.4f})')
            Trainer.logger.info(f'Scores on Fake: (train: {_train_res["scores_on_fake"]:.4f} | valid: {_valid_res["scores_on_fake"]:.4f})')

            train_discr_losses.append(_train_res['discr_losses'])
            train_genr_losses.append(_train_res['genr_losses'])
            train_scores_on_real.append(_train_res['scores_on_real'])
            train_scores_on_fake.append(_train_res['scores_on_fake'])

            valid_discr_losses.append(_valid_res['discr_losses'])
            valid_genr_losses.append(_valid_res['genr_losses'])
            valid_scores_on_real.append(_valid_res['scores_on_real'])
            valid_scores_on_fake.append(_valid_res['scores_on_fake'])

            # Save the best generator model based on loss
            if save_best_genr and (_valid_res['genr_losses'] < best_genr_loss):
                best_genr_loss = _valid_res['genr_losses']
                save_model(self.genr_model, f'{config.MODELS_PATH}/{self.genr_model.__class__.__name__}_best.pth')

            # Saving the models and the predictions of the generator
            if save_per and (epoch % save_per == 0):
                save_model(self.discr_model, f'{config.MODELS_PATH}/{self.discr_model.__class__.__name__}_checkpoint_{epoch:0=4d}.pth')
                save_model(self.genr_model, f'{config.MODELS_PATH}/{self.genr_model.__class__.__name__}_checkpoint_{epoch:0=4d}.pth')
                save_samples(self.genr_model, device=self.device)

            Trainer.logger.info(('-' * 100))

        Trainer.logger.info('Training Process Completed Successfully.')

        # After training, clear the CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'train_discr_losses': train_discr_losses,
            'train_genr_losses': train_genr_losses,
            'train_scores_on_real': train_scores_on_real,
            'train_scores_on_fake': train_scores_on_fake,
            'valid_discr_losses': valid_discr_losses,
            'valid_genr_losses': valid_genr_losses,
            'valid_scores_on_real': valid_scores_on_real,
            'valid_scores_on_fake': valid_scores_on_fake,
            'loss_fn': self.loss_fn.__class__.__name__,
            'discr_opt': self.opt_discr.__class__.__name__,
            'genr_opt': self.opt_genr.__class__.__name__,
            'device': self.device.type,
            'epochs': epochs,
            'total_time': timer() - start_time,
        }
