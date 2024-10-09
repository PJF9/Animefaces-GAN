# Animefaces-GAN
This project aims to generate anime faces using Generative Adversarial Networks (GANs). We have implemented and trained a generator and a discriminator model to achieve this task.


## Installation
1. Clone the repository:
```bash
git clone https://github.com/PJF9/Animefaces-GAN.git)
cd AnimeFaces-GAN
```

2. Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
1. Add `kaggle.json` file to /home/usr/.kaggle (you can generate a key [here](https://www.kaggle.com/settings).)

2. Modify `src/config.py` as needed to adjust the settings as you prefer.


## Usage
1. To downlaod the dataset, run:
```bash
python3 data.py
```

2. To train the GAN models, run:
```bash
python3 train.py
```

3. To make a single prediction, run:
```bash
python3 predict.py
```


## Modules
### Models
* `generator.py`: Defines the generator model architecture.
* `discriminator.py`: Defines the discriminator model architecture.

### Trainers
* `trainer.py`: Contains the training loop for the GAN models.

### Utils
* `data.py`: Handles data loading and preprocessing.
* `device.py`: Manages device configuration (CPU/GPU).
* `log.py`: Contains logging functionalities.
* `save.py`: Functions for saving models and outputs
* `visualization.py`: Functions for visualizing generated images.
* `config.py`: Handles configuration file parsing.

### Scripts
* `data.py`: Script to preprocess the dataset.
* `train.py`: Script to start training the GAN models.
* `predict.py`: Script to generate images using the trained generator model.


## Additional Notes
While running the scripts, some extra directories will be created:
* `./checkpoints` (**config.MODELS_PATH**): This directory will save all the checkpoints of training and the best generator model.
* `./plots` (**config.PLOTS_PATH**): This directory will save the loss curves after training.
* `./generated` (**config.IMAGES_PATH**): This directory will save the generated images during training and the results from the **predict.py** script.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.
