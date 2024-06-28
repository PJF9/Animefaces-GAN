## Dataset Configirations
ZIP_PATH = './data/animefacedataset.zip'
DATASET_DIR = './data'
MODELS_PATH = './checkpoints'
PLOTS_PATH = './plots'
IMAGES_PATH = './generated'

## Training Configirations
BATCH_SIZE = 128
LATENT_SIZE = 128 # Generally the higher the latent size the better
DISCR_PARAMS = dict(
    hidden_1 = 64,
    hidden_2 = 128,
    hidden_3 = 256,
    hidden_4 = 512,
)
GENR_PARAMS = dict(
    hidden_1 = 512,
    hidden_2 = 256,
    hidden_3 = 128,
    hidden_4 = 64,
)
DICR_LR = 0.0002
DICR_BETTAS = (0.5, 0.999)
GENR_LR = 0.0002
GENR_BETTAS = (0.5, 0.999)
EPOCHS = 25
