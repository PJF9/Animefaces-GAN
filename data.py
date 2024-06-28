from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    dataset = 'splcher/animefacedataset'

    download_path = './data'  # Specify your download directory

    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    api.dataset_download_files(dataset, path=download_path)


if __name__ == '__main__':
    main()
