# -*- coding: utf-8 -*-
import logging
import os
import torch
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datasets import Dataset as DatasetMain
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    """
    This is a class that extends the Dataset class from pytorch. It's used to make sure that the 
    input for the model is properly initialized. This means resizing it to be 500 x 500, initialising it
    as a tensor and normalizing the RGB values. 
    """
    def __init__(self, dataset):
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.Resize((500,500)),  # Resize to our desired size
            transforms.ToTensor(),          # Convert PIL Image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB channels
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self.transform(sample['image'])
        label = sample['labels']

        return image, label
    


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
def main():
    """ 
    Runs data processing scripts to turn raw data from (../raw) into 
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # if there is preprocessing it will go here

    logger.info('creating dataloaders from the preprocessed data')
    
    current_directory = os.getcwd()

    dataset_train = DatasetMain.from_file(os.path.join(current_directory, "../../data/raw/train/") + "data-00000-of-00001.arrow")

    dataset_validation = DatasetMain.from_file(os.path.join(current_directory, "../../data/raw/validation/") + "data-00000-of-00001.arrow")

    dataset_test = DatasetMain.from_file(os.path.join(current_directory, "../../data/raw/test/") + "data-00000-of-00001.arrow")

    custom_train = CustomDataset(dataset_train)
    custom_validation = CustomDataset(dataset_validation)
    custom_test = CustomDataset(dataset_test)

    # Create a DataLoader for training, validation and test
    train_loader = DataLoader(custom_train, batch_size=32, shuffle=True)    
    validation_loader = DataLoader(custom_validation, batch_size=32, shuffle=False)
    test_loader = DataLoader(custom_test, batch_size=32, shuffle=False)
    
    # Create the directory if it doesn't exist
    if not os.path.exists('../../data/processed'):
        os.makedirs('../../data/processed')

    # Save the DataLoader to a file
    torch.save(train_loader, '../../data/processed/train_loader.pt')
    torch.save(validation_loader, '../../data/processed/validation_loader.pt')
    torch.save(test_loader, '../../data/processed/test_loader.pt')
    
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
