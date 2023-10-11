from datasets import load_dataset
import os

#i had to install datasets, chardet, charset_normalizer, pyyaml

def main():
    dataset = load_dataset("beans")

    current_directory = os.getcwd()

    raw_directory = os.path.join(current_directory, "raw")

    # Save the dataset to the specified local directory
    dataset.save_to_disk(raw_directory)

if __name__ == '__main__':
    main()