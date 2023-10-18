import os
from datasets import load_dataset


def main():
    """
    This is a function that loads the dataset from online and puts it in a local "raw" directory
    """
    dataset = load_dataset("beans")

    current_directory = os.getcwd()

    raw_directory = os.path.join(current_directory, "raw")

    # Save the dataset to the specified local directory
    dataset.save_to_disk(raw_directory)

if __name__ == '__main__':
    main()