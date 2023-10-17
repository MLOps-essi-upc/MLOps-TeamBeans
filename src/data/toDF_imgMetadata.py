from datasets import load_from_disk
import pandas as pd
from PIL import Image
import numpy as np
import os
# Converts raw dataset into df and adds additional columms: image_size_bytes, image_extension, image_width, image_height, red_mean, green_mean, blue_mean
def extract_image_metadata(image_data, image_path):
    image_bytes = image_data['bytes']
    
    with Image.open(image_path) as img:
        img = np.array(img)

    img_array = np.array(img)

    red_mean = np.mean(img_array[:, :, 0])
    green_mean = np.mean(img_array[:, :, 1])
    blue_mean = np.mean(img_array[:, :, 2])

    image_width, image_height = img_array.shape[1], img_array.shape[0]
    
    metadata = {
        'image_size_bytes': len(image_bytes),
        'image_extension': image_path.split('.')[-1],
        'image_width': image_width,
        'image_height': image_height,
        'red_mean': red_mean,
        'green_mean': green_mean,
        'blue_mean': blue_mean,
    }

    return metadata

def add_image_metadata(df):
    df['image_metadata'] = df.apply(lambda row: extract_image_metadata(row['image'], row['image_file_path']), axis=1)
    df = pd.concat([df, df['image_metadata'].apply(pd.Series)], axis=1)
    df = df.drop(['image', 'image_metadata'], axis=1)
    return df



def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    raw_directory = os.path.join(script_dir, "raw")
    dataset = load_from_disk(raw_directory)

    train_data = dataset["train"]

    test_data = dataset["test"]

    validation_data = dataset["validation"]

    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
    validation_df = validation_data.to_pandas()

    train_df = add_image_metadata(train_df)
    test_df = add_image_metadata(test_df)
    validation_df = add_image_metadata(validation_df)


    output_folder = "dataframes"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return train_df, test_df, validation_df

if __name__ == '__main__':
    train_df, test_df, validation_df = main()
