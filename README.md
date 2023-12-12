---
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
pretty_name: Beans
dataset_info:
  features:
  - name: image_file_path
    dtype: string
  - name: image
    dtype: image
  - name: labels
    dtype:
      class_label:
        names:
          '0': angular_leaf_spot
          '1': bean_rust
          '2': healthy
  splits:
  - name: train
    num_bytes: 382110
    num_examples: 1034
  - name: validation
    num_bytes: 49711
    num_examples: 133
  - name: test
    num_bytes: 46584
    num_examples: 128
  download_size: 180024906
  dataset_size: 478405
---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Source Data](#source-data)
- [Data Author](#data-author)
## Dataset Description

- **Repository:** [MLOps-essi-upc/MLOps-TeamBeans](https://github.com/MLOps-essi-upc/MLOps-TeamBeans)
- **Paper:** N/A
- **Leaderboard:** N/A
- **Point of Contact:** N/A

### Dataset Summary

Beans leaf dataset with images of diseased and health leaves. Each image is 500 x 500 RGB. Dataset is balanced in terms of classes. There are 3 types of classes, 2 of them being diseased leafs and one being healthy: 
- Angular Leaf Spot which is a bacterial disease caused by Pseudomonas syringae pv.lachrymans
- Bean Rust which is caused by Uromyces phaseoli typica. 
- Healthy

### Supported Tasks and Leaderboards

- `image-classification`: Based on a leaf image, the goal of this task is to predict the disease type (Angular Leaf Spot and Bean Rust), if any.

### Languages

English

## Dataset Structure

### Data Instances

A sample from the training set is provided below:

```
{
    'image_file_path': '/root/.cache/huggingface/datasets/downloads/extracted/0aaa78294d4bf5114f58547e48d91b7826649919505379a167decb629aa92b0a/train/bean_rust/bean_rust_train.109.jpg',
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x16BAA72A4A8>,
    'labels': 1
}
```

### Data Fields

The data instances have the following fields:

- `image_file_path`: a `string` filepath to an image.
- `image`: A `PIL.Image.Image` object containing the image. Note that when accessing the image column: `dataset[0]["image"]` the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the `"image"` column, *i.e.* `dataset[0]["image"]` should **always** be preferred over `dataset["image"][0]`.
- `labels`: an `int` classification label.

Class Label Mappings:

```json
{
  "angular_leaf_spot": 0,
  "bean_rust": 1,
  "healthy": 2,
}
```

### Data Splits

 
|             |train|validation|test|
|-------------|----:|---------:|---:|
|# of examples|1034 |133       |128 |


## Source Data

The data has been sourced from repository at huggingface (https://huggingface.co/datasets/beans)


## Data Author
```
@ONLINE {beansdata,
    author="Makerere AI Lab",
    title="Bean disease dataset",
    month="January",
    year="2020",
    url="https://github.com/AI-Lab-Makerere/ibean/"
}
```

