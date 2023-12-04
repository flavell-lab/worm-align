from julia.api import Julia
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
import os
import yaml
import argparse

jl = Julia(compiled_modules=False)
jl.eval('include("adjust.jl")')
adjust_image_size = jl.eval("adjust_image_cm")


class QuotedStr(str):
    pass


def locate_dataset(dataset_date):

    '''
    Given the date when the dataset was collected, this function locates which
    directory this data file can be found
    '''

    neuropal_dir = '/home/alicia/data_prj_neuropal/data_processed'
    non_neuropal_dir = '/home/alicia/data_prj_kfc/data_processed'
    for directory in os.listdir(neuropal_dir):
        if dataset_date in directory:
            return os.path.join(neuropal_dir, directory)

    for directory in os.listdir(non_neuropal_dir):
        if dataset_date in directory:
            return os.path.join(non_neuropal_dir, directory)

    raise Exception(f'Dataset {dataset_date} cannot be founed.')


def filter_and_crop(image_T, image_median, target_dim):

    filtered_image_CM = get_image_CM(image_T)
    filtered_image_T = filter_image(image_T, image_median)

    return get_cropped_image(
                filtered_image_T,
                filtered_image_CM,
                target_dim, -1).astype(np.float32)


def filter_image(image, threshold):

    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def get_cropped_image(image_T, center, target_dim, projection):

    if projection >= 0:
        return adjust_image_size(
                image_T,
                center,
                target_dim).max(projection)
    elif projection == -1:
        return adjust_image_size(
                image_T,
                center,
                target_dim)


def get_image_T(image_path):

    '''Given the path, read image of .nrrd format as numpy array
    '''

    image_nrrd = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_nrrd)
    if image.ndim == 4:
        image = image.squeeze()
    image_T = np.transpose(image, (2,1,0))

    return image_T


def get_image_CM(image_T):

    '''Taking image of shape in the order (x, y, z) and find its center of mass
    after filtering
    '''
    # subtract the median pixel from the image; zero out the negative pixels
    image_T_wo_background = image_T - np.median(image_T)
    image_T_wo_background[image_T_wo_background < 0] = 0
    x, y, z = ndimage.center_of_mass(image_T_wo_background)

    return (round(x), round(y), round(z))


def quoted_str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


def write_config_file(dataset_directory,
                   train_datasets,
                   valid_datasets,
                   test_datasets):

    yaml.add_representer(QuotedStr, quoted_str_presenter)

    configuration = {
        "dataset": {
            "train": {
                "dir": [],
                "format": QuotedStr("h5"),
                "labeled": False
            },
            "valid": {
                "dir": [],
                "format": QuotedStr("h5"),
                "labeled": False
            },
            "test": {
                "dir": [],
                "format": QuotedStr("h5"),
                "labeled": False
            },
            "type": "paired",
            "moving_image_shape": [290, 120, 64],
            "fixed_image_shape": [290, 120, 64],
        },
        "train": {
            "method": QuotedStr("ddf"),
            "backbone": {
                "name": QuotedStr("local"),
                "num_channel_initial": 16,
                "extract_levels": [0, 1, 2, 3]
            },
            "loss": {
                "image": {
                    "name": QuotedStr("lncc"),
                    "kernel_size": 16,
                    "weight": 1.0
                },
                "regularization": {
                    "weight": 0.2,
                    "name": QuotedStr("nonrigid")
                }
            },
            "preprocess": {
                "data_augmentation": {
                    "name": QuotedStr("affine")
                },
                "batch_size": 4,
                "shuffle_buffer_num_batch": 2,
                "num_parallel_calls": -1
            },
            "optimizer": {
                "name": QuotedStr("Adam"),
                "learning_rate": 1.0e-3
            },
            "epochs": 20000,
            "save_period": 1
        }
    }

    configuration['dataset']['train']['dir'] = [
            QuotedStr(f"{dataset_directory}/train/{train_dataset}")
                for train_dataset in train_datasets
    ]
    configuration['dataset']['valid']['dir'] = [
            QuotedStr(f"{dataset_directory}/valid/{valid_dataset}")
                for valid_dataset in valid_datasets
    ]
    configuration['dataset']['test']['dir'] = [
            QuotedStr(f"{dataset_directory}/test/{test_dataset}")
                for test_dataset in test_datasets
    ]
    with open(f"configs/config_test.yaml", 'w') as f:
        yaml.dump(configuration, f, default_flow_style=None, default_style='')


if __name__ == "__main__":

    dataset_directory = "/home/alicia/data_personal/test_preprocess"
    train_datasets = ["2022-01-09-01", "2022-01-23-04"]
    valid_datasets = ["2022-07-26-01", "2022-07-20-01"]
    test_datasets = ["2022-08-02-01", "2022-04-18-04"]
    image_dimension = [208, 96, 56]
    parser = argparse.ArgumentParser(description="Example script to write a list of integers to a YAML file.")
    parser.add_argument("--shape", nargs="+", type=int, help="List of integers")
    args = parser.parse_args()
    write_config_file(dataset_directory,
                   train_datasets,
                   valid_datasets,
                   test_datasets
    )

