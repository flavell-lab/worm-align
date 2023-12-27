from julia.api import Julia
from scipy import ndimage
import SimpleITK as sitk
import argparse
import json
import numpy as np
import os

jl = Julia(compiled_modules=False)
jl.eval('include("adjust.jl")')
adjust_image_size = jl.eval("adjust_image_cm")


def write_to_json(input_, output_file):

    with open(f"resources/{output_file}.json", "w") as f:
        json.dump(input_, f, indent=4)

    print(f"{output_file} written under resources.")


def locate_dataset(dataset_name):

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


if __name__ == "__main__":
    """
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

