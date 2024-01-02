from julia.api import Julia
from numpy.typing import NDArray
from scipy import ndimage
from typing import Any, Dict, Tuple
import SimpleITK as sitk
import argparse
import json
import numpy as np
import os


jl = Julia(compiled_modules=False)
jl.eval('include("adjust.jl")')
ADJUST_IMAGE_SIZE = jl.eval("adjust_image_cm")

def write_to_json(input_: Dict[str, Any], output_file: str):

    """
    Write dictionary to .JSON file

    :param input_: dictionaty to be written
    :param output_file: .JSON file name
    """
    with open(f"resources/{output_file}.json", "w") as f:
        json.dump(input_, f, indent=4)

    print(f"{output_file} written under resources.")


def locate_dataset(dataset_name: str):

    """
    Given the date when the dataset was collected, this function locates which
    directory this data file can be found

    :param dataset_name: name of the dataset; e.g. `2022-03-16-02`
    """
    neuropal_dir = '/home/alicia/data_prj_neuropal/data_processed'
    non_neuropal_dir = '/home/alicia/data_prj_kfc/data_processed'
    others_dir = '/data3/prj_rim/data_processed'

    if dataset_name in ["2023-08-07-01", "2023-08-25-02", "2023-08-07-16"]:
        return f"{others_dir}/{dataset_name}_output"
        """candidates = os.listdir(others_dir)
        return os.path.join(
                others_dir,
                list(filter(lambda d: dataset_name in d, candidates))[0]
        )"""

    for directory in os.listdir(non_neuropal_dir):
        if dataset_name in directory:
            return os.path.join(non_neuropal_dir, directory)

    for directory in os.listdir(neuropal_dir):
        if dataset_name in directory:
            return os.path.join(neuropal_dir, directory)

    raise Exception(f'Dataset {dataset_name} cannot be founed.')


def filter_and_crop(
        image_T: NDArray[np.float32],
        image_median: float,
        target_image_shape: Tuple[int, int, int]
    ) -> NDArray[np.float32]:
    """
    Subtract the median pixel value regarded as the image background from the
    image and resize it to the target shape.

    :param image_T: image of shape (x_dim, y_dim, z_dim)
    :param image_median: the median pixel value of the image
    :param target_image_shape: the target image dimension; given in the order
        of (x_dim, y_dim, z_dim)

    :return image of the target shape with background subtracted
    """
    filtered_image_CM = get_image_CM(image_T)
    filtered_image_T = filter_image(image_T, image_median)

    return get_cropped_image(
                filtered_image_T,
                filtered_image_CM,
                target_image_shape, -1).astype(np.float32)


def filter_image(
        image: NDArray[np.float32],
        threshold: float
    ) -> NDArray[np.float32]:
    """
    Subtract the threshold value from each image pixel and set the resultant
    negative pixels to zero.

    :param image: input image of shape (x_dim, y_dim, z_dim)
    :param threshold: value to be subtracted from each image pixel

    :return thresholded image
    """
    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def get_cropped_image(
        image_T: NDArray[np.float32],
        center: Tuple[int, int, int],
        target_image_shape: Tuple[int, int, int],
        projection: int
    ) -> NDArray[np.float32]:
    """
    Resize image to the target image (that could also be projected along the
    maximum value of the given axis).

    :param image_T: input image of shape (x_dim, y_dim, z_dim)
    :param center: center of mass of the image
    :param target_image_shape: the target image dimension; given in the order
        of (x_dim, y_dim, z_dim)

    :param projection: axis to perform maximum projection; options are 0, 1, 2;
        if no projection then set to -1

    :return resized image
    """
    if projection in [0, 1, 2]:
        return ADJUST_IMAGE_SIZE(
                image_T,
                center,
                target_image_shape).max(projection)
    elif projection == -1:
        return ADJUST_IMAGE_SIZE(
                image_T,
                center,
                target_image_shape)
    else:
        raise Exception(f"projection is not 0, 1, 2, but {projection}")


def get_image_T(image_path: str) -> NDArray[np.float32]:
    """
    Read .NRRD image from path.

    :param image_path: path of the image

    :return image as numpy array
    """
    image_nrrd = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_nrrd)
    if image.ndim == 4:
        image = image.squeeze()
    image_T = np.transpose(image, (2,1,0))

    return image_T


def get_image_CM(image_T: NDArray[np.float32]) -> Tuple[int, int, int]:
    """
    Find the center of mass of image.

    :param image_T: image array

    :return center of mass of the image
    """
    # subtract the median pixel from the image; zero out the negative pixels
    image_T_wo_background = image_T - np.median(image_T)
    image_T_wo_background[image_T_wo_background < 0] = 0
    x, y, z = ndimage.center_of_mass(image_T_wo_background)

    return (round(x), round(y), round(z))

