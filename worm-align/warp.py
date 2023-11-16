from utils import locate_dataset, get_cropped_image
from numpy.typing import NDArray
import deepreg.model.layer as layer
import json
import nibabel as nib
import numpy as np
import tensorflow as tf


with open("resources/registration_problems.json", "r") as f:
    ALL_PROBLEMS = json.load(f)

with open("resources/center_of_mass.json", "r") as f:
    CM_DICT = json.load(f)

with open("resources/euler_parameters.json", "r") as f:
    EULER_PARAMETERS_DICT = json.load(f)


def warp_channel_image(ddf_directory: str,
                       dataset_name: str,
                       registration_problem: str) -> NDArray[np.float_]:

    pair_num = get_pair_num(dataset_name, registration_problem)
    network_outputs_path = f"{ddf_directory}/test/pair_{pair_num}"
    ddf_nii_path = f"{network_outputs_path}/ddf.nii.gz"
    ddf_array = nib.load(ddf_nii_path).get_fdata()
    warped_moving_image_nii_path = \
            f"{network_outputs_path}/pred_fixed_image.nii.gz"
    warped_moving_image = nib.load(warped_moving_image_nii_path).get_fdata()

    return ddf_array, warped_moving_image


def warp_image_roi(ddf_directory: str,
                   dataset_name: str,
                   registration_problem: str) -> NDArray[np.float_]:

    pair_num = get_pair_num(dataset_name, registration_problem)
    network_outputs_path = f"{ddf_directory}/test/pair_{pair_num}"
    ddf_nii_path = f"{network_outputs_path}/ddf.nii.gz"
    ddf_array = nib.load(ddf_nii_path).get_fdata()

    nrrd_images_path = locate_dataset(dataset_name)
    t_moving, t_fixed = registration_problem.split('to')
    fixed_image_roi_path = \
            f"{nrrd_images_path}/img_roi_watershed/{t_fixed}.nrrd"
    moving_image_roi_path = \
            f"{nrrd_images_path}/img_roi_watershed/{t_moving}.nrrd"

    problem_id = f"{dataset_name}/{registration_problem}"
    resized_fixed_image_roi = resize_image_roi(
                fixed_image_roi_path,
                CM_DICT[problem_id][1])
    resized_moving_image_roi = resize_image_roi(
                moving_image_roi_path,
                CM_DICT[problem_id][0])
    euler_transform_image_roi(resized_image_roi, ddf_array, problem_id)


def get_pair_num(dataset_name: str, registration_problem: str):

    problems = ALL_PROBLEMS["test"][dataset_name]

    return problems.index(registration_problem)


def resize_image_roi(image_roi_path: str, image_CM: list[int]):

    image_roi_T = get_image_T(image_roi_path)
    resized_image_roi = get_cropped_image(
        image_roi_T, image_CM, -1).astype(np.float32)

    return resized_image_roi


def euler_transform_image_roi(moving_image_roi: NDArray[np.int_],
                ddf_array: NDArray[np.float_],
                problem_id: str) -> NDArray[np.float_]:

    #EULER_PARAMETERS_DICT[problem_id]
    pass


if __name__ == "__main__":

    base = "/home/alicia/data_personal/regnet_ckpt/logs_predict"
    ckpt = "euler-gpu_resize-v1_size-v1_aug_reg-nonrigid_20231116-134048"
    ddf_directory = f"{base}/{ckpt}"
    dataset_name = "2022-04-18-04"
    registration_problem = "1012to1453"
    ddf_array, warped_moving_image = warp_channel_images(
            ddf_directory, dataset_name, registration_problem)
    print(ddf_array.shape)
    print(warped_moving_image.shape)
