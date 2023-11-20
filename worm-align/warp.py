from euler_gpu.preprocess import initialize
from euler_gpu.transform import transform_image_3d
from numpy.typing import NDArray
from utils import locate_dataset, get_cropped_image, get_image_T, get_image_CM
import deepreg.model.layer as layer
import json
import nibabel as nib
import numpy as np
import tensorflow as tf
import torch


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

    fixed_image_nii_path = \
            f"{network_outputs_path}/fixed_image.nii.gz"
    fixed_image = nib.load(fixed_image_nii_path).get_fdata()

    moving_image_nii_path = \
            f"{network_outputs_path}/moving_image.nii.gz"
    moving_image = nib.load(moving_image_nii_path).get_fdata()

    return ddf_array, fixed_image, moving_image, warped_moving_image


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
    device_name = 0
    # warp the moving image roi with euler parameters
    euler_warped_moving_image_roi = euler_transform_image_roi(
                resized_moving_image_roi,
                ddf_array,
                problem_id,
                device_name)
    # warp the moving image roi with ddf
    resized_moving_image_roi_tf = tf.cast(
                tf.expand_dims(euler_warped_moving_image_roi,
                axis=0), dtype=tf.float32
    )
    warping = layer.Warping(fixed_image_size=resized_moving_image_roi_tf.shape[1:4],
                            interpolation="nearest")
    warped_moving_image_roi_tf = warping(inputs=[ddf_array,
                resized_moving_image_roi_tf])
    warped_moving_image_roi = warped_moving_image_roi_tf.numpy()[0]

    return resized_fixed_image_roi, resized_moving_image_roi, euler_warped_moving_image_roi, warped_moving_image_roi


def get_pair_num(dataset_name: str, registration_problem: str):

    problems = ALL_PROBLEMS["test"][dataset_name]

    return problems.index(registration_problem)


def resize_image_roi(image_roi_path: str, image_CM: list[int]):

    image_roi_T = get_image_T(image_roi_path)
    resized_image_roi = get_cropped_image(
        image_roi_T, image_CM, (208, 96, 56), -1).astype(np.float32)

    return resized_image_roi


def euler_transform_image_roi(moving_image_roi: NDArray[np.int_],
                problem_id: str,
                device_name: int) -> NDArray[np.float_]:

    device = torch.device(f"cuda:{device_name}")
    x_dim, y_dim, z_dim = moving_image_roi.shape

    _memory_dict_xy = initialize(
            np.zeros((x_dim, y_dim)).astype(np.float32),
            np.zeros((x_dim, y_dim)).astype(np.float32),
            np.zeros(z_dim),
            np.zeros(z_dim),
            np.zeros(z_dim),
            z_dim,
            device
    )
    _memory_dict_xz = initialize(
            np.zeros((x_dim, z_dim)).astype(np.float32),
            np.zeros((x_dim, z_dim)).astype(np.float32),
            np.zeros(y_dim),
            np.zeros(y_dim),
            np.zeros(y_dim),
            y_dim,
            device
    )
    _memory_dict_yz = initialize(
            np.zeros((y_dim, z_dim)).astype(np.float32),
            np.zeros((y_dim, z_dim)).astype(np.float32),
            np.zeros(x_dim),
            np.zeros(x_dim),
            np.zeros(x_dim),
            x_dim,
            device
    )
    best_transformation_xy = torch.tensor(
                EULER_PARAMETERS_DICT[problem_id]["xy"]).to(device)
    transformed_moving_image_roi_xyz = transform_image_3d(
                moving_image_roi,
                _memory_dict_xy,
                best_transformation_xy,
                device
    )
    best_transformation_xz = torch.tensor(
                EULER_PARAMETERS_DICT[problem_id]["xz"]).to(device)
    transformed_moving_image_roi_xzy = transform_image_3d(
                np.transpose(transformed_moving_image_roi_xyz, (0, 2, 1)),
                _memory_dict_xz,
                best_transformation_xz,
                device
    )
    best_transformation_yz = torch.tensor(
                EULER_PARAMETERS_DICT[problem_id]["yz"]).to(device)
    transformed_moving_image_roi_yzx = transform_image_3d(
                np.transpose(transformed_moving_image_roi_xzy, (2, 1, 0)),
                _memory_dict_yz,
                best_transformation_yz,
                device
    )
    dz = EULER_PARAMETERS_DICT[problem_id]["dz"]
    warped_moving_image_roi = np.transpose(transformed_moving_image_roi_yzx, (2, 0, 1))

    final_moving_image_roi_xyz = np.full(
            warped_moving_image_roi.shape, 0)
    # translate along the z-axis
    if dz < 0:
        final_moving_image_roi_xyz[:, :, :dz] = \
            warped_moving_image_roi[:, :, -dz:]
    elif dz > 0:
        final_moving_image_roi_xyz[:, :, dz:] = \
            warped_moving_image_roi[:, :, :-dz]
    elif dz == 0:
        final_moving_image_roi_xyz = warped_moving_image_roi

    return final_moving_image_roi_xyz


if __name__ == "__main__":

    base = "/home/alicia/data_personal/regnet_ckpt/logs_predict"
    ckpt = "euler-gpu_resize-v1_size-v1_aug_reg-nonrigid_20231116-134048"
    ddf_directory = f"{base}/{ckpt}"
    dataset_name = "2022-04-18-04"
    registration_problem = "1012to1453"
    #ddf_array, warped_moving_image = warp_channel_images(
    #        ddf_directory, dataset_name, registration_problem)
    #print(ddf_array.shape)
    #print(warped_moving_image.shape)
    warp_image_roi(ddf_directory,
                   dataset_name,
                   registration_problem)

