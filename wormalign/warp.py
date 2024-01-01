from euler_gpu.preprocess import initialize
from euler_gpu.transform import transform_image_3d
from numpy.typing import NDArray
from typing import Dict, List, Tuple
from wormalign.utils import (locate_dataset, get_cropped_image, get_image_T)
import deepreg.model.layer as layer
import json
import nibabel as nib
import numpy as np
import tensorflow as tf
import torch


# Load JSON resources
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

ALL_PROBLEMS = load_json("resources/registration_problems.json")
CM_DICT = load_json("resources/center_of_mass.json")
EULER_PARAMETERS_DICT = load_json("resources/euler_parameters.json")
PAIR_NUM_DICT = load_json("resources/problem_to_pairnum.json")

class ImageWarper:

    def __init__(
        self,
        ddf_directory: str,
        dataset_name: str,
        registration_problem: str,
        image_shape: Tuple[int, int, int],
        device_name: str = "cuda:0"
    ):
        self.ddf_directory = ddf_directory
        self.dataset_name = dataset_name
        self.registration_problem = registration_problem
        self.problem_id = f"{dataset_name}/{self.registration_problem}"
        self.image_shape = image_shape
        self.device = torch.device(self.device_name)

    def get_network_outputs(
        self,
        target: str = "all"
    ):
        network_output_path = self._get_problem_path(
                self.registration_problem)
        network_outputs = self._load_images(network_outputs_path)

        if target == "all":
            return network_outputs
        elif:
            return network_outputs[target]

    def _get_problem_path(self):
        pair_num = PAIR_NUM_DICT[self.problem_id]
        return f"{self.ddf_directory}/test/pair_{pair_num}"

    def _load_images(
        self,
        network_outputs_path: str
    ):
        ddf_nii_path = f"{network_outputs_path}/ddf.nii.gz"
        ddf_array = nib.load(ddf_nii_path).get_fdata()
        warped_moving_image = nib.load(
                f"{network_outputs_path}/pred_fixed_image.nii.gz").get_fdata()
        fixed_image = nib.load(
                f"{network_outputs_path}/fixed_image.nii.gz").get_fdata()
        moving_image = nib.load(
                f"{network_outputs_path}/moving_image.nii.gz").get_fdata()

        return {
                "ddf": ddf_array,
                "warped_moving_image": warped_moving_image,
                "fixed_image": fixed_image,
                "moving_image": moving_image
        }

    def _preprocess_image_roi(self):
        """
        Resize the ROI images and Euler-transform them with the same parameters
        of their corresponding channel images.
        """
        nrrd_images_path = locate_dataset(self.dataset_name)
        t_moving, t_fixed = self.problem_id.split('to')
        fixed_image_roi_path = f"{nrrd_images_path}/img_roi_watershed/{t_fixed}.nrrd"
        moving_image_roi_path = f"{nrrd_images_path}/img_roi_watershed/{t_moving}.nrrd"

        # resize the fixed and moving image ROIs
        resized_fixed_image_roi = self._resize_image_roi(
                fixed_image_roi_path,
                CM_DICT[problem_id][1]
        )
        resized_moving_image_roi = self._resize_image_roi(
                moving_image_roi_path,
                CM_DICT[problem_id][0]
        )
        euler_transformed_moving_image_roi = self._euler_transform_image_roi(
                resized_moving_image_roi
        )
        return {
            "fixed_image_roi": resized_fixed_image_roi,
            "moving_image_roi": resized_fixed_image_roi,
            "euler_tfmed_moving_image_roi": euler_transformed_moving_image_roi
        }

    def _resize_image_roi(
        self,
        image_roi_path: str,
        image_CM: List[int, int, int]
    ):
        image_roi_T = get_image_T(image_roi_path)

        return get_cropped_image(
                image_roi_T,
                image_CM,
                self.image_shape,
                -1).astype(np.float32)

    def _euler_transform_image_roi(
        self,
        moving_image_roi: NDArray[np.int_]
    ):
        x_dim, y_dim, z_dim = self.image_shape
        _memory_dict_xy = self._initialize_memory_dict(x_dim, y_dim, z_dim)
        return self._apply_euler_parameters(
                moving_image_roi,
                _memory_dict_xy
        )

    def _initialize_memory_dict(
        self,
        dim_1: int,
        dim_2: int,
        dim_3: int
    ) -> Dict[str, torch.Tensor]:
        _memory_dict = initialize(
                np.zeros((dim_1, dim_2)).astype(np.float32),
                np.zeros((dim_1, dim_2)).astype(np.float32),
                np.zeros(dim_3),
                np.zeros(dim_3),
                np.zeros(dim_3),
                dim_3,
                self.device
        )
        return _memory_dict

    def _apply_euler_parameters(
        self,
        moving_image_roi: NDArray[np.int_],
        memory_dict: Dict[str, torch.Tensor],
        dimension: str = "xy"
    ) ->  NDArray[np.int_]:

        best_transformation = torch.tensor(
            EULER_PARAMETERS_DICT[self.problem_id][dimension]
        ).to(self.device)

        moving_image_roi = self._adjust_image_shape(
                moving_image_roi,
                dimension
        )
        transformed_moving_image_roi = transform_image_3d(
                moving_image_roi,
                memory_dict,
                best_transformation,
                self.device
        )
        translated_moving_image_roi = self._translate_image(
                self._adjust_image_shape(
                    transformed_moving_image_roi,
                    "xy"
                )
        )

        return translated_moving_image_roi

    def _adjust_image_shape(
        self,
        image: NDArray[np.int_],
        dimension: str,
    ) -> NDArray[np.int_]:

        x_dim, y_dim, z_dim = self.image_shape
        reshaping_dict = {
            (y_dim, z_dim, x_dim): {
                "xy": (2, 0, 1),
                "xz": (2, 1, 0),
                "yz": (0, 1, 2)
            },
            (x_dim, z_dim, y_dim): {
                "xy": (0, 2, 1),
                "yz": (2, 1, 0),
                "xz": (0, 1, 2)
            },
            (x_dim, y_dim, z_dim): {
                "yz": (1, 2, 0),
                "xz": (0, 2, 1),
                "xy": (0, 1, 2)
            }
        }
        order = reshaping_dict[image.shape][dimension]

        return np.transpose(image, order)

    def _translate_image(
        self,
        image: NDArray[np.int_]
    ) -> NDArray[np.int_]:

        dz = EULER_PARAMETERS_DICT[self.problem_id]["dz"]
        translated_image = np.full(self.image_shape, 0)

        if dz < 0:
            translated_image[:, :, :dz] = image[:, :, -dz:]
        elif dz > 0:
            translated_image[:, :, dz:] = image[:, :, :-dz]
        elif dz == 0:
            translated_image = image

    return translated_image

