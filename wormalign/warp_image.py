from euler_gpu.preprocess import initialize
from euler_gpu.transform import transform_image_3d
from numpy.typing import NDArray
from typing import Dict
from wormalign.preprocess import RegistrationProcessor
from wormalign.utils import filter_and_crop, get_image_T
import glob
import h5py
import json
import numpy as np
import torch


class Channel2ImageWarper(RegistrationProcessor):

    def __init__(self, *args, **kwargs):

        super().__init__(euler_search=False, *args, **kwargs)
        self.batch_size = None
        self.downsample_factor = None
        self.CM_dict = self._load_json(
                "resources/center_of_mass_ALv1.json")
        self.euler_parameters_dict = self._load_json(
                "resources/euler_parameters_ALv1.json")
        self.memory_dict_xy, self._memory_dict_xy = {}, {}

    def _load_json(self, file_path: str):
        """Load JSON resources"""
        with open(file_path, "r") as f:
            return json.load(f)

    def process_datasets(self, augment: bool = False):

        with open("resources/registration_problems_ALv1.json", 'r') as f:
            registration_problem_dict = json.load(f)

        for dataset_type, problem_dict in registration_problem_dict.items():
            self.process_dataset_type(dataset_type, problem_dict)

    def process_problem(
        self,
        problem_id: str,
        dataset_path: str,
        hdf5_m_file: h5py.File,
        hdf5_f_file: h5py.File,
    ) -> Dict[str, NDArray[np.float32]]:
        """
        Euler-transform the moving image from a given registration problem.

        :param problem_id: ID that identifies the dataset name and problem name
        :param dataset_path: the path to save the dataset
        :param hdf5_m_file: hdf5 File that keeps the moving images
        :param hdf5_f_file: hdf5 File that keeps the fixed images
        """
        t_moving, t_fixed = problem_id.split('/')[1].split('to')
        t_moving_4 = t_moving.zfill(4)
        t_fixed_4 = t_fixed.zfill(4)
        fixed_image_path = glob.glob(
                f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
        )[0]
        moving_image_path = glob.glob(
                f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
        )[0]
        fixed_image_T = get_image_T(fixed_image_path)
        fixed_image_median = np.median(fixed_image_T)
        moving_image_T = get_image_T(moving_image_path)
        moving_image_median = np.median(moving_image_T)

        resized_fixed_image_xyz = filter_and_crop(
                fixed_image_T,
                fixed_image_median,
                self.target_image_shape
        )
        resized_moving_image_xyz = filter_and_crop(
                moving_image_T,
                moving_image_median,
                self.target_image_shape
        )
        euler_transformed_moving_image = self._euler_transform_image(
                resized_moving_image_xyz,
                problem_id
        )
        return {
            "fixed_image": resized_fixed_image_xyz,
            "moving_image": euler_transformed_moving_image
        }

    def _euler_transform_image(
        self,
        moving_image: NDArray[np.float_],
        problem_id: str,
        interpolation: str = "bilinear"
    ):
        x_dim, y_dim, z_dim = self.target_image_shape
        _memory_dict_xy = self._initialize_memory_dict(x_dim, y_dim, z_dim)

        return self._apply_euler_parameters(
                moving_image,
                _memory_dict_xy,
                interpolation,
                problem_id
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
                self.device_name
        )
        return _memory_dict

    def _apply_euler_parameters(
        self,
        moving_image_roi: NDArray[np.float_],
        memory_dict: Dict[str, torch.Tensor],
        interpolation: str,
        problem_id: str,
        dimension: str = "xy",
    ) ->  NDArray[np.float_]:

        best_transformation = torch.tensor(
            self.euler_parameters_dict[problem_id][dimension]
        ).to(self.device_name)

        moving_image_roi = self._adjust_image_shape(
                moving_image_roi,
                dimension
        )
        if dimension == "xy":
            axis = 2
        elif dimension == "xz":
            axis = 1
        elif dimension == "yz":
            axis = 0
        transformed_moving_image_roi = transform_image_3d(
                moving_image_roi,
                memory_dict,
                best_transformation,
                self.device_name, # "cuda:2"
                axis,
                interpolation
        )
        translated_moving_image_roi = self._translate_image(
                self._adjust_image_shape(
                    transformed_moving_image_roi,
                    "xy"
                ),
                problem_id
        )

        return translated_moving_image_roi

    def _adjust_image_shape(
        self,
        image: NDArray[np.float_],
        dimension: str,
    ) -> NDArray[np.float_]:

        x_dim, y_dim, z_dim = self.target_image_shape
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
        image: NDArray[np.float_],
        problem_id: str
    ) -> NDArray[np.float_]:

        dz = self.euler_parameters_dict[problem_id]["dz"]
        translated_image = np.full(self.target_image_shape, 0)

        if dz < 0:
            translated_image[:, :, :dz] = image[:, :, -dz:]
        elif dz > 0:
            translated_image[:, :, dz:] = image[:, :, :-dz]
        elif dz == 0:
            translated_image = image

        return translated_image


