from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import (initialize,
    max_intensity_projection_and_downsample)
from euler_gpu.transform import (transform_image_3d, translate_along_z)
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Dict, List, Tuple
from wormalign.evaluate import calculate_gncc
from wormalign.utils import (write_to_json, locate_dataset, filter_and_crop,
        get_image_T, get_image_CM, filter_image)
import glob
import h5py
import json
import numpy as np
import os
import random
import torch


class RandomRotate:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be
    either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent
    between raw and labeled datasets, otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across
    (1,2) axis)
    """

    def __init__(
        self,
        random_state: int,
        axes: List[Tuple[int, int]] = [(1,2)],
        **kwargs,
    ):
        """
        Init.

        :param random_state: an integer as random seed
        :param axes: axes of rotation
        """
        self.random_state = random_state
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.axes = axes

    def __call__(
        self,
        image: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Rotate image by 180 degrees probalistically.

        :param image: array of shape: (x_dim, y_dim, z_dim)
        """
        axis = self.axes[self.random_state.randint(len(self.axes))]
        assert image.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        k = self.random_state.choice([0, 2])
        # rotate k times around a given plane
        if image.ndim == 3:
            image = np.rot90(image, k, axis)
        else:
            channels = [np.rot90(image[c], k, axis) for c in range(image.shape[0])]
            image = np.stack(channels, axis=0)

        return image

    def _adjust_shape(
        self,
        image: NDArray[np.float32],
        target_shape: Tuple[int, int, int],
    ) -> NDArray[np.float32]:

        x_dim, y_dim, z_dim = target_shape

        if image.shape == (x_dim, z_dim, y_dim):
            image = np.transpose(image, (0, 2, 1))
        elif image.shape == (y_dim, x_dim, z_dim):
            image = np.transpose(image, (1, 0, 2))
        elif image.shape == (y_dim, z_dim, x_dim):
            image = np.transpose(image, (2, 0, 1))
        elif image.shape == (z_dim, y_dim, x_dim):
            image = np.transpose(image, (2, 1, 0))
        elif image.shape == (z_dim, x_dim, y_dim):
            image = np.transpose(image, (1, 2, 0))

        return image

    def augment(
        self,
        image_dir: str, 
        dataset_names: List[str],
    ):
        """
        Generate a new set of images from the given datasets that are augmented
        by rotation.

        :param image_dir: directory to keep the new set of images as .h5 files
        :param dataset_names: a list of dataset names

        :example
            >>> image_dir = "/home/alicia/data_personal"
            >>> random_seed = 0
            >>> rotate180 = RandomRotate(np.random.RandomState(random_seed))
            >>> dataset_names = ["2022-01-09-01", "2022-01-17-01"]
            >>> rotate180.augment(image_dir, dataset_names)
        """
        for dataset_name in dataset_names:

            h5_m_file = h5py.File(
                    f"{image_dir}/nonaugmented/{dataset_name}/moving_images.h5",
                    "r")
            h5_f_file = h5py.File(
                    f"{image_dir}/nonaugmented/{dataset_name}/fixed_images.h5",
                    "r")
            problems = list(h5_m_file.keys())

            augmented_image_dir = f"{image_dir}/augmented/{dataset_name}"
            if not os.path.exists(augmented_image_dir):
                os.mkdir(augmented_image_dir)

            h5_aug_m_file = h5py.File(
                    f"{augmented_image_dir}/moving_images.h5", "w")
            h5_aug_f_file = h5py.File(
                    f"{augmented_image_dir}/fixed_images.h5", "w")

            x_dim, y_dim, z_dim = h5_m_file[problems[0]][:].shape

            for problem in tqdm(problems):

                moving_image = h5_m_file[problem][:]
                fixed_image = h5_f_file[problem][:]

                rotated_moving_image = self._adjust_shape(
                            self.__call__(moving_image),
                            fixed_image.shape)
                rotated_fixed_image = self._adjust_shape(
                            self.__call__(fixed_image),
                            fixed_image.shape)

                h5_aug_m_file.create_dataset(
                        problem,
                        data=rotated_moving_image)
                h5_aug_f_file.create_dataset(
                        problem,
                        data=rotated_fixed_image)

            h5_aug_m_file.close()
            h5_aug_f_file.close()


class RegistrationProcessor:

    def __init__(
        self,
        target_image_shape: Tuple[int, int, int],
        save_directory: str,
        batch_size: int = 200,
        device_name: str = "cuda:2",
        downsample_factor: int = 4,
    ):
        """
        Init.

        :param target_image_shape: shape of the image (x_dim, y_dim, z_dim)
        :param save_directory: the directory to save the registered images
        :param batch_size: the size of a batch to process with Euler-GPU
        :param device name
        :param downsample_factor: factor to downsample the images during grid
            search stage of running Euler-GPU
        """
        self.target_image_shape = target_image_shape
        self.save_directory = save_directory
        self.batch_size = batch_size
        self.device_name = device_name
        self.downsample_factor = downsample_factor

        self.euler_parameters_dict = dict()
        self.outcomes = dict()
        self.CM_dict = dict()
        self.memory_dict_xy, self._memory_dict_xy = \
                self._initialize_memory_dict()

        self._ensure_directory_exists(self.save_directory)

    def _ensure_directory_exists(self, path):
        """
        Create the given directory if it does not already exist

        :param path: directory to create if does not exist
        """
        if not os.path.exists(path):
            os.mkdir(path)

    def _initialize_memory_dict(self):
        """
        Initialize memory dictory for storing the parameters for running
        Euler-GPU
        """
        x_dim, y_dim, z_dim = self.target_image_shape
        z_translation_range = range(-z_dim, z_dim)
        x_translation_range_xy = np.sort(np.concatenate((
                np.linspace(-0.24, 0.24, 49),
                np.linspace(-0.46, -0.25, 8),
                np.linspace(0.25, 0.46, 8),
                np.linspace(0.5, 1, 3),
                np.linspace(-1, -0.5, 3)
        )))
        y_translation_range_xy = np.sort(np.concatenate((
                np.linspace(-0.28, 0.28, 29),
                np.linspace(-0.54, -0.3, 5),
                np.linspace(0.3, 0.54, 5),
                np.linspace(0.6, 1.4, 3),
                np.linspace(-1.4, -0.6, 3)
        )))
        theta_rotation_range_xy = np.sort(np.concatenate((
                np.linspace(0, 19, 20),
                np.linspace(20, 160, 29),
                np.linspace(161, 199, 39),
                np.linspace(200, 340, 29),
                np.linspace(341, 359, 19)
        )))
        memory_dict_xy = initialize(
                    np.zeros((
                        x_dim // self.downsample_factor,
                        y_dim // self.downsample_factor)).astype(np.float32),
                    np.zeros((
                        x_dim // self.downsample_factor,
                        y_dim // self.downsample_factor)).astype(np.float32),
                    x_translation_range_xy,
                    y_translation_range_xy,
                    theta_rotation_range_xy,
                    self.batch_size,
                    self.device_name
        )
        _memory_dict_xy = initialize(
                    np.zeros((x_dim, y_dim)).astype(np.float32),
                    np.zeros((x_dim, y_dim)).astype(np.float32),
                    np.zeros(z_dim),
                    np.zeros(z_dim),
                    np.zeros(z_dim),
                    z_dim,
                    self.device_name
        )
        return memory_dict_xy, _memory_dict_xy

    def process_datasets(
        self,
        augment: bool = False,
    ):
        """
        Process datasets by Euler-transforming the moving images with the set
        of parameters that maximize the GNCC between the fixed and moving
        image.

        :example
            >>> processor = RegistrationProcessor(
            >>>    (208, 96, 56),
            >>>    "datasets",
            >>>    device_name = "cuda:0"
            >>> )
            >>> processor.process_datasets()
        """
        with open("resources/registration_problems.json", 'r') as f:
            registration_problem_dict = json.load(f)

        for dataset_type, problem_dict in registration_problem_dict.items():

            # process training, validation, and testing datasets respectively
            self.process_dataset_type(dataset_type, problem_dict)

        write_to_json(self.outcomes, "eulergpu_outcomes")
        write_to_json(self.CM_dict, "center_of_mass")
        write_to_json(self.euler_parameters_dict, "euler_parameters")

    def process_dataset_type(
        self,
        dataset_type: str,
        problem_dict: Dict[str, List[str]],
    ):
        """
        Process datasets that belong to the same type

        :param dataset_type: the type of dataset
            (i.e., `train`, `valid`, or `test`)
        :param problem_dict: a dictionary of problems for each dataset as
            {
                "YYYY-MM-DD-XX": [xtox, ..],
                "YYYY-MM-DD-XX": [xtox, ..],
                ...
            }
        """
        dataset_type_dir = f"{self.save_directory}/{dataset_type}"
        self._ensure_directory_exists(dataset_type_dir)
        self._ensure_directory_exists(f"{dataset_type_dir}/nonaugmented")
        for dataset_name, problems in problem_dict.items():
            print(f"=====Processing {dataset_name} in {dataset_type}=====")
            self.process_dataset(
                    dataset_name,
                    problems,
                    f"{dataset_type_dir}/nonaugmented"
            )

    def process_dataset(
        self,
        dataset_name: str,
        problems: List[str],
        dataset_type_dir: str,
    ):
        """
        Process a given dataset with Euler transformation.

        :param dataset_name: name of the dataset
        :param problems: a list of problems from this dataset
        :param dataset_type_dir: directory to save the processed dataset
        """
        save_path = f"{dataset_type_dir}/{dataset_name}"
        self._ensure_directory_exists(save_path)
        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')
        dataset_path = locate_dataset(dataset_name)

        for problem in tqdm(problems):
            problem_id = f"{dataset_name}/{problem}"
            self.process_problem(
                    problem_id,
                    dataset_path,
                    hdf5_m_file,
                    hdf5_f_file
            )
        hdf5_m_file.close()
        hdf5_f_file.close()

    def process_problem(
        self,
        problem_id: str,
        dataset_path: str,
        hdf5_m_file: h5py.File,
        hdf5_f_file: h5py.File,
    ):
        """
        Euler-transform the moving image from a given registration problem.

        :param problem_id: ID that identifies the dataset name and problem name
        :param dataset_path: the path to save the dataset
        :param hdf5_m_file: hdf5 File that keeps the moving images
        :param hdf5_f_file: hdf5 File that keeps the fixed images
        """
        self.outcomes[problem_id] = dict()
        self.euler_parameters_dict[problem_id] = dict()

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
        # project onto the x-y plane along the maximum z
        downsampled_resized_fixed_image_xy = \
                max_intensity_projection_and_downsample(
                        resized_fixed_image_xyz,
                        self.downsample_factor,
                        projection_axis = 2).astype(np.float32)
        downsampled_resized_moving_image_xy = \
                max_intensity_projection_and_downsample(
                        resized_moving_image_xyz,
                        self.downsample_factor,
                        projection_axis = 2).astype(np.float32)

        # update the memory dictionary for grid search on x-y image
        self.memory_dict_xy["fixed_images_repeated"][:] = torch.tensor(
                downsampled_resized_fixed_image_xy,
                device = self.device_name,
                dtype = torch.float32
            ).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        self.memory_dict_xy["moving_images_repeated"][:] = torch.tensor(
                downsampled_resized_moving_image_xy,
                device = self.device_name,
                dtype = torch.float32
            ).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

        # search optimal parameters with projected image on the x-y plane
        best_score_xy, best_transformation_xy = grid_search(self.memory_dict_xy)
        # transform the 3d image with the searched parameters
        transformed_moving_image_xyz = transform_image_3d(
                    resized_moving_image_xyz,
                    self._memory_dict_xy,
                    best_transformation_xy,
                    self.device_name
        )
        # search for the optimal dz translation
        z_dim = self.target_image_shape[2]
        z_translation_range = range(-z_dim, z_dim)
        dz, gncc, transformed_moving_image_xyz = translate_along_z(
                    z_translation_range,
                    resized_fixed_image_xyz,
                    transformed_moving_image_xyz,
                    moving_image_median
        )
        # log the results
        self.CM_dict[problem_id] = {
                "moving": get_image_CM(moving_image_T),
                "fixed": get_image_CM(fixed_image_T)
        }
        self.euler_parameters_dict[problem_id]["xy"] = [
                score.item() for score in list(best_transformation_xy)
        ]
        self.euler_parameters_dict[problem_id]["dz"] = dz
        self.outcomes[problem_id]["registered_image_xy_gncc"] = \
                best_score_xy.item()
        self.outcomes[problem_id]["registered_image_yz_gncc"] = \
                calculate_gncc(
                    resized_fixed_image_xyz.max(0),
                    transformed_moving_image_xyz.max(0)
                ).item()
        self.outcomes[problem_id]["registered_image_xz_gncc"] = \
                calculate_gncc(
                    resized_fixed_image_xyz.max(1),
                    transformed_moving_image_xyz.max(1)
                ).item()
        self.outcomes[problem_id]["registered_image_xyz_gncc"] = gncc
