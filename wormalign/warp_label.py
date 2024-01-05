from numpy.typing import NDArray
from tqdm import tqdm
from typing import List, Dict, Tuple
from wormalign.utils import get_cropped_image
from wormalign.warp import ImageWarper
import h5py
import json
import nibabel as nib
import numpy as np
import os


class LabelWarper(ImageWarper):

    def __init__(self, *args, **kwargs):

        super().__init__(None, *args, **kwargs)
        self.registration_problems = self._load_json(
                "resources/registration_problems_ALv0.json")
        self.label_path = "/data1/prj_register/deepreg_labels"

    def warp_label(self):
        """
        Warp the ROI images with the Euler parameters obtained from
        preprocessing the registration problems for training and validation
        """
        if self._label_exists():
            return self._preprocess_image_roi()
        else:
            print(
            f"{self.dataset_name}/{self.registration_problem} has no label")

    def _label_exists(self):

        return self.registration_problem in os.listdir(
                f"{self.label_path}/{self.dataset_name}/register_labels/"
        )

    def _update_problem(self):
        self.problem_id = f"{self.dataset_name}/{self._registration_problem}"

    def _nonempty(self):
        return self.dataset_name == "" or self.registration_problem == ""

    def _preprocess_image_roi(self):
        """
        Redefine this method for LabelWarper.
        Implement the changes to the method here.
        """
        roi_path = \
            f"{self.label_path}/{self.dataset_name}/register_labels/{self.registration_problem}"
        fixed_image_roi = nib.load(
                f"{roi_path}/img_fixed.nii.gz").get_fdata().astype(np.float32)
        moving_image_roi = nib.load(
                f"{roi_path}/img_moving.nii.gz").get_fdata().astype(np.float32)

        resized_fixed_image_roi = self._resize_image_roi(
                fixed_image_roi,
                self.cm_dict[self.problem_id]["fixed"]
        )
        resized_moving_image_roi = self._resize_image_roi(
                moving_image_roi,
                self.cm_dict[self.problem_id]["moving"]
        )
        euler_transformed_moving_image_roi = self._euler_transform_image_roi(
                resized_moving_image_roi
        )
        return {
            "fixed_image_roi": resized_fixed_image_roi,
            "moving_image_roi": resized_moving_image_roi,
            "euler_tfmed_moving_image_roi": euler_transformed_moving_image_roi
        }

    def _resize_image_roi(
        self,
        image_roi: NDArray[np.float32],
        image_CM: List[int]
    ) -> NDArray[np.float32]:

        return get_cropped_image(
                image_roi,
                image_CM,
                self.image_shape, -1).astype(np.float32)


def generate_labels(
        train_datasets: List[str],
        valid_datasets: List[str],
        device_name: str,
        target_image_shape: Tuple[int, int, int],
        registration_problem_file: str,
        save_directory: str
    ):

    with open(f"resources/{registration_problem_file}", "r") as f:
        problem_dict = json.load(f)

    warper = LabelWarper(
        None,  # Assuming dataset_name and registration_problem are set later
        None,
        target_image_shape,
        device_name
    )
    dataset_types = {
        "train": train_datasets,
        "valid": valid_datasets,
    }
    for dataset_type, datasets in dataset_types.items():
        generate_label(
                datasets,
                dataset_type,
                warper,
                save_directory,
                problem_dict
        )


def generate_label(
        datasets: List[str],
        dataset_type: str,
        warper: LabelWarper,
        save_directory: str,
        problem_dict: Dict[str, Dict[str, List[str]]]
    ):
    for dataset in tqdm(datasets):
        label_path = f"{save_directory}/{dataset_type}/nonaugmented/{dataset}"
        problems = problem_dict[dataset_type][dataset]
        warper.dataset_name = dataset

        with h5py.File(f"{label_path}/moving_labels.h5", "w") as h5_m_file, \
             h5py.File(f"{label_path}/fixed_labels.h5", "w") as h5_f_file:

            for problem in problems:
                warper.registration_problem = problem
                label_dict = warper.warp_label()

                h5_m_file.create_dataset(
                        problem,
                        data = label_dict["euler_tfmed_moving_image_roi"]
                )
                h5_f_file.create_dataset(
                        problem,
                        data = label_dict["fixed_image_roi"]
                )
            print(f"{dataset} generated!")


if __name__ == "__main__":
    train_datasets = ["2022-01-09-01", "2022-01-17-01", "2022-01-23-04"]
    valid_datasets = ["2022-02-16-04", "2022-04-05-01"]
    device_name = "cuda:0"
    target_image_shape = (284, 120, 64)
    registration_problem_file = "registration_problems_ALv0.json"
    save_directory = "/home/alicia/data_personal/wormalign/datasets"

    generate_labels(train_datasets, valid_datasets, device_name,
            target_image_shape, registration_problem_file, save_directory)

