from numpy.typing import NDArray
from tqdm import tqdm
from typing import List, Dict, Tuple
from wormalign.utils import get_cropped_image, write_to_json
from wormalign.warp import ImageWarper
import glob
import h5py
import json
import nibabel as nib
import numpy as np
import os


class LabelWarper(ImageWarper):

    def __init__(self, *args, **kwargs):

        super().__init__(None, *args, **kwargs)
        self.label_path = "/data1/prj_register/deepreg_labels"
        self.bad_labels = []

    def warp_label(self):
        """
        Warp the ROI images with the Euler parameters obtained from
        preprocessing the registration problems for training and validation
        """
        if self._label_exists():
            return self._preprocess_image_roi()

    def _label_exists(self):

        return self.registration_problem in os.listdir(
                f"{self.label_path}/{self.dataset_name}/register_labels/"
        )

    def _update_problem(self):
        self.problem_id = f"{self.dataset_name}/{self._registration_problem}"

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

        if self.nonzero_labels(fixed_image_roi) and \
                self.nonzero_labels(moving_image_roi):

            resized_fixed_image_roi = self._resize_image_roi(
                    fixed_image_roi,
                    self.CM_dict[self.problem_id]["fixed"]
            )
            resized_moving_image_roi = self._resize_image_roi(
                    moving_image_roi,
                    self.CM_dict[self.problem_id]["moving"]
            )
            # pass interpolation method
            euler_transformed_moving_image_roi = self._euler_transform_image_roi(
                    resized_moving_image_roi
            )
            return {
                "fixed_image_roi": resized_fixed_image_roi,
                "moving_image_roi": resized_moving_image_roi,
                "euler_tfmed_moving_image_roi": euler_transformed_moving_image_roi
            }
        else:
            return {}

    def _resize_image_roi(
        self,
        image_roi: NDArray[np.float32],
        image_CM: List[int]
    ) -> NDArray[np.float32]:

        return get_cropped_image(
                image_roi,
                image_CM,
                self.image_shape, -1).astype(np.float32)

    def nonzero_labels(self, label_roi):

        if len(np.unique(label_roi)) == 1:
            self.bad_labels.append(self.registration_problem)
            return False
        else:
            return True


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
    all_bad_problems = {"train": {}, "valid": {}}
    for dataset_type, datasets in dataset_types.items():
        all_bad_problems[dataset_type] = generate_label(
                datasets,
                dataset_type,
                warper,
                save_directory,
                problem_dict
        )
    write_to_json(all_bad_problems, "bad_registration_problems_ALv1")

def generate_label(
        datasets: List[str],
        dataset_type: str,
        warper: LabelWarper,
        save_directory: str,
        problem_dict: Dict[str, Dict[str, List[str]]]
    ):
    bad_problems = dict()

    for dataset in datasets:
        label_path = f"{save_directory}/{dataset_type}/nonaugmented/{dataset}"
        problems = problem_dict[dataset_type][dataset]
        warper.dataset_name = dataset

        with h5py.File(f"{label_path}/moving_rois.h5", "w") as h5_m_file, \
             h5py.File(f"{label_path}/fixed_rois.h5", "w") as h5_f_file:

            for problem in tqdm(problems):
                warper.registration_problem = problem
                label_dict = warper.warp_label()
                if len(label_dict) > 0:
                    moving_image_roi = \
                            label_dict["euler_tfmed_moving_image_roi"]
                    h5_m_file.create_dataset(
                            problem,
                            data = moving_image_roi
                    )
                    fixed_image_roi = label_dict["fixed_image_roi"]
                    h5_f_file.create_dataset(
                            problem,
                            data = fixed_image_roi
                    )
        print(f"{dataset} generated!")
        bad_problems[dataset] = warper.bad_labels

    return bad_problems
