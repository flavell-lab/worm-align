"""
This script crops image into smaller size given datasets with preprocessed
images and ROI labels
"""
from tqdm import tqdm
from typing import List, Tuple
from wormalign.utils import get_cropped_image
import h5py
import json
import numpy as np
import os


class Cropper:

    def __init__(
        self,
        dataset_path: str,
        dataset_name: List[str],
        target_image_shape: Tuple[int, int, int]
    ):
        self.dataset_path = dataset_path
        self._dataset_name = dataset_name
        self.target_image_shape = target_image_shape
        self.CM_dict = self._load_json("resources/center_of_mass_ALv0.json")
        self._update_dataset()

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, new_dataset: str):
        self._dataset_name = new_dataset
        self._update_dataset()

    def _update_dataset(self):
        self.base = f"{self.dataset_path}/{self._dataset_name}"
        self.path_dict = {
            "fixed_images": f"{self.base}/fixed_images.h5",
            "fixed_labels": f"{self.base}/fixed_labels.h5",
            "moving_images": f"{self.base}/moving_images.h5",
            "moving_labels": f"{self.base}/moving_labels.h5"
        }
        self.problems = list(h5py.File(self.path_dict["fixed_images"],
            "r").keys())

    def _load_json(self, file_path: str):
        with open(file_path, "r") as f:
             return json.load(f)

    def crop_dataset(self):

        for image_type, path in self.path_dict.items():

            hdf5_file = h5py.File(path, "r+")

            for problem in self.problems:
                new_CM = self._compute_new_CM(problem)
                image = hdf5_file[problem][:]
                print(image)
                print(type(image[0,0,0]))
                if "label" in image_type:
                    cropped_image = get_cropped_image(
                            image.astype(np.float64),
                            new_CM,
                            list(self.target_image_shape),
                            -1).astype(np.int32)

                elif "image" in image_type:
                    cropped_image = get_cropped_image(
                            image.astype(np.float64),
                            new_CM,
                            list(self.target_image_shape),
                            -1).astype(np.float32)
                    hdf5_file.create_dataset(
                            problem, data=cropped_image
                    )
            hdf5_file.close()
            print(f"{path} is updated!")

    def _compute_new_CM(self, problem):
        dataset_date = self.dataset_name.split("_")[0]
        problem_id = f"{dataset_date}/{problem}"
        center_of_mass = self.CM_dict[problem_id]

        return ((np.array(center_of_mass["moving"]) +
                np.array(center_of_mass["fixed"]))/2).tolist()


def crop_datasets():
    dataset_path = "/data1/prj_register"
    datasets = ["2022-01-27-04_subset_cleaned_sizev0"]
                #"2022-03-16-02_subset_cleaned_sizev0",
                #"2022-06-14-01_subset_cleaned_sizev0"]
    target_image_shape = (208, 96, 56)
    cropper = Cropper(dataset_path, datasets[0], target_image_shape)
    for dataset in datasets:
        cropper.dataset_name = dataset
        cropper.crop_dataset()
        print(f"{len(cropper.problems)} problems in {cropper._dataset_name} cropped")


if __name__ == "__main__":
    crop_datasets()

