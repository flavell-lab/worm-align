"""
This scripts pairs the inputs to network by removing the problems from both
`*_images.h5` and `*_labels.h5` that have empty labels; it also generates a new
`registration_problem.json` that contains the filtered problems.
"""
from typing import List
import glob
import h5py
import json
import numpy as np
import os
import yaml


def find_problems_without_labels(dataset_path: str):

    bad_registration_problems_dict = {"train": dict(), "valid": dict()}

    def filter_problems(image_path: str, dataset_type: str):

        with h5py.File(
                f"{image_path}/fixed_images.h5", "r+") as fixed_images_file, \
            h5py.File(
                f"{image_path}/moving_images.h5", "r+") as moving_images_file, \
            h5py.File(
                f"{image_path}/fixed_rois.h5", "r") as fixed_rois_file, \
            h5py.File(
                f"{image_path}/moving_rois.h5", "r") as moving_rois_file:

            # Get the set of keys from fixed_rois.h5
            fixed_rois_keys = set(fixed_rois_file.keys())
            moving_rois_keys = set(moving_rois_file.keys())

            # List of keys to delete from fixed_images.h5
            keys_to_delete_from_fixed = [key for key in fixed_images_file.keys() if
                    key not in fixed_rois_keys]
            keys_to_delete_from_moving = [key for key in moving_images_file.keys() if
                    key not in moving_rois_keys]
            keys_to_delete = np.unique(keys_to_delete_from_fixed + \
                    keys_to_delete_from_moving).tolist()

            dataset_name = image_path.split("/")[-1]
            bad_registration_problems_dict[dataset_type][dataset_name] = keys_to_delete

            # Delete keys not present in fixed_rois.h5 from fixed_images.h5
            for key in keys_to_delete:
                del fixed_images_file[key]
                del moving_images_file[key]
                print(f"{dataset_name}/{key} deleted")

    # check train datasets
    dateset_path_dict = {
        "train": glob.glob(f"{dataset_path}/train/nonaugmented/*"),
        "valid": glob.glob(f"{dataset_path}/valid/nonaugmented/*")
    }
    for dataset_type, dataset_paths in dateset_path_dict.items():

        for dataset_path in dataset_paths:
            # check ROI images are generated
            if os.path.exists(f"{dataset_path}/fixed_rois.h5") and \
                os.path.exists(f"{dataset_path}/moving_rois.h5"):
                filter_problems(dataset_path, dataset_type)

    with open("resources/bad_registration_problems_ALv1.json", "w") as f:
        json.dump(bad_registration_problems_dict, f, indent=4)


def generate_pair_num(config_file_name):

    problem_to_pairnum = dict()
    problem_index = -1

    with open(f"configs/{config_file_name}.yaml", "r") as file:
        config_dict = yaml.safe_load(file)
    test_dataset_paths = config_dict["dataset"]["test"]["dir"]

    for test_dataset_path in test_dataset_paths:

        with h5py.File(f"{test_dataset_path}/moving_images.h5", "r") as f:
            problems = list(f.keys())

        dataset_name = test_dataset_path.split("/")[-1]
        for problem in problems:
            problem_id = f"{dataset_name}/{problem}"
            problem_index += 1
            problem_to_pairnum[problem_id] = problem_index

    with open("resources/problem_to_pairnum_2022-06-14-01_subset_cleaned.json", "w") as f:
        json.dump(problem_to_pairnum, f, indent=4)


if __name__ == "__main__":
    dataset_path = "/data3/prj_register"
    find_problems_without_labels(dataset_path)
    #generate_pair_num("config_label")
