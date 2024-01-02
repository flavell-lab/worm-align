import h5py
import json
import yaml

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

    with open("resources/problem_to_pairnum.json", "w") as f:
        json.dump(problem_to_pairnum, f, indent=4)

