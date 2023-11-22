import json
import h5py

def generate_pair_num(datasets, dataset_path):

    problem_to_pairnum = dict()
    problem_index = -1
    for dataset in datasets:
        with h5py.File(f"{dataset_path}/{dataset}/moving_images.h5", "r") as f:
            problems = list(f.keys())
        for problem in problems:
            problem_id = f"{dataset}/{problem}"
            problem_index += 1
            problem_to_pairnum[problem_id] = problem_index

    with open("resources/problem_to_pairnum.json", "w") as f:
        json.dump(problem_to_pairnum, f, indent=4)

if __name__ == "__main__":
    datasets = ["2022-04-14-04", "2022-04-18-04", "2022-08-02-01"]
    dataset_path = \
    "/home/alicia/data_personal/regnet_dataset/euler-gpu_size-v1/test"
    generate_pair_num(datasets, dataset_path)
