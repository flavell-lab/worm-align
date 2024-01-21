from tqdm import tqdm
from typing import Dict, List
from wormalign.utils import (locate_dataset, write_to_json)
import os
import random


class Sampler:
    """
    Create a .JSON file that keeps a subset of registration problems sampled
    from the problems obtained from the registration graph.
    """
    def __init__(
        self,
        dataset_dict: Dict[str, List[str]],
    ):
        """
        Init.

        :param dataset_dict: dataset dictionary formated as:
            {"train": ["YYYY-MM-DD-X"],
             "valid": ["YYYY-MM-DD-X"]
             "test": ["YYYY-MM-DD-X"]
            }
        """
        self.dataset_dict = dataset_dict
        self.output_dict = {
                "train": dict(),
                "valid": dict(),
                "test": dict()
        }

    def _get_all_problems(
        self,
        dataset_name: str
    ) -> List[str]:
        """
        Read all the problems suitable for registration.
        """
        dataset_path = locate_dataset(dataset_name)
        if os.path.exists(f"{dataset_path}/registration_problems.txt"):
            lines = open(
                f"{dataset_path}/registration_problems.txt", "r").readlines()
            problems = [line.strip().replace(" ", "to") for line in lines]
        else:
            raise FileNotFoundError(
                f"Can't find {dataset_path}/registration_problems.txt")
        return problems

    def _sample_registration_problems(
        self,
        problems: List[str],
        cutoff: int = 600
    ) -> List[str]:
        """
        Sample a subset of registration problems.
        """
        # sort problems by the length of moving and fixed time interval
        interval_to_problems_dict = dict()
        for problem in problems:

            interval = abs(
                        int(problem.split("to")[0]) -
                        int(problem.split("to")[1])
                    )
            if interval not in interval_to_problems_dict.keys():
                interval_to_problems_dict[interval] = [problem]
            else:
                interval_to_problems_dict[interval].append(problem)

        sampled_problems = []
        for interval, problems in interval_to_problems_dict.items():
            if interval > cutoff:
                sampled_problems += random.sample(
                        problems, int(0.8 * len(problems)))
            else:
                sampled_problems += random.sample(
                        problems, int(0.5 * len(problems)))

        return sampled_problems

    def __call__(self, output_file_name, num_problems: int = -1):
        """
        Create a .JSON file that keeps all the regsitration problems.

        :param output_file_name: name of the file to be created

        :example
            >>> dataset_dict = {
            >>>        "train": ["2023-08-07-01"],
            >>>        "valid": ["2023-08-07-16"],
            >>>        "test": ["2022-04-14-04"]
            >>> }
            >>> sampler = Sampler(dataset_dict)
            >>> output_file_name = "registration_problems"
            >>> sampler(output_file_name)
        """
        for dataset_type, dataset_names in self.dataset_dict.items():

            for dataset_name in tqdm(dataset_names):
                problems = self._get_all_problems(dataset_name)
                # sample accordings to the defined scheme if the number of
                # samples per dataset is not specified
                if num_problems == -1:
                    sampled_problems = self._sample_registration_problems(problems)
                else:
                    sampled_problems = random.sample(problems, num_problems)
                self.output_dict[dataset_type][dataset_name] = sampled_problems

        write_to_json(self.output_dict, output_file_name)

