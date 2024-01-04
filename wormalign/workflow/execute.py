from wormalign.sample import Sampler
from wormalign.preprocess import RegistrationProcessor
from wormalign.ConfigGenerator import ConfigGenerator
from typing import List, Tuple

"""
preparing training →
    generate registration problems;
    preprocess problems;
    generate config filew
training
predicting
jupyter notebook - evaluate outputs
"""

def prepare(
    # for sampling problems
    train_datasets: List[str],
    valid_datasets: List[str],
    test_datasets: List[str],
    target_image_shape: Tuple[int, int, int],
    # for preprocessing images
    save_directory: str,
    batch_size: int,
    device_name: str,
    downsample_factor: int,
):

    dataset_dict = {
        "train": train_datasets,
        "valid": valid_datasets,
        "test:": test_datasets
    }
    sampler = Sampler(dataset_dict)
    sampler("registration_problems.json")

    # donwsample factor needs to be divisible by each image dimension
    processor = RegistrationProcessor(
            target_image_shape,
            save_directory,
            batch_size,
            device_name,
            downsample_factor
    )
    processor.process_datasets()

def train(

):
    """
    # for write config.yaml
    config_writer = ConfigGenerator(
        dataset_path: str,
        image_shape: List[int],
        method: str = "ddf",
        image_loss: str = "lncc",
        image_loss_weight: int = 1,
        regularization_loss: str = "nonrigid",
        regularization_loss_weight: Union[dict, int] = 0.02,
        augmentation: str = "affine",
        batch_size: int = 4,
        optimizer: str = "Adam",
        learning_rate: float = 1.0e-3,
        max_epochs: int = 20000,
        save_period: int = 1,
   )"""
   pass


