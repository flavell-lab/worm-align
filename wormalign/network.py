from datetime import datetime
from deepreg.predict import predict
from deepreg.train import train
from typing import List, Optional, Union
import copy
import glob
import os
import yaml


class QuotedStr(str):
    pass

def quoted_str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

class ConfigGenerator:
    """
    Class that generates the configuration file for training and testing the
    DDF netwok
    """

    def __init__(
        self,
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
    ):
        """
        Init.

        :param dataset_path: the path to train, valid, and test datasets
        :param image_shape: the shape of images (x_dim, y_dim, z_dim)
        :param method: the registration type; one of `ddf`, `dvf`,
            `conditional`
        :param image_loss: loss between the fixed image and predicted fixed
            image (warped moving image); one of `lncc`, `ssd` or `gmi`
        :param image_loss_weight: weight of the image loss
        :param regularization_loss: loss on predicted dense displacement field
        :param regularization_loss_weight: weight of the regularizer
        :param augmentation: augmention applied to training images
        :param batch_size: the number of samples per step for prediction; if
            using multiple GPUs, i.e. n GPUs, each GPU will have mini batch size
            batch_size / n
        :param optimizer: an algorithm that adjusts the parameters of a model
            to minimize the loss function during training; chosen from
            `tf.keras.optimizers`; e.g. `Adam`, `SGD`
        :param learning_rate: the size of the steps taken during the
            optimization process to minimize the loss function
        :param max_epochs: the number of epochs to train the network for
        :param save_period: the save frequency--the model will be saved every
            `save_period` epochs
        """
        train_datasets = glob.glob(f"{dataset_path}/train/nonaugmented/*")
        valid_datasets = glob.glob(f"{dataset_path}/valid/*")
        test_datasets = glob.glob(f"{dataset_path}/test/*")

        self.configuration = {
            "dataset": {
                "train": {
                    "dir": [
                        QuotedStr(train_dataset) for train_dataset in
                        train_datasets
                    ],
                    "format": QuotedStr("h5"),
                    "labeled": False
                },
                "valid": {
                    "dir": [
                        QuotedStr(valid_dataset) for valid_dataset in
                        valid_datasets
                    ],
                    "format": QuotedStr("h5"),
                    "labeled": False
                },
                "test": {
                    "dir": [
                        QuotedStr(test_dataset) for test_dataset in
                        test_datasets
                    ],
                    "format": QuotedStr("h5"),
                    "labled": False
                },
                "type": "paired",
                "moving_image_shape": copy.deepcopy(image_shape),
                "fixed_image_shape": copy.deepcopy(image_shape),
            },
            "train": {
                "method": QuotedStr("ddf"),
                "backbone": {
                    "name": QuotedStr("local"),
                    "num_channel_initial": 16,
                    "extract_levels": [0, 1, 2, 3]
                },
                "loss": {
                    "image": {
                        "name": QuotedStr("lncc"),
                        "kernel_size": 16,
                        "weight": image_loss_weight,
                    },
                    "regularization": {
                        "weight": regularization_loss_weight,
                        "name": QuotedStr(regularization_loss),
                        "img_size": copy.deepcopy(image_shape),
                        "hybrid": False
                    }
                },
                "preprocess": {
                    "data_augmentation": {
                        "name": QuotedStr(augmentation)
                    },
                    "batch_size": batch_size,
                    "shuffle_buffer_num_batch": 2,
                    "num_parallel_calls": -1
                },
                "optimizer": {
                    "name": QuotedStr(optimizer),
                    "learning_rate": learning_rate,
                },
                "epochs": max_epochs,
                "save_period": save_period
            }
        }

    def write_to_file(
        self,
        file_name: str
    ):
        """
        Write the configuration settings to a .YAML file

        :param file_name: name of the .YAML file to be generated

        :example
            >>> dataset_path = "/home/alicia/data_personal/datasets"
            >>> image_shape = [208, 96, 56]
            >>> config_gen = ConfigGenerator(dataset_path, image_shape)
            >>> config_gen.write_to_file("config_file")
        """
        self._ensure_directory_exists("configs")
        yaml.add_representer(QuotedStr, quoted_str_presenter)
        with open(f"configs/{file_name}.yaml", 'w') as f:
            yaml.dump(
                self.configuration,
                f,
                default_flow_style = False
            )

    def _ensure_directory_exists(self, path: str):
        """
        Create the given directory if it does not already exist

        :param path: directory to create if does not exist
        """
        if not os.path.exists(path):
            os.mkdir(path)


class DDFNetworkTrainer:

    def __init__(
        self,
        config_file_path: str,
        log_directory: str,
        experiment_name: str,
        cuda_device: str,
        checkpoint: Optional[int] = None,
        gpu_growth: bool = True
    ):
        """
        Init.

        :param config_file_path: path to the configuration file
        :param log_directory: directory to saved the warped moving images,
            fixed images, DDF arrays
        :param experiment_name: name of the experiment
        :param cuda_device: CUDA device; one of "0", "1", "2", "3"
        :param checkpoint: checkpoint from the trained model to be applied on
            the unseen images during testing
        :param gpu_growth: whether or not allowing using more than one GPU
        """
        self.config_file_path = config_file_path
        self.log_directory = log_directory
        self.experiment_name = experiment_name
        self.cuda_device = cuda_device
        self.checkpoint = checkpoint
        self.gpu_growth = gpu_growth

    def __call__(self):

        """
        Train the DDF network.
        """
        self._ensure_directory_exists(self.log_directory)
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_path = \
            f"logs_train/{self.experiment_name}_{experiment_time}"
        if self.checkpoint:
            checkpoint_path = \
                f"{self.log_directory}/logs_train/{self.experiment_name}/save/ckpt-{self.checkpoint}"
        else:
            checkpoint_path = ""

        train(
            gpu = self.cuda_device,
            config_path = self.config_file_path,
            gpu_allow_growth = self.gpu_growth,
            ckpt_path = checkpoint_path,
            log_dir = self.log_directory,
            exp_name = experiment_path,
        )

    def _ensure_directory_exists(self, path: str):
        """
        Create the given directory if it does not already exist

        :param path: directory to create if does not exist
        """
        if not os.path.exists(path):
            os.mkdir(path)


class DDFNetworkTester:

    def __init__(
        self,
        config_file_path: str,
        log_directory: str,
        trained_ddf_model: str,
        experiment_name: str,
        cuda_device: str,
        checkpoint: int,
        gpu_growth: bool = True
    ):
        """
        Init.

        :param config_file_path: path to the configuration file
        :param log_directory: directory to saved the warped moving images,
            fixed images, DDF arrays
        :param trained_ddf_model: name of the trained DDF model
        :param experiment_name: name of the experiment
        :param cuda_device: CUDA device; one of "0", "1", "2", "3"
        :param checkpoint: checkpoint from the trained model to be applied on
            the unseen images during testing
        :param gpu_growth: whether or not allowing using more than one GPU
        """
        self.config_file_path = config_file_path
        self.log_directory = log_directory
        self.trained_ddf_model = trained_ddf_model
        self.experiment_name = experiment_name
        self.cuda_device = cuda_device
        self.checkpoint = checkpoint
        self.gpu_growth = gpu_growth
        self.checkpoint_path = \
            f"{self.log_directory}/logs_train/{self.trained_ddf_model}/save/ckpt-{self.checkpoint}"

    def __call__(
        self,
        report_time: bool = True
    ):
        """
        Apply the DDF from the trained model on unseen images.

        :param report_time: whether or not to report time taken for testing
        """
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_path = f"logs_predict/{experiment_name}_{experiment_time}"
        start_time = time.time()
        predict(
            gpu = self.cuda_device,
            gpu_allow_growth = self.gpu_growth,
            ckpt_path = self.checkpoint_path,
            split = "test",
            batch_size = 1,
            log_dir = self.log_directory,
            exp_name = experiment_path,
            config_path = self.config_file_path,
        )
        end_time = time.time()
        if report_time:
            print(f"Time taken: {end_time - start_time} seconds")
