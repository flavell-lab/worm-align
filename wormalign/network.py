from datetime import datetime
from deepreg.predict import predict
from deepreg.train import train
from typing import Optional
import os
import time


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

        :example
            >>> config_file_path = "configs/config.yaml"
            >>> log_directory = "/home/alicia/data_personal"
            >>> experiment_name = "example"
            >>> cuda_device = "1"
            >>> network_trainer = DDFNetworkTrainer(
            >>>    config_file_path,
            >>>     log_directory,
            >>>    experiment_name,
            >>>    cuda_device
            >>> )
            >>> # start network training
            >>> network_trainer()
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
        #self.checkpoint_path = \
        #    f"{self.log_directory}/logs_train/{self.trained_ddf_model}/save/ckpt-{self.checkpoint}"
        self.checkpoint_path = \
        f"{self.log_directory}/{self.trained_ddf_model}/save/ckpt-{self.checkpoint}"

    def __call__(
        self,
        report_time: bool = True
    ):
        """
        Apply the DDF from the trained model on unseen images.

        :param report_time: whether or not to report time taken for testing

        :example
            >>> config_file_path = "configs/config.yaml"
            >>> log_directory = "/home/alicia/data_personal"
            >>> trained_ddf_model = "model0"
            >>> experiment_name = "example_test"
            >>> cuda_device = "1"
            >>> checkpoint = 1000
            >>> network_tester = DDFNetworkTester(
            >>>    config_file_path,
            >>>    log_directory,
            >>>    trained_ddf_model,
            >>>    experiment_name,
            >>>    cuda_device,
            >>>    checkpoint
            >>> )
            >>> # start network training
            >>> network_tester()
        """
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_path = f"logs_predict/{self.experiment_name}_{experiment_time}"
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
