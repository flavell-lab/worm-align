from datetime import datetime
from deepreg.train import train
import copy
import os


def train_ddf(config_file_path,
              log_directory,
              experiment_name,
              cuda_device,
              checkpoint=None):

    if not os.path.exists(log_directory):
        os.mkdir(log_directory)

    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_path = f"logs_train/{experiment_name}_{experiment_time}"

    if checkpoint is not None:
        checkpoint_path = \
            f"{log_directory}/logs_train/{experiment_name}/save/ckpt-{checkpoint}"
    else:
        checkpoint_path = ""

    train(
        gpu = cuda_device,
        config_path = config_file_path,
        gpu_allow_growth = True,
        ckpt_path = checkpoint_path,
        log_dir = log_directory,
        exp_name = experiment_path,
    )


if __name__ == "__main__":

    """
    config_file_path = "/home/alicia/notebook/register/configs/config_raw-ddf_filter-crop-v1_size-v2.yaml"
    log_directory = "/home/alicia/data_personal/test_train_ddf"
    experiment_name = "raw_ddf"
    cuda_device = "2"
    train_ddf(config_file_path,
              log_directory,
              experiment_name,
              cuda_device,
              checkpoint=None)
    """
    config_file_path = "/home/alicia/notebook/worm-align/worm-align/configs/config_euler-gpu-ddf_resize-v1_size-v1.yaml"
    log_directory = "/home/alicia/data_personal/regnet_ckpt"
    experiment_name = "euler-gpu_resize-v1_size-v1_aug-v1_reg-nonrigid-w0.02"
    cuda_device = "3"

    #configs_home = "/home/alicia/notebook/register/configs"
    #config_file_path = f"{configs_home}/config_euler-gpu-ddf_resize-v1_size-v1.yaml"
    #experiment_name = "euler-gpu_resize-v1_size-v1_aug-affine"
    train_ddf(config_file_path,
              log_directory,
              experiment_name,
              cuda_device)

