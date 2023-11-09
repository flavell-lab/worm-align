from datetime import datetime
from deepreg.predict import predict
from deepreg.train import train
import copy
import os
import yaml


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


class QuotedStr(str):
    pass


def quoted_str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


def write_config_file(dataset_directory,
                   train_datasets,
                   valid_datasets,
                   test_datasets,
                   image_dimension,
                   outfile_path,
                   max_epochs=300,
                   save_period=1):

    yaml.add_representer(QuotedStr, quoted_str_presenter)

    configuration = {
        "dataset": {
            "train": {
                "dir": [],
                "format": QuotedStr("h5"),
                "labeled": False
            },
            "valid": {
                "dir": [],
                "format": QuotedStr("h5"),
                "labeled": False
            },
            "test": {
                "dir": [],
                "format": QuotedStr("h5"),
                "labeled": False
            },
            "type": "paired",
            "moving_image_shape": [208, 96, 56],
            "fixed_image_shape": [208, 96, 56],
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
                    "weight": 1.0
                },
                "regularization": {
                    "weight": 0.2,
                    "name": QuotedStr("bending")
                }
            },
            "preprocess": {
                "data_augmentation": {
                    "name": QuotedStr("affine")
                },
                "batch_size": 4,
                "shuffle_buffer_num_batch": 2,
                "num_parallel_calls": -1
            },
            "optimizer": {
                "name": QuotedStr("Adam"),
                "learning_rate": 1.0e-3
            },
            "epochs": max_epochs,
            "save_period": save_period
        }
    }

    configuration['dataset']['train']['dir'] = [
            QuotedStr(f"{dataset_directory}/train/{train_dataset}")
                for train_dataset in train_datasets
    ]
    configuration['dataset']['valid']['dir'] = [
            QuotedStr(f"{dataset_directory}/valid/{valid_dataset}")
                for valid_dataset in valid_datasets
    ]
    configuration['dataset']['test']['dir'] = [
            QuotedStr(f"{dataset_directory}/test/{test_dataset}")
                for test_dataset in test_datasets
    ]
    with open(outfile_path, 'w') as f:
        yaml.dump(configuration, f, default_flow_style=None)


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
    dataset_directory = "/home/alicia/data_personal/test_preprocess"
    train_datasets = ["2022-01-09-01", "2022-01-23-04"]
    valid_datasets = ["2022-07-26-01", "2022-07-20-01"]
    test_datasets = ["2022-08-02-01", "2022-04-18-04"]
    image_dimension = [208, 96, 56]
    outfile_path = "configs/test_config.yaml"
    write_config_file(dataset_directory,
                   train_datasets,
                   valid_datasets,
                   test_datasets,
                   image_dimension,
                   outfile_path,
                   max_epochs=300,
                   save_period=1)
