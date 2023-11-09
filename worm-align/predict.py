from datetime import datetime
from deepreg.predict import predict
import time


def test_ddfnet(config_file_path,
                log_directory,
                experiment_name,
                cuda_device,
                checkpoint):

    checkpoint_path = \
        f"{log_directory}/logs_train/{experiment_name}/save/ckpt-{checkpoint}"
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_path = f"logs_predict/{experiment_name}_{experiment_time}"

    predict(
        gpu = cuda_device,
        gpu_allow_growth = True,
        ckpt_path = checkpoint_path,
        split = "test",
        batch_size = 4,
        log_dir = log_directory,
        exp_name = experiment_path,
        config_path = config_file_path,
    )
