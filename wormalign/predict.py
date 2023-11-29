from datetime import datetime
from deepreg.predict import predict
import time


def test_ddfnet(config_file_path,
                log_directory,
                trained_model,
                experiment_name,
                cuda_device,
                checkpoint):

    checkpoint_path = \
        f"{log_directory}/logs_train/{trained_model}/save/ckpt-{checkpoint}"
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_path = f"logs_predict/{experiment_name}_{experiment_time}"

    start_time = time.time()
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
    end_time = time.time()
    print(f"total time taken: {end_time - start_time} seconds")


if __name__ == "__main__":

    config_file_path = "configs/config_euler-gpu-ddf_resize-v1_size-v1.yaml"
    #    "/home/alicia/notebook/register/configs/config_euler-gpu-ddf_resize-v1_size-v1.yaml"
    log_directory = "/home/alicia/data_personal/regnet_ckpt"
    experiment_name = "euler-gpu_resize-v1_size-v1_aug_reg-nonrigid"
    cuda_device = "2"
    checkpoint = 2452
    trained_model = f"{experiment_name}_20231115-141643"
    test_ddfnet(config_file_path,
                log_directory,
                trained_model,
                experiment_name,
                cuda_device,
                checkpoint)
