from deepreg.predict import unwrapped_predict
from tqdm import tqdm
from wormalign.utils import get_image_T, write_to_json, filter_and_crop
import numpy as np
import os
import tensorflow as tf

DEVICE = 0
GPUS = tf.config.list_physical_devices('GPU')
if GPUS:
    try:
        tf.config.set_visible_devices(GPUS[DEVICE], 'GPU')
    except RuntimeError as e:
        print(e)

def normalize_batched_image(batched_image, eps=1e-7):
    """
    Normalizes each image in a batch to [0, 1] range separately.
    """
    eps = tf.constant(eps, dtype=tf.float32)
    # Calculate the min and max values for each image in the batch
    min_vals = tf.math.reduce_min(batched_image, axis=[1, 2, 3], keepdims=True)
    max_vals = tf.math.reduce_max(batched_image, axis=[1, 2, 3], keepdims=True)

    # Normalize each image separately
    batched_image = batched_image - min_vals
    batched_image = batched_image / tf.maximum(tf.cast(max_vals - min_vals,
        tf.float32), eps)

    return batched_image


def calculate_ncc(moving, fixed):
    """
    Computes the NCC (Normalized Cross-Correlation) of two image arrays
    `moving` and `fixed` corresponding to a registration.
    """
    assert fixed.shape == moving.shape, "Fixed and moving images must have the same shape."

    med_f = np.median(np.max(fixed, axis=2))
    med_m = np.median(np.max(moving, axis=2))

    fixed_new = np.maximum(fixed - med_f, 0)
    moving_new = np.maximum(moving - med_m, 0)

    mu_f = np.mean(fixed_new)
    mu_m = np.mean(moving_new)

    fixed_new = (fixed_new / mu_f) - 1
    moving_new = (moving_new / mu_m) - 1

    numerator = np.sum(fixed_new * moving_new)
    denominator = np.sqrt(np.sum(fixed_new ** 2) * np.sum(moving_new ** 2))

    return numerator / denominator


def compute_centroid_labels(image, max_centroids = 200):

    centroids = np.zeros((max_centroids, 3), dtype=np.int32) - 1

    for val in range(1, max_centroids + 1):
        indices = np.argwhere(image == val)
        if len(indices) > 0:
            centroid_x = np.mean(indices[:, 0])
            centroid_y = np.mean(indices[:, 1])
            centroid_z = np.mean(indices[:, 2])
            centroids[val-1] = [centroid_x, centroid_y, centroid_z]

    return centroids


def register_single_image_pair(
    problem,
    target_image_shape,
    target_label_shape,
    model_ckpt_path,
    model_config_path,
    output_dir,
):
    def read_ch1_problem(
        problem,
        dataset_path="/data3/prj_register/2022-01-06-01_diffnorm_ckpt287",
    ):
        moving_image_path = \
                f"{dataset_path}/ch1_registered/{problem}/euler_registered.nrrd"
        t_fixed = problem.split("to")[1]
        fixed_image_path = f"{dataset_path}/ch1_recropped/{t_fixed}.nrrd"
        fixed_image = get_image_T(fixed_image_path).astype(np.float32)
        moving_image = get_image_T(moving_image_path).astype(np.float32)

        return moving_image, fixed_image

    def read_ch2_problem(
        problem,
        dataset_path="/data3/prj_register/2022-01-06-01_diffnorm_ckpt287",
    ):
        t_moving, t_fixed = problem.split("to")
        t_fixed_4 = t_fixed.zfill(4)
        fixed_image_path = f"{dataset_path}/NRRD_filtered/2022-01-06-01-SWF360-animal1-610LP_t{t_fixed_4}_ch2.nrrd"
        moving_image_path = f"{dataset_path}/Registered/{problem}/euler_transformed.nrrd"
        moving_image = get_image_T(moving_image_path).astype(np.float32)
        fixed_image_T = get_image_T(fixed_image_path).astype(np.float32)
        fixed_image_median = np.median(fixed_image_T)
        fixed_image = filter_and_crop(fixed_image_T, fixed_image_median, target_image_shape)

        return moving_image, fixed_image


    def compute_score(score_class, *args):

        if len(args) == 2:
            return score_class().call(args[0], args[1]).numpy()[0]
        elif len(args) == 1:
            return score_class(
                img_size=target_image_shape
            ).call(args[0]).numpy()[0]

    def reformat(numpy_array):
        return tf.convert_to_tensor(
                numpy_array[np.newaxis, ...].astype(np.float32))

    def _warp_with_ddf(problem, fixed_image, moving_image):
        batched_fixed_image = normalize_batched_image(
            np.expand_dims(fixed_image, axis=0)
        )
        batched_moving_image = normalize_batched_image(
            np.expand_dims(moving_image, axis=0)
        )
        if batched_moving_image.dtype == np.float32:

            ddf_output, pred_fixed_image, model = unwrapped_predict(
                batched_fixed_image,
                batched_moving_image,
                output_dir,
                target_label_shape,
                target_label_shape,
                model = None,
                model_ckpt_path = model_ckpt_path,
                model_config_path = model_config_path,
            )
            raw_ncc = calculate_ncc(
                batched_moving_image.numpy().squeeze(),
                batched_fixed_image.numpy().squeeze()
            )
            ncc = calculate_ncc(
                pred_fixed_image.squeeze(),
                batched_fixed_image.numpy().squeeze()
            )

        return {
            "fixed_image": batched_fixed_image.numpy().squeeze(),
            "moving_image": batched_moving_image.numpy().squeeze(),
            "warped_moving_image": pred_fixed_image.squeeze(),
            "raw_ncc": raw_ncc,
            "ncc": ncc,
        }

    all_outputs = {"ch1": dict(), "ch2": dict()}
    moving_image, fixed_image = read_ch1_problem(problem)
    all_outputs["ch1"] = _warp_with_ddf(problem, fixed_image, moving_image)
    moving_image, fixed_image = read_ch2_problem(problem)
    all_outputs["ch2"] = _warp_with_ddf(problem, fixed_image, moving_image)

    return all_outputs


def find_problems_meet_criteria(full_experiment, control_experiment, ckpt=287):

    def get_swf360_problems(
        dataset_path="/data3/prj_register/2022-01-06-01_diffnorm_ckpt287/ch1_registered"
    ):
        return [name for name in os.listdir(dataset_path) if
                os.path.isdir(os.path.join(dataset_path, name))][::-1]

    def _print_score(experiment, outputs_dict):
        print(
            f"""{experiment} ch1 NCC: {'{:.2f}'.format(outputs_dict['ch1']['ncc'])}
            {experiment} ch2 NCC: {'{:.2f}'.format(outputs_dict['ch2']['ncc'])}"""
        )

    problems = get_swf360_problems()

    ncc_score_dict = dict()

    base = "/data3/prj_register"
    subdirectory = "centroid_labels_augmented_batched_hybrid"
    print(len(problems))
    print(problems[:100])
    for problem in problems:

        full_network_outputs = register_single_image_pair(
            problem,
            target_image_shape=(284, 120, 64),
            target_label_shape=(200, 3),
            model_ckpt_path=f"{base}/{full_experiment}/{subdirectory}/save/ckpt-{ckpt}",
            model_config_path=f"{base}/{full_experiment}/config_batch.yaml",
            output_dir="")

        control_network_outputs = register_single_image_pair(
            problem,
            target_image_shape=(284, 120, 64),
            target_label_shape=(200, 3),
            model_ckpt_path=f"{base}/{control_experiment}/{subdirectory}/save/ckpt-{ckpt}",
            model_config_path=f"{base}/{control_experiment}/config_batch.yaml",
            output_dir="",
        )

        condition = control_network_outputs["ch1"]["ncc"] < 0.8 and \
            control_network_outputs["ch2"]["ncc"] > 0.9 and \
            full_network_outputs["ch1"]["ncc"] > 0.9 and \
            full_network_outputs["ch2"]["ncc"] > 0

        _print_score(full_experiment, outputs_dict)
        _print_score(control_experiment, outputs_dict)

        if condition:
            ncc_score_dict[problem] = {
                "full": [
                    full_network_outputs["ch1"]["ncc"],
                    full_network_outputs["ch2"]["ncc"]
                ],
                "control": [
                    control_network_outputs["ch1"]["ncc"],
                    control_network_outputs["ch2"]["ncc"]
                ]
            }
            write_to_json(ncc_dict, "swf360_condition_met", "scores")

find_problems_meet_criteria("2024-01-30-train", "2024-03-08-train")