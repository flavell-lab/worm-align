from deepreg.predict import unwrapped_predict
from tqdm import tqdm
from wormalign.utils import get_image_T, write_to_json, filter_and_crop
import deepreg.model.layer as layer
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

def read_problem_h5(
    problem,
    channel_num,
):
    dataset_path = \
    f"/data3/prj_register/ALv7_swf360_ch{channel_num}/train/nonaugmented/2022-03-30-02"
    with h5py.File(f"{dataset_path}/fixed_images.h5", "r") as f:
        fixed_image = f[problem][:].astype(np.float32)

    with h5py.File(f"{dataset_path}/moving_images.h5", "r") as f:
        moving_image = f[problem][:].astype(np.float32)

    return moving_image, fixed_image

def register_single_image_pair(
    problem,
    target_image_shape,
    target_label_shape,
    model_ckpt_path,
    model_config_path,
    output_dir,
):
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

    def _warp_ch2_with_ddf(problem, fixed_image, moving_image):

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
            "ddf": ddf_output
        }

    def _warp_ch1_with_ddf(problem, ch2_ddf, fixed_image, moving_image):

        batched_fixed_image = normalize_batched_image(
            np.expand_dims(fixed_image, axis=0)
        )
        batched_moving_image = normalize_batched_image(
            np.expand_dims(moving_image, axis=0)
        )
        warping = layer.Warping(
            fixed_image_size=batched_fixed_image.shape[1:4],
            batch_size=1
        )
        warped_moving_image = warping(inputs=[ch2_ddf, batched_moving_image])
        raw_ncc = calculate_ncc(
            batched_moving_image.numpy().squeeze(),
            batched_fixed_image.numpy().squeeze()
        )
        ncc = calculate_ncc(
            warped_moving_image.numpy().squeeze(),
            batched_fixed_image.numpy().squeeze()
        )

        return {
            "fixed_image": fixed_image,
            "moving_image": moving_image,
            "warped_moving_image": warped_moving_image.numpy().squeeze(),
            "raw_ncc": raw_ncc,
            "ncc": ncc
        }

    all_outputs = {"ch1": dict(), "ch2": dict()}

    ch2_moving_image, ch2_fixed_image = read_problem_h5(problem, 2)
    all_outputs["ch2"] = _warp_ch2_with_ddf(problem, ch2_fixed_image, ch2_moving_image)

    ch1_moving_image, ch1_fixed_image = read_problem_h5(problem, 1)
    all_outputs["ch1"] = _warp_ch1_with_ddf(problem, all_outputs["ch2"]["ddf"],
            ch1_fixed_image, ch1_moving_image)

    return all_outputs

def find_problems_meet_criteria(full_experiment, control_experiment, ckpt=287):

    def get_swf360_problems():
        return json.load(
                open("resources/registration_problems_ALv7-swf360.json"))["train"]["2022-03-30-02"]

    def _print_score(experiment, outputs_dict):
        print(
            f"""{experiment} ch1 NCC: {'{:.2f}'.format(outputs_dict['ch1']['ncc'])}
            {experiment} ch2 NCC: {'{:.2f}'.format(outputs_dict['ch2']['ncc'])}"""
        )

    ncc_score_dict = dict()
    every_ncc_score_dict = dict()
    """
    with open(f"scores/swf360_all_ncc_score_{control_experiment}.json", "r") as f:
        every_ncc_score_dict = json.load(f)
        problems_searched = list(every_ncc_score_dict.keys())
    """
    problems_searched = []
    base = "/data3/prj_register"
    subdirectory = "centroid_labels_augmented_batched_hybrid"

    problems = get_swf360_problems()
    problems_to_search = list(set(problems) - set(problems_searched))
    print(f"remaining problems: {len(problems_to_search)}")

    for problem in problems_to_search:

        full_network_outputs = register_single_image_pair(
            problem,
            target_image_shape=(284, 120, 64),
            target_label_shape=(200, 3),
            model_ckpt_path=f"{base}/{full_experiment}/{subdirectory}/save/ckpt-{ckpt}",
            model_config_path=f"{base}/{full_experiment}/config_batch.yaml",
            output_dir="/data3/prj_register/2024-02-15_debug"
        )

        control_network_outputs = register_single_image_pair(
            problem,
            target_image_shape=(284, 120, 64),
            target_label_shape=(200, 3),
            model_ckpt_path=f"{base}/{control_experiment}/{subdirectory}/save/ckpt-{ckpt}",
            model_config_path=f"{base}/{control_experiment}/config_batch.yaml",
            output_dir="/data3/prj_register/2024-02-15_debug"
        )
        condition = control_network_outputs["ch1"]["ncc"] < 0.8 and \
            control_network_outputs["ch2"]["ncc"] > 0.9 and \
            full_network_outputs["ch1"]["ncc"] > 0.9 and \
            full_network_outputs["ch2"]["ncc"] > 0

        _print_score(full_experiment, full_network_outputs)
        _print_score(control_experiment, control_network_outputs)

        every_ncc_score_dict[problem] = {
            "full": [
                full_network_outputs["ch1"]["ncc"],
                full_network_outputs["ch2"]["ncc"]
            ],
            "control": [
                control_network_outputs["ch1"]["ncc"],
                control_network_outputs["ch2"]["ncc"]
            ]
        }
        write_to_json(
            every_ncc_score_dict,
            f"ALv7-swf360_all_ncc_score_{control_experiment}",
            "scores"
        )
        """
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
            write_to_json(
                ncc_score_dict,
                f"swf360_ncc_condition_met_{control_experiment}",
                "scores"
            )
        """
# full network: "2024-01-30-train"
# no-label network: '2024-03-08-train"
# no-regualrization network: "2024-03-15-train-2I"
# no-image network: "2024-05-02-train"
# ALv6 - "2022-03-30-01" - SWF360
# ALv7 - "2022-03-30-02" - SWF360
# ALv8 - "2022-03-31-01" - SWF360
#find_problems_meet_criteria("2024-01-30-train", "2024-05-02-train")
find_problems_meet_criteria("2024-01-30-train", "2024-03-08-train")

