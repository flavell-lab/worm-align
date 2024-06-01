from deepreg.predict import unwrapped_predict
from tqdm import tqdm
from wormalign.benchmark import CentroidDistScore, calculate_ncc
from wormalign.utils import write_to_json
import deepreg.model.layer as layer
import h5py
import numpy as np
import tensorflow as tf
import yaml

DEVICE = 0
GPUS = tf.config.list_physical_devices('GPU')
if GPUS:
    try:
        tf.config.set_visible_devices(GPUS[DEVICE], 'GPU')
    except RuntimeError as e:
        print(e)

def compute_centroid_labels(image, max_centroids = 200):
    """
    Get the x, y, z coordinates of the centers of all the neurons on a given image.
    """
    centroids = np.zeros((max_centroids, 3), dtype=np.int32) - 1

    for val in range(1, max_centroids + 1):
        indices = np.argwhere(image == val)
        if len(indices) > 0:
            centroid_x = np.mean(indices[:, 0])
            centroid_y = np.mean(indices[:, 1])
            centroid_z = np.mean(indices[:, 2])
            centroids[val-1] = [centroid_x, centroid_y, centroid_z]

    return centroids


def normalize_batched_image(batched_image, eps=1e-7):
    """
    Normalizes each image in a batch to [0, 1] range separately.
    """
    # calculate the min and max values for each image in the batch
    min_vals = tf.math.reduce_min(batched_image, axis=[1, 2, 3], keepdims=True)
    max_vals = tf.math.reduce_max(batched_image, axis=[1, 2, 3], keepdims=True)

    # normalize each image separately
    batched_image = batched_image - min_vals
    batched_image = batched_image / tf.maximum(max_vals - min_vals, eps)

    return batched_image

def bulk_register(
    target_image_shape,
    target_label_shape,
    model_ckpt_path,
    model_config_path,
    output_dir,
    dataset_type,
    experiment="full",
):
    """
    Register on problems from test datasets.
    """

    def read_problem(file_name, problem):
        return h5py.File(f"{dataset_dir}/{file_name}.h5", "r")[problem][:]

    def get_problems(file_name):
        return list(h5py.File(f"{dataset_dir}/{file_name}.h5", "r").keys())

    def compute_score(score_class, *args):

        if len(args) == 2:
            return score_class().call(args[0], args[1]).numpy()[0]
        elif len(args) == 1:
            return score_class(
                img_size=target_image_shape
            ).call(args[0]).numpy()[0]

    def reformat(numpy_array):
        return tf.convert_to_tensor(numpy_array[np.newaxis, ...].astype(np.float32))

    centroid_distance = dict()
    ncc_scores = dict()

    with open(model_config_path, "r") as f:
        dataset_dirs = yaml.safe_load(f)["dataset"][dataset_type]["dir"]

    for dataset_dir in dataset_dirs:

        dataset_name = dataset_dir.split("/")[-1]

        centroid_distance[dataset_name] = dict()
        ncc_scores[dataset_name] = dict()

        problems = list(h5py.File(f"{dataset_dir}/fixed_images.h5", "r").keys())[:100]

        for problem in tqdm(problems):

            batched_fixed_image = normalize_batched_image(
                np.expand_dims(read_problem("fixed_images", problem), axis=0).astype(np.float32)
            )
            batched_moving_image = normalize_batched_image(
                np.expand_dims(read_problem("moving_images", problem), axis=0).astype(np.float32)
            )

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

            moving_roi = read_problem("moving_rois", problem)
            fixed_roi = read_problem("fixed_rois", problem)

            moving_image_roi_tf = tf.cast(tf.expand_dims(moving_roi, axis=0), dtype=tf.float32)
            warping = layer.Warping(fixed_image_size = target_image_shape, interpolation = "nearest", batch_size=1)

            ddf = ddf_output[0, :, :, :]
            warped_moving_image_roi_tf = warping(inputs = [ddf, moving_image_roi_tf])
            warped_moving_roi = warped_moving_image_roi_tf.numpy()[0]

            warped_moving_centroids = reformat(compute_centroid_labels(warped_moving_roi))
            fixed_centroids = reformat((compute_centroid_labels(fixed_roi)))
            moving_centroids = reformat((compute_centroid_labels(moving_roi)))

            centroid_distance[dataset_name][problem] = compute_score(
                CentroidDistScore,
                fixed_centroids,
                warped_moving_centroids
            )
            ncc_scores[dataset_name][problem] = calculate_ncc(
                pred_fixed_image.squeeze(),
                batched_fixed_image.numpy().squeeze()
            )
            print(f"NCC for {dataset_name}/{problem}: {ncc_scores[dataset_name][problem]}")
            print(f"dist for {dataset_name}/{problem}: {centroid_distance[dataset_name][problem]}")

        write_to_json(centroid_distance, f"{experiment}_centroid_dist_on_test", "scores")
        write_to_json(ncc_scores, f"{experiment}_ncc_on_test", "scores")

experiment = "2024-03-30-train"
model_ckpt_path = \
f"/data3/prj_register/{experiment}/centroid_labels_augmented_batched_hybrid/save/ckpt-300"
model_config_path = f"/data3/prj_register/{experiment}/config_batch.yaml"
target_image_shape = (284, 120, 64)
target_label_shape = (200, 3)
output_dir = "/data3/prj_register/2024-02-15_debug"

bulk_register(
    target_image_shape,
    target_label_shape,
    model_ckpt_path,
    model_config_path,
    output_dir,
    dataset_type ="test",
    experiment="register_everything"
)
