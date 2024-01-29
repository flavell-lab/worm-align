from numpy.typing import NDArray
from tqdm import tqdm
from wormalign.benchmark import (CentroidDistScore,
        LocalNormalizedCrossCorrelation, GlobalNormalizedCrossCorrelation,
        NonRigidPenalty)
from wormalign.utils import write_to_json
from wormalign.warp import ImageWarper
from wormalign.warp_label import LabelWarper
import glob
import numpy as np
import h5py
import tensorflow as tf


def get_all_scores(test_dataset_path, network_output_path, target_image_shape):

    image_warper = ImageWarper(
            network_output_path,
            None, # `dataset_name` to be set later
            None, # `registration_problem` to be set later
            target_image_shape
    )
    label_warper = LabelWarper(
            None,
            None,
            target_image_shape,
    )
    label_warper.ddf_directory = network_output_path
    #all_dataset_paths = glob.glob(f"{test_dataset_path}/*")
    all_dataset_paths = [test_dataset_path]

    centroid_distance = dict()
    image_lncc = dict()
    image_gncc = dict()
    nonrigid_penalty = dict()

    def compute_score(score_class, *args):
        if len(args) == 2:
            return score_class().call(args[0], args[1]).numpy()[0]
        elif len(args) == 1:
            return score_class(
                    img_size=target_image_shape
            ).call(args[0]).numpy()[0]

    def get_scores_for_one_dataset(dataset_path):

        problems = list(h5py.File(f"{dataset_path}/fixed_images.h5").keys())
        dataset_name = dataset_path.split("/")[-1].split("_")[0]
        image_warper.dataset_name = dataset_name
        label_warper.dataset_name = dataset_name

        centroid_distance[dataset_name] = dict()
        image_lncc[dataset_name] = dict()
        image_gncc[dataset_name] = dict()
        nonrigid_penalty[dataset_name] = dict()

        for problem in tqdm(problems):

            image_warper.registration_problem = problem
            label_warper.registration_problem = problem
            # need to manually update `LabelWarper.pair_num`
            label_warper.pair_num = label_warper.pairnum_dict[label_warper.problem_id]

            output_image_dict = image_warper.get_network_outputs()
            output_roi_label_dict = label_warper.get_image_roi()

            ddf = reformat(output_image_dict["ddf"])
            warped_moving_image = reformat(output_image_dict["warped_moving_image"])
            fixed_image = reformat(output_image_dict["fixed_image"])
            moving_image = reformat(output_image_dict["moving_image"])

            """
            warped_moving_centroids = reformat(
                compute_centroids_3d(
                    output_roi_label_dict["warped_moving_image_roi"], 200
                )
            )
            fixed_centroids = reformat(
                compute_centroids_3d(
                    output_roi_label_dict["fixed_image_roi"], 200
                )
            )
            moving_centroids = reformat(
                compute_centroids_3d(
                    output_roi_label_dict["moving_image_roi"], 200
                )
            )
            centroid_distance[dataset_name][problem] = compute_score(
                    CentroidDistScore, fixed_centroids, moving_centroids)
            """
            image_lncc[dataset_name][problem] = compute_score(
                    LocalNormalizedCrossCorrelation, fixed_image, warped_moving_image)

            image_gncc[dataset_name][problem] = compute_score(
                    GlobalNormalizedCrossCorrelation, fixed_image, warped_moving_image)
            """
            nonrigid_penalty[dataset_name][problem] = compute_score(
                    NonRigidPenalty, ddf)
            """
    for dataset_path in all_dataset_paths:
        get_scores_for_one_dataset(dataset_path)

    #write_to_json(centroid_distance, "centroid_distance_A1", "scores")
    write_to_json(image_lncc, "image_lncc", "scores")
    write_to_json(image_gncc, "image_gncc", "scores")
    #write_to_json(nonrigid_penalty, "nonrigid_penalty_ALv1", "scores")

def reformat(numpy_array):
    return tf.convert_to_tensor(numpy_array[np.newaxis, ...].astype(np.float32))

def compute_centroids_3d(
    image: NDArray[np.int32],
    max_centroids: int
) -> NDArray[np.float32]:
    """
    Compute the centroids of all pixels with each unique value in a 3D image.

    :param image: A 3D numpy array representing the image with dimensions
            (x, y, z).
    :param max_val: the maximum number of centroids contained in a given label;
        included because the network expects all labels to have shape 
        (max_val, 3)

    :return: A Nx3 numpy array, where N is the maximum value in the image plus
            one. Each row corresponds to the centroid coordinates (x, y, z) for
            each value.
    """
    centroids = np.zeros((max_centroids, 3), dtype=np.int32) - 1  # Initialize the centroids array

    for val in range(1, max_centroids + 1):
        # Find the indices of pixels that have the current value
        indices = np.argwhere(image == val)

        # Compute the centroid if the value is present in the image
        if len(indices) > 0:
            centroid_x = np.mean(indices[:, 0])  # x-coordinate
            centroid_y = np.mean(indices[:, 1])  # y-coordinate
            centroid_z = np.mean(indices[:, 2])  # z-coordinate
            centroids[val-1] = [centroid_x, centroid_y, centroid_z]

    return centroids


def calculate_gncc(fixed, moving):

    mu_f = np.mean(fixed)
    mu_m = np.mean(moving)
    a = np.sum(abs(fixed - mu_f) * abs(moving - mu_m))
    b = np.sqrt(np.sum((fixed - mu_f) ** 2) * np.sum((moving - mu_m) ** 2))
    return a / b


def calculate_ncc(fixed, moving):
    assert fixed.shape == moving.shape

    med_f = np.median(np.max(fixed, axis=2))
    med_m = np.median(np.max(moving, axis=2))
    fixed_new = np.maximum(fixed - med_f, 0)
    moving_new = np.maximum(moving - med_m, 0)

    mu_f = np.mean(fixed_new)
    mu_m = np.mean(moving_new)
    fixed_new = fixed_new / mu_f - 1
    moving_new = moving_new / mu_m - 1
    numerator = np.sum(fixed_new * moving_new)
    denominator = np.sqrt(np.sum(fixed_new ** 2) * np.sum(moving_new ** 2))

    return numerator / denominator


def calculate_dice_score(fixed, moving, threshold = 0):

    binarized_moving  = (moving > threshold).astype(np.uint)
    binarized_fixed = (fixed > threshold).astype(np.uint)
    num = binarized_moving * binarized_fixed
    denom = binarized_moving + binarized_fixed
    dice_score = np.sum(num) / np.sum(denom) * 2

    return dice_score
