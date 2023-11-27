from tqdm import tqdm
import h5py
import numpy as np
import random
import os


class RandomRotate:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be
    either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent
    between raw and labeled datasets, otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across
    (1,2) axis)
    """

    def __init__(self, random_state, axes=[(1,2)],  **kwargs):

        self.random_state = random_state
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.axes = axes

    def __call__(self, image):

        axis = self.axes[self.random_state.randint(len(self.axes))]
        assert image.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        k = self.random_state.choice([0, 2])
        # rotate k times around a given plane
        if image.ndim == 3:
            image = np.rot90(image, k, axis)
        else:
            channels = [np.rot90(image[c], k, axis) for c in range(image.shape[0])]
            image = np.stack(channels, axis=0)

        return image


def augment(image_dir, dataset_names, random_seed=88):

    rotate180 = RandomRotate(np.random.RandomState(random_seed))

    for dataset_name in dataset_names:

        h5_m_file = h5py.File(
                f"{image_dir}/nonaugmented/{dataset_name}/moving_images.h5", "r")
        h5_f_file = h5py.File(
                f"{image_dir}/nonaugmented/{dataset_name}/fixed_images.h5", "r")
        problems = list(h5_m_file.keys())

        augmented_image_dir = f"{image_dir}/augmented/{dataset_name}"
        if not os.path.exists(augmented_image_dir):
            os.mkdir(augmented_image_dir)

        h5_aug_m_file = h5py.File(f"{augmented_image_dir}/moving_images.h5", "w")
        h5_aug_f_file = h5py.File(f"{augmented_image_dir}/fixed_images.h5", "w")

        x_dim, y_dim, z_dim = h5_m_file[problems[0]][:].shape

        for problem in tqdm(problems):

            moving_image = h5_m_file[problem][:]
            fixed_image = h5_f_file[problem][:]

            x_dim, y_dim, z_dim = fixed_image.shape
            rotated_moving_image = adjust_shape(
                        rotate180(moving_image),
                        (x_dim, y_dim, z_dim))
            rotated_fixed_image = adjust_shape(
                        rotate180(fixed_image),
                        (x_dim, y_dim, z_dim))

            h5_aug_m_file.create_dataset(problem, data = rotated_moving_image)
            h5_aug_f_file.create_dataset(problem, data = rotated_fixed_image)

        h5_aug_m_file.close()
        h5_aug_f_file.close()


def adjust_shape(image, target_shape):

    x_dim, y_dim, z_dim = target_shape

    if image.shape == (x_dim, z_dim, y_dim):
        image = np.transpose(image, (0, 2, 1))
    elif image.shape == (y_dim, x_dim, z_dim):
        image = np.transpose(image, (1, 0, 2))
    elif image.shape == (y_dim, z_dim, x_dim):
        image = np.transpose(image, (2, 0, 1))
    elif image.shape == (z_dim, y_dim, x_dim):
        image = np.transpose(image, (2, 1, 0))
    elif image.shape == (z_dim, x_dim, y_dim):
        image = np.transpose(image, (1, 2, 0))

    return image


if __name__ == "__main__":

    image_dir = \
    "/home/alicia/data_personal/regnet_dataset/euler-gpu_size-v1/train"
    #dataset_names = ["2022-01-17-01", "2022-01-23-04",
    #        "2022-01-27-01", "2022-01-27-04", "2022-03-16-02", "2022-06-14-01",
    #        "2022-06-28-01", "2022-07-15-06"]
    dataset_names = ["2022-01-09-01"]
    #augment(image_dir, dataset_names)
