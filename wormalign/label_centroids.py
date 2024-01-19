from numpy.typing import NDArray
from tqdm import tqdm
import h5py
import numpy as np
import os


class CentroidLabel:

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.save_path = f"{self.dataset_path}/{self.dataset_name}_centroid"
        self._ensure_directory_exists(self.save_path)

        self.fixed_label_path = \
        f"{self.dataset_path}/{self.dataset_name}/fixed_labels.h5"
        self.moving_label_path = \
        f"{self.dataset_path}/{self.dataset_name}/moving_labels.h5"

    def write_labels(self, max_centroids: int = 200):

        problems = list(h5py.File(self.moving_label_path, "r").keys())

        with h5py.File(f"{self.save_path}/moving_labels.h5", "w") as hdf5_m_file, \
                h5py.File(f"{self.save_path}/fixed_labels.h5", "w") as hdf5_f_file:

            for problem in tqdm(problems):

                fixed_label = self._read_label(self.fixed_label_path, problem)
                moving_label = self._read_label(self.moving_label_path, problem)

                fixed_centroids = self._compute_centroids_3d(
                        fixed_label,
                        max_centroids
                )
                moving_centroids = self._compute_centroids_3d(
                        moving_label,
                        max_centroids
                )
                # IMPORTANT: flip moving and fixed labels
                hdf5_m_file[problem] = fixed_centroids
                hdf5_f_file[problem] = moving_centroids

    def _read_label(self, label_path: str, key: str) -> NDArray[np.int32]:

        return h5py.File(label_path, "r")[key][:]

    def _ensure_directory_exists(self, path):
        """
        Create the given directory if it does not already exist

        :param path: directory to create if does not exist
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _compute_centroids_3d(
        self,
        image: NDArray[np.int32],
        max_centroids: int
    ) -> NDArray[np.int32]:
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
                centroid_x = int(round(np.mean(indices[:, 0])))  # x-coordinate
                centroid_y = int(round(np.mean(indices[:, 1])))  # y-coordinate
                centroid_z = int(round(np.mean(indices[:, 2])))  # z-coordinate
                centroids[val-1] = [centroid_x, centroid_y, centroid_z]

        return centroids


def main(dataset_name, dataset_path):

    centroid_labeler = CentroidLabel(dataset_name, dataset_path)
    centroid_labeler.write_labels()

if __name__ == "__main__":

    main("2022-01-27-01_subset", "/data1/prj_register")

