from tqdm import tqdm
import h5py
import numpy as np


def main(kind: str, dtype: str, num_problems: int):

    base = "/home/alicia/data_personal/wormalign/datasets/train/nonaugmented"
    data_from = f"{base}/2022-01-23-04"
    data_to = f"{base}/2022-01-23-04_onehot"

    with h5py.File(f"{data_from}/fixed_images.h5", "r") as f:
        problems = list(f.keys())[:num_problems]

    with h5py.File(f"{data_from}/{kind}_{dtype}.h5", "r") as f_from, \
        h5py.File(f"{data_to}/{kind}_{dtype}.h5", "w") as f_to:

            for problem in tqdm(problems):

                if dtype == "labels":
                    label = f_from[problem][:].astype(np.int32)
                    one_hot_label = one_hot_encode_3d(label).astype(np.float32)
                    f_to[problem] = remove_empty_channels(one_hot_label)
                elif dtype == "images":
                    f_to[problem] = f_from[problem][:].astype(np.float32)


def remove_empty_channels(one_hot_encoded):
    # Check for non-empty channels (any channel that has at least one '1')
    non_empty_channels = np.any(one_hot_encoded, axis=(0, 1, 2))

    # Keep only non-empty channels
    return one_hot_encoded[:, :, :, non_empty_channels]


def one_hot_encode_3d(array):
    # Find the maximum label in the array to determine the depth of one-hot encoding
    num_labels = np.max(array)

    # Create a 4D array of zeros with the shape (x, y, z, num_labels)
    one_hot = np.zeros((*array.shape, num_labels))

    # Iterate through the array and set the appropriate one-hot element to 1
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                label = array[x, y, z]
                if label > 0:
                    one_hot[x, y, z, label - 1] = 1

    return one_hot


def convert_to_nii5(image_path, name, typ):

    problems = list(h5py.File(f"{image_path}/{name}_{typ}.h5", "r").keys())
    new_path = f"{image_path}/{name}_{typ}"
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for problem in problems:

        with h5py.File(f"{image_path}/{name}_{typ}.h5", "r") as f:
            image = f[problem][:]
        nifti_image = nib.Nifti1Image(image, np.eye(4))
        nifti_path = os.path.join(new_path, f"{problem}.nii.gz")
        nib.save(nifti_image, nifti_path)
        print(f"image created!")


if __name__ == "__main__":

    image_path = \
    "/home/alicia/data_personal/wormalign/datasets/train/nonaugmented/2022-01-23-04_onehot"
    #delete_unmatched_images(image_path)
    convert_to_nii5(image_path, "fixed", "labels")
    convert_to_nii5(image_path, "fixed", "images")
    convert_to_nii5(image_path, "moving", "labels")
    convert_to_nii5(image_path, "moving", "images")

    num_problems = 10
    main("fixed", "labels", num_problems)
    main("fixed", "images", num_problems)
    main("moving", "labels", num_problems) 
    main("moving", "images", num_problems)

