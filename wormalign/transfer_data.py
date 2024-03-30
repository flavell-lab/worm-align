from tqdm import tqdm
import os
import random
import h5py
import shutil


def transfer(dataset_type):

    src_dir = f'/data3/prj_register/ALv3/{dataset_type}/nonaugmented'
    dst_dir = '/data3/prj_register/ALv3_sample'

    os.makedirs(dst_dir, exist_ok=True)

    folders = [folder for folder in os.listdir(src_dir) if
            os.path.isdir(os.path.join(src_dir, folder))]

    for folder in tqdm(folders):

        dst_folder = os.path.join(dst_dir, f"{dataset_type}/nonaugmented",
                folder)
        os.makedirs(dst_folder, exist_ok=True)
        print(dst_folder)

        # Open the HDF5 files
        src_fixed_images = h5py.File(os.path.join(src_dir, folder, 'fixed_images.h5'), 'r')
        src_fixed_rois = h5py.File(os.path.join(src_dir, folder, 'fixed_rois.h5'), 'r')
        src_moving_images = h5py.File(os.path.join(src_dir, folder, 'moving_images.h5'), 'r')
        src_moving_rois = h5py.File(os.path.join(src_dir, folder, 'moving_rois.h5'), 'r')
        keys = list(src_fixed_images.keys())
        selected_keys = random.sample(keys, 100)

        # Create the destination HDF5 files
        dst_fixed_images = h5py.File(os.path.join(dst_folder, 'fixed_images.h5'), 'w')
        dst_fixed_rois = h5py.File(os.path.join(dst_folder, 'fixed_rois.h5'), 'w')
        dst_moving_images = h5py.File(os.path.join(dst_folder, 'moving_images.h5'), 'w')
        dst_moving_rois = h5py.File(os.path.join(dst_folder, 'moving_rois.h5'), 'w')

        # Copy the data to the destination files
        for key in selected_keys:
            dst_fixed_images.create_dataset(key, data=src_fixed_images[key][:])
            dst_fixed_rois.create_dataset(key, data=src_fixed_rois[key][:])
            dst_moving_images.create_dataset(key, data=src_moving_images[key][:])
            dst_moving_rois.create_dataset(key, data=src_moving_rois[key][:])

        # Close the files
        src_fixed_images.close()
        src_fixed_rois.close()
        src_moving_images.close()
        src_moving_rois.close()
        dst_fixed_images.close()
        dst_fixed_rois.close()
        dst_moving_images.close()
        dst_moving_rois.close()


if __name__ == "__main__":

    transfer("train")
