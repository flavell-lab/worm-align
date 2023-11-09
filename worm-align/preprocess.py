from julia.api import Julia
from scipy import ndimage
from tqdm import tqdm
from utils import locate_directory
import SimpleITK as sitk
import glob
import h5py
import json
import numpy as np
import os

jl = Julia(compiled_modules=False)
jl.eval('include("adjust.jl")')
adjust_image_size = jl.eval("adjust_image_cm")

def resize_images(save_directory, registration_problem_file_path):

    with open(registration_problem_file_path, 'r') as f:
        registration_problem_dict = json.load(f)

    for dataset_type_n_name, problems in registration_problem_dict.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f"{save_directory}/{dataset_type}/{dataset_name}"

        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        if not os.path.exists(f"{save_directory}/{dataset_type}"):
            os.mkdir(f"{save_directory}/{dataset_type}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            t_moving, t_fixed = problem.split('to')
            t_moving_4 = t_moving.zfill(4)
            t_fixed_4 = t_fixed.zfill(4)
            fixed_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
            )[0]
            moving_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
            )[0]

            fixed_image_T = get_image_T(fixed_image_path)
            fixed_image_median = np.median(fixed_image_T)
            moving_image_T = get_image_T(moving_image_path)
            moving_image_median = np.median(moving_image_T)

            target_dim = (208, 96, 56)
            resized_fixed_image_xyz = filter_and_crop(fixed_image_T,
                        fixed_image_median, target_dim)
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median, target_dim)

            hdf5_m_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_moving_image_xyz)
            hdf5_f_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_fixed_image_xyz)

        hdf5_m_file.close()
        hdf5_f_file.close()


def filter_and_crop(image_T, image_median, target_dim):

    filtered_image_CM = get_image_CM(image_T)
    filtered_image_T = filter_image(image_T, image_median)

    return get_cropped_image(
                filtered_image_T,
                filtered_image_CM,
                target_dim, -1).astype(np.float32)


def filter_image(image, threshold):

    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def get_cropped_image(image_T, center, target_dim, projection):

    if projection >= 0:
        return adjust_image_size(
                image_T,
                center,
                target_dim).max(projection)
    elif projection == -1:
        return adjust_image_size(
                image_T,
                center,
                target_dim)


def get_image_T(image_path):

    '''Given the path, read image of .nrrd format as numpy array
    '''

    image_nrrd = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_nrrd)
    if image.ndim == 4:
        image = image.squeeze()
    image_T = np.transpose(image, (2,1,0))

    return image_T


def get_image_CM(image_T):

    '''Taking image of shape in the order (x, y, z) and find its center of mass
    after filtering
    '''
    # subtract the median pixel from the image; zero out the negative pixels
    image_T_wo_background = image_T - np.median(image_T)
    image_T_wo_background[image_T_wo_background < 0] = 0
    x, y, z = ndimage.center_of_mass(image_T_wo_background)

    return (round(x), round(y), round(z))


if __name__ == "__main__":
    save_directory = "/home/alicia/data_personal/test_preprocess"
    registration_problem_file_path = "/home/alicia/notebook/register/jungsoo_registration_problems.json"
    resize_images(save_directory, registration_problem_file_path)
