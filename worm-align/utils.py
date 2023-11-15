from julia.api import Julia
from scipy import ndimage
import numpy as np
import os

jl = Julia(compiled_modules=False)
jl.eval('include("adjust.jl")')
adjust_image_size = jl.eval("adjust_image_cm")

def locate_directory(dataset_date):

    '''
    Given the date when the dataset was collected, this function locates which
    directory this data file can be found
    '''

    neuropal_dir = '/home/alicia/data_prj_neuropal/data_processed'
    non_neuropal_dir = '/home/alicia/data_prj_kfc/data_processed'
    for directory in os.listdir(neuropal_dir):
        if dataset_date in directory:
            return os.path.join(neuropal_dir, directory)

    for directory in os.listdir(non_neuropal_dir):
        if dataset_date in directory:
            return os.path.join(non_neuropal_dir, directory)

    raise Exception(f'Dataset {dataset_date} cannot be founed.')


def calculate_gncc(fixed, moving):

    mu_f = np.mean(fixed)
    mu_m = np.mean(moving)
    a = np.sum(abs(fixed - mu_f) * abs(moving - mu_m))
    b = np.sqrt(np.sum((fixed - mu_f) ** 2) * np.sum((moving - mu_m) ** 2))
    return a / b


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

