import os


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
