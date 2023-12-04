from wormalign.preprocess import generate_pregistered_images
import argparse


def execute(args):

    # generate registration problems
    generate_registration_problems(
            args.train_datasets,
            args.valid_datasets,
            args.test_datasets
    )
    # generate and write preregistered images in HDF5
    generate_pregistered_images(
                tuple(args.shape),
                args.path,
                args.batchsize,
                args.device
    )
    # generate configuration file for the DDF network


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocessing pipeline")
    parser.add_argument("-p", "--path", type=str, nargs=1,
                metavar="path_to_save_datasets",
                help="Path where to keep the preprocessed datasets")
    parser.add_argument("-s", "--shape", type=int, nargs=3, 
                metavar=("xdim", "ydim", "zdim"),
                help="Shape of the target image in the order of x-y-z")
    parser.add_argument("-b", "--batchsize", type=int, nargs=1,
                metavar="batch_size",
                help="Batch size used for searching Euler parameters")
    parser.add_argument("-d", "--device", type=str, nargs=1,
                metavar="device_name",
                help="CUDA device name")
    parser.add_argument("-train", "--train_datasets", type=str, nargs="+",
                metavar="datasets for training",
                help="enter names of training datasets")
    parser.add_argument("-valid", "--valid_datasets", type=str, nargs="+",
                metavar="datasets for validation",
                help="enter names of validation datasets")
    parser.add_argument("-test", "--test_datasets", type=str, nargs="+",
                metavar="datasets for testing",
                help="enter names of test datasets")
    args = parser.parse_args()
    execute(args)
