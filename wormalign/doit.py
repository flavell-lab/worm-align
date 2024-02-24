from wormalign.network import DDFNetworkTrainer, DDFNetworkTester
from wormalign.sample import Sampler
from wormalign.preprocess import RegistrationProcessor
from wormalign.warp_label import generate_labels
from wormalign.warp_image import Channel2ImageWarper
import json
from tqdm import tqdm

TRAIN_DATASETS = [
    '2023-06-24-28', '2022-04-18-04', '2022-06-14-07', '2022-02-08-04',
    '2023-07-07-18', '2023-07-28-04', '2023-08-22-08', '2023-01-17-01',
    '2022-06-14-13', '2022-01-17-01', '2023-01-23-01', '2023-07-07-11',
    '2023-07-06-01', '2022-06-28-01', '2023-01-19-15', '2023-08-07-01',
    '2022-07-20-01', '2023-06-24-02', '2023-09-15-08', '2022-06-28-07',
    '2023-07-01-09', '2023-01-23-08', '2023-08-17-08', '2023-01-23-21',
    '2023-07-01-01', '2022-03-22-01', '2023-07-11-02', '2022-08-02-01',
    '2023-06-09-01', '2023-07-12-01', '2022-03-15-04', '2023-08-18-11',
    '2023-01-06-01', '2022-07-15-06', '2022-12-21-06', '2023-08-07-16',
    '2023-07-13-01', '2023-06-09-10', '2023-01-19-01', '2023-01-19-08',
    '2023-01-19-22', '2023-01-05-18', '2023-08-25-02', '2022-02-16-04',
    '2023-07-11-10', '2023-08-17-01', '2023-01-05-01', '2023-06-24-11',
    '2023-01-23-15', '2023-03-07-01', '2022-07-26-01', '2023-08-19-08',
    '2023-07-07-01', '2022-07-15-12', '2022-02-16-01', '2023-01-09-28',
    '2022-06-14-01'
]
VALID_DATASETS = [
'2023-08-23-23', '2022-04-12-04', '2023-09-02-10', '2023-01-06-08',
'2023-01-09-22', '2023-08-22-01', '2023-08-18-18', '2023-01-13-07',
'2023-01-16-08', '2023-01-06-15', '2023-06-24-19', '2023-08-31-03',
'2023-01-10-07', '2023-10-03-02', '2023-01-09-08', '2023-01-10-14',
'2023-09-15-01', '2022-04-05-01', '2023-10-15-18', '2023-07-01-23',
'2023-01-09-15', '2023-01-16-01'
]
TEST_DATASETS = [
'2022-04-14-04',
"2023-01-16-15",
'2023-01-16-22',
'2023-01-17-07',
'2023-01-17-14',
"2023-01-18-01",
'2023-08-19-01',
'2023-08-25-09',
"2023-08-23-09",
"2023-09-01-09",
"2023-09-01-01",
"2023-08-07-08",
"2023-07-13-09"
]

def sample():
    dataset_dict = {
        "train": TRAIN_DATASETS,
        "valid": VALID_DATASETS,
        "test": TEST_DATASETS
    }
    sampler = Sampler(dataset_dict, diy_registration_problems = True)
    sampler("registration_problems_ALv3", num_problems=700)

def sample_from_problems():

    with open("resources/registration_problems_elastix_solved.json", "r") as f:
        problem_dict = json.load(f)
    sampler = Sampler(None, problem_dict)
    sampler("registration_problems_ALv2", num_problems=150)

def preprocess():

    processor = RegistrationProcessor(
        #(208, 96, 56),
        (284, 120, 64),
        save_directory = "/data3/prj_register/ALv3",
        problem_file = "registration_problems_ALv3-test-1",
        euler_search = False
        #device_name = "cuda:2"
    )
    processor.process_datasets()

def preprocess_image():

    processor = Channel2ImageWarper(
            (284, 120, 64),
            save_directory = "/data3/prj_register",
            device_name = "cuda:2"
    )
    processor.process_datasets()

def preprocess_roi():

    device_name = "cuda:1"
    target_image_shape = (284, 120, 64)
    problem_file = "registration_problems_ALv3-test-cleaned"
    #save_directory = "/home/alicia/data_personal/wormalign/datasets"
    save_directory = "/data3/prj_register/ALv3"
    generate_labels(device_name,
            target_image_shape, problem_file, save_directory, simply_crop=True)


def create_centroid_labels():

    centroid_labeler = CentroidLabel(
            "2022-01-27-01_subset",
            "/data3/prj_register")
    centroid_labeler.write_labels()


def train():
    config_file_path = "configs/config_label.yaml"
    #config_file_path = "configs/config_euler-gpu-ddf_resize-v1_size-v1.yaml"
    #config_file_path = "configs/config_euler-ddf.yaml"
    #config_file_path = "/data1/prj_register/updated_config_centroid_label_1.yaml"
    log_directory =  "/data1/alicia/regnet_ckpt"
    #experiment_name = "euler_network-ckpt21"
    experiment_name = "labeled_network_20240118-161347"
    cuda_device = "1" # best one: "0" (on `flv-c3`)
    checkpoint = 22
    network_trainer = DDFNetworkTrainer(config_file_path, log_directory,
            experiment_name, cuda_device, checkpoint, gpu_growth=True)
    network_trainer()


def test():
    config_file_path = "/data3/prj_register/2024-01-25-train/config.yaml"
    #trained_ddf_model = "labeled_network_20240118-161347"
    log_directory =  "/data3/prj_register/2024-01-25-train"
    #trained_ddf_model = \
    #"labeled_network_20240118-161347_20240119-174036"
    trained_ddf_model = "centroid_labels_augmented"
    #experiment_name = "labeled_network_ck115"
    experiment_name = "centroid_label_network_ckpt19"
    checkpoint = 19
    cuda_device = "1" # best one: "0" (on `flv-c3`)
    network_tester = DDFNetworkTester(config_file_path, log_directory, trained_ddf_model,
            experiment_name, cuda_device, checkpoint)
    network_tester()

def test_centroid_label_network():
    config_file_path = "/data3/prj_register/2024-01-25-train/config_ALv1.yaml"
    log_directory =  "/data3/prj_register/2024-01-25-train"
    trained_ddf_model = "centroid_labels_augmented_batched"
    experiment_name = "centroid_label_network_ckpt154"

    checkpoint = 154
    cuda_device = "cuda:1"
    print(cuda_device)
    network_tester = DDFNetworkTester(config_file_path, log_directory, trained_ddf_model,
                                    experiment_name, cuda_device, checkpoint)
    print(network_tester)
    network_tester()

# Load the JSON content from two files
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Assuming the structure is the same for both files
def find_shared_content(json1, json2):
    shared_content = {}
    for key in json1:  # key would be 'train', 'valid', 'test'
        shared_content[key] = {}
        for subkey in tqdm(json1[key]):  # subkey would be dates like "2022-02-08-04"
            if subkey in json2[key]:  # Check if the subkey exists in json2
                # Find intersection of lists
                shared_values = list(set(json1[key][subkey]) & set(json2[key][subkey]))
                if shared_values:  # If there's shared content, add it
                    shared_content[key][subkey] = shared_values
    return shared_content

def write_shared_content():
    json1 = load_json('resources/registration_problems_elastix_solved.json')
    json2 = load_json('resources/registration_problems_ALv1.json')

    # Find shared content
    shared_content = find_shared_content(json2, json1)

    # Write shared content to a new JSON file
    with open('shared_content.json', 'w') as file:
        json.dump(shared_content, file, indent=4)

if __name__ == "__main__":
    #sample_from_problems()
    #sample()
    #preprocess()
    preprocess_roi()
    #preprocess_image()
    #test_centroid_label_network()

