module ScanUtils

using JLD2, FileIO
using JSON
using YAML
using Glob

export filter_registration_problems, find_elastix_solved_problems, locate_dataset

function filter_registration_problems(config_file_path::String)

    config_dict = YAML.load_file(config_file_path)
    new_config_dict = Dict("train" => Dict(), "valid" => Dict(), "test" => Dict())

    for dataset_type in ["train", "valid", "test"]
        dataset_paths = config_dict["dataset"][dataset_type]["dir"]
        for dataset_path in dataset_paths
            dataset_name = split(dataset_path, "/")[end]
            new_config_dict[dataset_type][dataset_name] = find_elastix_solved_problems(String(locate_dataset(String(dataset_name))))
            open("resources/registration_problems_elastix_solved.json", "w") do file
                JSON.print(file, new_config_dict, 4)
            end
        end
    end
end

function find_elastix_solved_problems(dataset_path::String)

    @load joinpath(dataset_path, "data_dict.jld2") data_dict

    q_dict = data_dict["q_dict"]
    problem_list = []

    for (problem_tuple, register_dict) in q_dict
        problem = join(string.(problem_tuple), "to")
        ncc_scores = [d["NCC"] for d in values(register_dict)]

        if any(ncc < 2 for ncc in ncc_scores)
            push!(problem_list, problem)
        end
    end
    return problem_list
end


function locate_dataset(dataset_name::String)
    """
    Given the name of the dataset, this function locates the directory where
    this data file can be found.

    - `dataset_name`: name of the dataset; e.g. `2022-03-16-02`
    """
    operating_dir = pwd()

    neuropal_dir = "/data1/prj_neuropal/data_processed"
    kfc_dir = "/data1/prj_kfc/data_processed"
    rim_dir = "/data3/prj_rim/data_processed"

    dir_dataset_dict = Dict(
        neuropal_dir => readdir(neuropal_dir),
        kfc_dir => readdir(kfc_dir),
        rim_dir => readdir(rim_dir)
    )

    for (base_dir, dataset_dirs) in dir_dataset_dict
        if any(occursin(dataset_name, dataset_dir) for dataset_dir in dataset_dirs)
            cd(base_dir)
            dataset_path = joinpath(base_dir, glob("$(dataset_name)_*")[1])
            cd(operating_dir)
            return dataset_path
        end
    end
end

end # module
