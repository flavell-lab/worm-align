using JLD2, FileIO
using JSON


function find_elastix_solved_problems(dataset_path)

    @load dataset_path data_dict

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

