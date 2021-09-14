import numpy as np
from numpy import Inf
import pandas as pd
from os.path import join
from julia import Main  # For interfacing with julia

Main.using("OPFLearn")

DEFAULT_INPUTS = ["pl", "ql"]
DEFAULT_OUTPUTS = ["pg", "qg", "vm_gen", "v_bus",
                   "p_to", "q_to", "p_fr", "q_fr"]
DEFAULT_DUALS = ["v_min", "v_max", "pg_min", "pg_max", "qg_min", "qg_max",
                 "p_to_max", "q_to_max", "p_fr_max", "q_fr_max"]


def create_samples(net_file, K=Inf, U=0.0, S=0.0, V=0.0, max_iter=Inf, T=Inf, discard=False, variance=False,
                   input_vars=DEFAULT_INPUTS, output_vars=DEFAULT_OUTPUTS, dual_vars=DEFAULT_DUALS,
                   sampler='sample_polytope_cprnd', sampler_opts=dict(),  # sampler_opts keys must be Main.Symbol("arg")
                   pl_max=None, pf_min=0.7, pf_lagging=True,
                   print_level=0, stat_track=False, save_while=False, save_infeasible=False, save_path="", net_path="",
                   model_type='PM.QCLSPowerModel',  # model_type should be a string that defines a PowerModels type
                   r_solver='JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)',
                   opf_solver='JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)'):
    """ create_samples(net, K; <keyword arguments>)

    Creates an AC OPF dataset for the given PowerModels network dictionary. Generates samples until one of the given stopping criteria is met.
    Takes options to determine how to sample points, what information to save, and what information is printed.

    Examples:
    ```jldoctest
    julia> results = create_samples("case5.m", 100; T=1000, net_path="data")
    ```

    Keyword arguments:
    - 'net::Dict': network information stored in a PowerModels.jl format specified dictionary
    - 'K::Integer': the maximum number of samples before stopping sampling
    - 'U::Float': the minimum % of unique active sets sampled in the previous 1 / U samples to continue sampling
    - 'S::Float': the minimum % of saved samples in the previous 1 / L samples to continue sampling
    - 'V::Float': the minimum % of feasible samples that increase the variance of the dataset in the previous 1 / L samples to continue sampling
    - 'T::Integer': the maximum time for the sampler to run in seconds.
    - 'max_iter::Integer': maximum number of iterations for the sampler to run for.
    - 'sampler::Function': the sampling function to use. This function must take arguements A and b, and can take optional arguments.
    - 'sampler_opts::Dict': a dictionary of optional arguments to pass to the sampler function.
    - 'pl_max::Array': the maximum active load values to use when initializing the sampling space and constraining the loads. If nothing, finds the maximum load at each bus with the given relaxed model type.
    - 'pf_min::Array/Float:' the minimum power factor for all loads in the system (Number) or an array of minimum power factors for each load in the system.
    - 'pf_lagging::Bool': indicating if load power factors can be only lagging (True), or both lagging or leading (False).
    - 'reset_level::Integer': determines how to reset the load point to be inside the polytope before sampling. 2: Reset closer to nominal load & chebyshev center, 1: Reset closer to chebyshev center, 0: Reset at chebyshev center.
    - 'model_type::Type': an abstract PowerModels type indicating the network model to use for the relaxed AC-OPF formulations (Max Load & Nearest Feasible)
    - 'r_solver': an optimizer constructor used for solving the relaxed AC-OPF optimization problems.
    - 'opf_solver': an optimizer constructor used to find the AC-OPF optimal solution for each sample.
    - 'print_level::Integer': from 0 to 3 indicating the level of info to print to console, with 0 indicating minimum printing.
    - 'stat_track::Integer': from 0 to 3 indicating the level of stats info saved during each iteration	0: No information saved, 1: Feasibility, New Certificate, Added Sample, Iteration Time, 2: Variance for all input & output variables
    - 'save_while::Bool': indicates whether results and stats information is saved to a csv file during processing.
    - 'save_infeasible::Bool': indicates if infeasible samples are saved. If true saves infeasible samples in a seperate file from feasible samples.
    - 'save_path::String:' a string with the file path to the desired result save location.
    - 'net_path::String': a string with the file path to the network file.
    - 'variance::Bool': indicates if dataset variance information is tracked for each unique active set.
    - 'discard::Bool': indicates if samples that do not increase the variance within a unique active set are discarded.

    See 'OPF-Learn: An Open-Source Framework for Creating Representative AC Optimal Power Flow Datasets'
    for more information on how the AC OPF datasets are created.

    Modified from AgenerateACOPFsamples.m written by Ahmed Zamzam
    """
    return Main.create_samples(net_file, K, U=U, S=S, V=V, max_iter=max_iter, T=T, discard=discard, variance=variance,
                               input_vars=input_vars, output_vars=output_vars, dual_vars=dual_vars,
                               sampler=Main.eval(sampler), sampler_opts=sampler_opts,
                               pl_max=pl_max, pf_min=pf_min, pf_lagging=pf_lagging,
                               print_level=print_level, stat_track=stat_track, save_while=save_while,
                               save_infeasible=save_infeasible, save_path=save_path, net_path=net_path,
                               model_type=Main.eval(model_type), r_solver=Main.eval(r_solver),
                               opf_solver=Main.eval(opf_solver))


def dist_create_samples(net_file, K=Inf, nproc=None, U=0.0, S=0.0, V=0.0, max_iter=Inf, T=Inf, discard=False,
                        variance=False,
                        input_vars=DEFAULT_INPUTS, output_vars=DEFAULT_OUTPUTS, dual_vars=DEFAULT_DUALS,
                        sampler='sample_polytope_cprnd', sampler_opts=dict(),  # sampler_opts keys: Main.Symbol("arg")
                        pl_max=None, pf_min=0.7, pf_lagging=True, print_level=0, stat_track=False, save_while=False,
                        save_infeasible=False, save_path="", net_path="",
                        model_type='PM.QCLSPowerModel',  # model_type should be a string that defines a PowerModels type
                        r_solver='JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)',
                        opf_solver='JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)'):
    """ Creates an AC OPF dataset for the given PowerModels network dictionary. Generates samples until one of the given stopping criteria is met.
    Takes options to determine how to sample points, what information to save, and what information is printed.

    Keyword arguments:
    - 'net::Dict': network information stored in a PowerModels.jl format specified dictionary
    - 'K::Integer': the maximum number of samples before stopping sampling
    - 'U::Float': the minimum % of unique active sets sampled in the previous 1 / U samples to continue sampling
    - 'S::Float': the minimum % of saved samples in the previous 1 / L samples to continue sampling
    - 'V::Float': the minimum % of feasible samples that increase the variance of the dataset in the previous 1 / L samples to continue sampling
    - 'T::Integer': the maximum time for the sampler to run in seconds.
    - 'max_iter::Integer': maximum number of iterations for the sampler to run for.
    - 'nproc::Integer': the number of processors for the sampler to run with. Defaults to the number reported by Distributed.nprocs().
    - 'sampler::Function': the sampling function to use. This function must take arguements A and b, and can take optional arguments.
    - 'sampler_opts::Dict': a dictionary of optional arguments to pass to the sampler function.
    - 'pl_max::Array': the maximum active load values to use when initializing the sampling space and constraining the loads. If nothing, finds the maximum load at each bus with the given relaxed model type.
    - 'pf_min::Array/Float:' the minimum power factor for all loads in the system (Number) or an array of minimum power factors for each load in the system.
    - 'pf_lagging::Bool': indicating if load power factors can be only lagging (True), or both lagging or leading (False).
    - 'reset_level::Integer': determines how to reset the load point to be inside the polytope before sampling. 2: Reset closer to nominal load & chebyshev center, 1: Reset closer to chebyshev center, 0: Reset at chebyshev center.
    - 'model_type::Type': an abstract PowerModels type indicating the network model to use for the relaxed AC-OPF formulations (Max Load & Nearest Feasible)
    - 'r_solver': an optimizer constructor used for solving the relaxed AC-OPF optimization problems.
    - 'opf_solver': an optimizer constructor used to find the AC-OPF optimal solution for each sample.
    - 'print_level::Integer': from 0 to 3 indicating the level of info to print to console, with 0 indicating minimum printing.
    - 'stat_track::Integer': from 0 to 3 indicating the level of stats info saved during each iteration	0: No information saved, 1: Feasibility, New Certificate, Added Sample, Iteration Time, 2: Variance for all input & output variables
    - 'save_while::Bool': indicates whether results and stats information is saved to a csv file during processing.
    - 'save_infeasible::Bool': indicates if infeasible samples are saved. If true saves infeasible samples in a seperate file from feasible samples.
    - 'save_path::String:' a string with the file path to the desired result save location.
    - 'net_path::String': a string with the file path to the network file.
    - 'variance::Bool': indicates if dataset variance information is tracked for each unique active set.
    - 'discard::Bool': indicates if samples that do not increase the variance within a unique active set are discarded.

    See 'OPF-Learn: An Open-Source Framework for Creating Representative AC Optimal Power Flow Datasets'
    for more information on how the AC OPF datasets are created.

    Modified from AgenerateACOPFsamples.m written by Ahmed Zamzam
    """
    if nproc is None:
        nproc = Main.eval("Sys.CPU_THREADS")
    Main.eval('Distributed.nprocs() > 1 && Distributed.rmprocs(Distributed.workers())')
    Main.addprocs(nproc - 1, exeflags="--project")
    Main.eval('Distributed.@everywhere using OPFLearn')
    return Main.dist_create_samples(net_file, K, U=U, S=S, V=V, max_iter=max_iter, T=T, discard=discard,
                                    variance=variance, nproc=nproc,
                                    input_vars=input_vars, output_vars=output_vars, dual_vars=dual_vars,
                                    sampler=Main.eval(sampler), sampler_opts=sampler_opts,
                                    pl_max=pl_max, pf_min=pf_min, pf_lagging=pf_lagging,
                                    print_level=print_level, stat_track=stat_track, save_while=save_while,
                                    save_infeasible=save_infeasible, save_path=save_path, net_path=net_path,
                                    model_type=Main.eval(model_type), r_solver=Main.eval(r_solver),
                                    opf_solver=Main.eval(opf_solver))


def save_results_csv(results, file_name, save_order, dir=""):
    """ Saves the results AC_output, AC_input, duals to a single csv file with variables
    saved in the given save_order
    """
    Main.save_results_csv(results, file_name, save_order, dir=dir)


def save_stats(stats, file_name, dir=""):
    """ Saves the last iteration of level 1 stats to a csv file.
    If the csv file does not already exist then calls save_stats
    to save the stats data with a header to a new csv file.
    """
    Main.save_stats(stats, file_name, dir=dir)


def results_to_array(results, save_order, header=True, imag_j=False):
    """ Takes the results objects from a create samples run and converts them to a single array.
    Requires the save_order, an array of variables, to save from the results objects. Takes
    optional boolean arguments header, whether to include a header as the first row, and
    imag_j, whether to convert complex data values to strings and replace 'im' with 'j' for
    datasets that are to be used outside of the Julia environment.
    """
    return Main.results_to_array(results, save_order, header=header, imag_j=imag_j)


def load_csv_data(data_file, input_labels, output_labels, data_dir="", match=False):
    """ Loads OPFLearn generated data from a csv file into two numpy arrays with headers
        of variables. Takes input_labels and output_labels are used to specify which
        variables are saved to their respective arrays. These labels can be specified to
        match exactly or just be contained in the variable headers of the OPFLearn csv file
        with the boolean match argument.
    """
    df = pd.read_csv(join(data_dir, data_file))

    col_names = df.columns
    data = df.to_numpy()

    # Find input, output, and dual values based on labels
    # Have labels be an input to the function?
    input_indx = []
    output_indx = []
    for indx, col in enumerate(col_names):
        if match:
            if col in input_labels:
                input_indx.append(indx)
            if col in output_labels:
                output_indx.append(indx)
        else:
            if any(substring in col for substring in input_labels):
                input_indx.append(indx)
            if any(substring in col for substring in output_labels):
                output_indx.append(indx)

    X = np.transpose(data[:, input_indx])
    Y = np.transpose(data[:, output_indx])

    return X, Y


if __name__ == "__main__":
    net_file = "pglib_opf_case5_pjm.m"
    dir = "C:\\Users\\Trager Joswig-Jones\\GithubRepos\\LearningOPF\\NetworkData"
    results = create_samples(net_file, 10, net_path=dir)
    # results = dist_create_samples(net_file, 10, net_path=dir)
    print(results)
