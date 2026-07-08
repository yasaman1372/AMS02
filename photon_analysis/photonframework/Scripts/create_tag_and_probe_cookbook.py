#!/usr/bin/env python3

from collections import defaultdict
from datetime import datetime
import os
import subprocess as sp

from cookbook.Cookbook import Cookbook

from tools.config import get_config
from tools.utilities import decompose_graph, recursive_dependencies, start_time_of_bartels_rotation


def parse_date(date_str):
    if date_str.startswith("BR"):
        br_number = int(date_str[2:])
        return start_time_of_bartels_rotation(br_number)
    return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())

def visualize_dependencies(cookbook):
    import graphviz
    graph = graphviz.Digraph("task-dependency", engine="neato")
    for task in cookbook.tasklist:
        graph.node(task.ref)
    for task in cookbook.tasklist:
        for dependency in task.dependencies:
            graph.edge(task.ref, dependency)
    graph.render()

def print_graph(cookbook):
    nodes = [task.ref for task in cookbook.tasklist]
    tasks = {task.ref: task.ref for task in cookbook.tasklist}
    edges = [(task.ref, tasks[dependency]) for task in cookbook.tasklist for dependency in task.dependencies]
    resolved_nodes = set()
    remaining_nodes = set(nodes)
    level = 0
    task_levels = {}
    while remaining_nodes:
        new_resolved_nodes = set(remaining_nodes)
        for edge in edges:
            if edge[1] not in resolved_nodes and edge[0] in new_resolved_nodes:
                new_resolved_nodes.remove(edge[0])
        print(f"Graph level {level}:")
        for node in sorted(new_resolved_nodes):
            dependencies = sorted([edge[1] for edge in edges if edge[0] == node], key=lambda r: (-task_levels[r], r))
            print(f" {node:<60}: {','.join(dependencies)}")
        remaining_nodes -= new_resolved_nodes
        resolved_nodes |= new_resolved_nodes
        for node in new_resolved_nodes:
            task_levels[node] = level
        level += 1



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--workdir", default=os.getcwd())
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--visualize-dependencies", action="store_true")
    parser.add_argument("--partition", default="amsclx")
    parser.add_argument("--cores", type=int)
    parser.add_argument("--timelimit-iss", default="03:00")
    parser.add_argument("--timelimit-mc", default="01:00")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--jobtag", help="Prefix for jobnames, to distinguish jobs from multiple running cookbooks.")

    args = parser.parse_args()

    os.putenv("PHOTON_CONFIG", args.config)
    jobtag = args.jobtag
    if jobtag is None:
        jobtag = os.path.splitext(os.path.basename(args.config))[0]
    cores = args.cores
    if cores is None:
        cores = int(sp.check_output(["cluster_info", "--partition", args.partition, "cores_per_node"]))

    config = get_config(args.config)
    workdir = args.workdir
    cookbook = Cookbook(main_path=workdir, jobtag=jobtag)
    repo_region_dir = os.path.expandvars("$PHOTONFRAMEWORK/data/Regions")

    # write trees
    datadir = args.datadir
    if not os.path.isdir(datadir):
        raise ValueError(f"Expected photon trees in {datadir!r}, but it does not exist.")
    
    if "photon_analyses" in config:
        for photon_analysis_name, photon_analysis_config in config["photon_analyses"].items():
            photon_analysis_dir = os.path.join(workdir, photon_analysis_name)
            os.makedirs(photon_analysis_dir, exist_ok=True)
            tag_tree_dir = os.path.join(photon_analysis_dir, "tag_and_probe")
            os.makedirs(tag_tree_dir, exist_ok=True)
            sample_name = photon_analysis_config["sample"]
            preselection_name = config["samples"][sample_name].get("preselection", None)
            iss_dataset_name = photon_analysis_config["iss_dataset"]
            mc_signal_dataset_name = photon_analysis_config["mc_signal_dataset"]

            iss_gamma_tree_signal_dir = os.path.join(tag_tree_dir, 'iss_signal')
            iss_gamma_tree_background_dir = os.path.join(tag_tree_dir, 'iss_background')
            mc_gamma_tree_dir = os.path.join(tag_tree_dir, 'mc')
            os.makedirs(iss_gamma_tree_signal_dir, exist_ok=True)
            os.makedirs(iss_gamma_tree_background_dir, exist_ok=True)
            os.makedirs(mc_gamma_tree_dir, exist_ok=True)
            iss_gamma_tree_signal_arguments = []
            iss_gamma_tree_background_arguments = []
            mc_gamma_tree_arguments = []

            iss_photon_tree_list = os.path.join(datadir, "data", iss_dataset_name, "trees", f"PhotonTree{preselection_name}.list" if preselection_name is not None else "trees.list")

            parallel_total = config["datasets"][iss_dataset_name]["parallel_factor"]
            for parallel_index in range(parallel_total):
                iss_gamma_tree_signal_arguments.append(["--tree", iss_photon_tree_list, "--config", args.config, datadir, "--sample", sample_name, "--nprocesses", str(cores), "--parallel", str(parallel_index), str(parallel_total), "--outputprefix", "iss", "--chunk-size", str(args.chunk_size), "--plot-style", "iss", "--preselections", "is_signal"])
                iss_gamma_tree_background_arguments.append(["--tree", iss_photon_tree_list, "--config", args.config, datadir, "--sample", sample_name, "--nprocesses", str(cores), "--parallel", str(parallel_index), str(parallel_total), "--outputprefix", "iss", "--chunk-size", str(args.chunk_size), "--plot-style", "iss", "--preselections", "is_background"])

            for dataset_name in [mc_signal_dataset_name] + photon_analysis_config.get("mc_background_datasets", []):
                photon_tree_list = os.path.join(datadir, "data", dataset_name, "trees", f"PhotonTree{preselection_name}.list")
                parallel_total = config["datasets"][dataset_name]["parallel_factor"]
                for parallel_index in range(parallel_total):
                    mc_gamma_tree_arguments.append(["--tree", photon_tree_list, "--config", args.config, datadir, "--sample", sample_name, "--nprocesses", str(cores), "--parallel", str(parallel_index), str(parallel_total), "--outputprefix", f"{dataset_name}", "--chunk-size", str(args.chunk_size), "--plot-style", "mc", "--energy-estimators", 'McMomentum', 'TrkTrackBestPairRigidity'])

            with open(os.path.join(iss_gamma_tree_signal_dir, "arguments.list"), "w") as arguments_file:
                for line in iss_gamma_tree_signal_arguments:
                    arguments_file.write(" ".join(line) + "\n")

            with open(os.path.join(iss_gamma_tree_background_dir, "arguments.list"), "w") as arguments_file:
                for line in iss_gamma_tree_background_arguments:
                    arguments_file.write(" ".join(line) + "\n")

            with open(os.path.join(mc_gamma_tree_dir, "arguments.list"), "w") as arguments_file:
                for line in mc_gamma_tree_arguments:
                    arguments_file.write(" ".join(line) + "\n")

            iss_gamma_tree_signal_task = cookbook.new_task("joblist",
                comment=f"creates tag and probe histograms for signal {photon_analysis_name}",
                job="create_tag_and_probe_histograms.py",
                args=f"--arguments '%s' --threads_per_job {cores} --partition {args.partition} --timelimit {args.timelimit_iss}",
                jobs_from=os.path.join(iss_gamma_tree_signal_dir, "arguments.list"),
                directory=iss_gamma_tree_signal_dir,
                ref=f"photons_{photon_analysis_name}_trees_iss_signal",
                depends="none")
            
            iss_gamma_tree_task = cookbook.new_task("joblist",
                comment=f"creates tag and probe histograms for {photon_analysis_name}",
                job="create_tag_and_probe_histograms.py",
                args=f"--arguments '%s' --threads_per_job {cores} --partition {args.partition} --timelimit {args.timelimit_iss}",
                jobs_from=os.path.join(iss_gamma_tree_background_dir, "arguments.list"),
                directory=iss_gamma_tree_background_dir,
                ref=f"photons_{photon_analysis_name}_trees_iss",
                depends="none")
            
            mc_gamma_tree_task = cookbook.new_task("joblist",
                comment=f"Create tag and probe histograms for {photon_analysis_name}",
                job="create_tag_and_probe_histograms.py",
                args=f"--arguments '%s' --threads_per_job {cores} --partition {args.partition} --timelimit {args.timelimit_mc}",
                jobs_from=os.path.join(mc_gamma_tree_dir, "arguments.list"),
                directory=mc_gamma_tree_dir,
                ref=f"photons_{photon_analysis_name}_trees_mc",
                depends="none")




    cookbook.save_db()

    if args.visualize_dependencies:
        visualize_dependencies(cookbook)
        print_graph(cookbook)

if __name__ == "__main__":
    main()