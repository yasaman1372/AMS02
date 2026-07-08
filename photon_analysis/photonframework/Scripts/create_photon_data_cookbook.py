#!/usr/bin/env python3

from collections import defaultdict
from datetime import datetime
import os
import subprocess as sp

from cookbook.Cookbook import Cookbook

from tools.config import get_config
from tools.utilities import decompose_graph, recursive_dependencies, start_time_of_bartels_rotation
from tools.variables import DerivedVariables


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
    parser.add_argument("--visualize-dependencies", action="store_true")
    parser.add_argument("--partition", default="amsclx")
    parser.add_argument("--cores", type=int)

    args = parser.parse_args()

    os.putenv("PHOTON_CONFIG", args.config)
    cores = args.cores
    if cores is None:
        cores = int(sp.check_output(["cluster_info", "--partition", args.partition, "cores_per_node"]))

    config = get_config(args.config)
    workdir = args.workdir

    cookbook = Cookbook(main_path=workdir)

    treename = config["analysis"].get("treename", "PhotonTree")

    # write trees
    datadir = os.path.join(workdir, "data")
    os.makedirs(datadir, exist_ok=True)
    trees_tasks = {}
    trees_lists = {}
    acqt_lists = {}
    dataset_configs = {}
    dataset_labels = {}
    dataset_fractions = {}
    mc_triggers_tasks = {}
    mc_triggers_files = {}
    mva_tasks = {}
    mva_dirs = {}
    flux_tasks = {}
    flux_dirs = {}
    likelihood_tasks = {}
    resolution_tasks = {}
    mc_weighting_tasks = {}

    for dataset_name, dataset in config["datasets"].items():
        dataset_dir = os.path.join(datadir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        writer_tasks = []
        triggers_tasks = []
        acqt_lists[dataset_name] = []
        preselections = config["analysis"]["preselections"]
        for acqt_dataset in dataset["acqt_datasets"]:
            acqt_name = acqt_dataset["name"]
            acqt_name_clean = acqt_name.replace(".", "_")
            acqt_dir = os.path.join(dataset_dir, acqt_dataset["name"])
            os.makedirs(acqt_dir, exist_ok=True)
            list_dir = os.path.join(acqt_dir, "acqt_list")
            list_filename = f"{dataset_name}_{acqt_name}.list"
            writer_dir = os.path.join(acqt_dir, "trees")
            scalings_dir = os.path.join(acqt_dir, "scalings")
            auxiliary_dir = os.path.join(acqt_dir, "auxiliary")

            writer_args = ["--AuxiliaryObjectManager/merge"]
            if "preselections" in dataset:
                preselections = dataset["preselections"]
            writer_args.extend(["--preselection", ",".join(preselections)])
            if dataset["acqt_datatype"] == "MC":
                writer_args.extend(["----McSpectrumScaler/inputfile", os.path.join(scalings_dir, "mcScalingWeights.root")])
            if len(writer_args) == 1:
                writer_args.append(" ")
            if dataset.get("prescale", True):
                writer_args.extend(["--prescale-trd-hits", "--prescale-same-sign", "--prescale-single-track", "--prescale-hadronic-shower"])
            nruns = 20
            timelimit = 3
            throttle = None
            if "nruns" in acqt_dataset:
                nruns = acqt_dataset["nruns"]
            if dataset.get("with_ams_variables", False):
                writer_args.append("--with-ams-variables")
                nruns = 1
                timelimit = 6
                throttle = 4
            if dataset.get("with_tracker_hits", True):
                writer_args.append("--with-tracker-hits")
            writer_args.append("--MPI/NonCollectiveIO")
            acqt_args = ["--arguments", "'" + " ".join(writer_args) + "'", "--timelimit", f"{timelimit}:00", "--mpi", "--sandbox", "--partition", args.partition, "--mpi_cores_per_node", str(cores)]
            if throttle is not None:
                acqt_args.extend(["--throttle", str(throttle)])
            acqt_args.extend(["--nruns", str(nruns)])
            list_args = [dataset['acqt_datatype'], acqt_name, "-o", list_filename]
            if "acqt_datadir" in dataset:
                list_args.extend(["--acqt-datadir", dataset["acqt_datadir"]])
            if "max_files" in acqt_dataset:
                list_args.extend(["--max-files", str(acqt_dataset["max_files"])])
            if "min_run" in acqt_dataset:
                list_args.extend(["--min-run", str(acqt_dataset["min_run"])])
            if "max_run" in acqt_dataset:
                list_args.extend(["--max-run", str(acqt_dataset["max_run"])])

            list_task = cookbook.new_task("interactive",
                comment=f"ACQt file list for {dataset_name} {acqt_name}",
                command="create_acqt_list.py",
                args=" ".join(list_args),
                targets=list_filename,
                directory=list_dir,
                ref=f"{dataset_name}_{acqt_name_clean}_list",
                depends=None)
            list_path = os.path.join(list_dir, list_filename)
            acqt_lists[dataset_name].append(list_path)
            writer_dependencies = [list_task.ref]
            if dataset["acqt_datatype"] == "MC":
                mc_scalings_task = cookbook.new_task("interactive",
                    comment=f"Create MC scalings files for {dataset_name} {acqt_name}",
                    command="produce_mc_scalings",
                    args=f"--inputlist {list_path}",
                    targets="mcScalingWeights.root",
                    directory=scalings_dir,
                    ref=f"{dataset_name}_{acqt_name_clean}_scalings",
                    depends=list_task.ref)
                writer_dependencies.append(mc_scalings_task.ref)
            writer_task = cookbook.new_task("acqt",
                comment=f"Run tree writer on {dataset_name} {acqt_name}",
                job=config.get("analysis", {}).get("tree_writer", "am_photon_writer"),
                filelist=list_path,
                args=" ".join(acqt_args),
                nexpected_per_job=1,
                nexpected_per_process=int(2**len(preselections) - 1),
                directory=writer_dir,
                ref=f"{dataset_name}_{acqt_name_clean}",
                depends=",".join(writer_dependencies))
            writer_tasks.append(writer_task)
            auxiliary_task = cookbook.new_task("interactive",
                command="ac_merge",
                args=f"--input {writer_dir}/results/PhotonFramework_Auxiliary_*.root --resultprefix Preselections",
                targets="Preselections.root",
                directory=auxiliary_dir,
                ref=f"{dataset_name}_{acqt_name_clean}_aux",
                depends=writer_task.ref)
            if dataset["acqt_datatype"] == "MC":
                triggers_task = cookbook.new_task("interactive",
                    comment=f"Store number of MC triggers of {dataset_name} {acqt_name}",
                    command="am_mc_triggers",
                    args=f"--inputlist {list_path} --species {dataset['species']}",
                    targets="triggers.txt",
                    directory=os.path.join(acqt_dir, "triggers"),
                    ref=f"{dataset_name}_{acqt_name_clean}_triggers",
                    depends=list_task.ref)
                triggers_tasks.append(triggers_task)
        trees_list_dir = os.path.join(dataset_dir, "trees")
        os.makedirs(trees_list_dir, exist_ok=True)
        tree_list_names = ["trees.list"] + [f"{treename}{preselection}.list" for preselection in preselections]
        trees_list_task = cookbook.new_task("interactive",
            comment=f"Create list of tree files for {dataset_name}",
            command="create_tree_list.py",
            args=" ".join(["--prefix", treename, f"--infixes"] + preselections + ["--directories"] + [os.path.join(task.directory, "results") for task in writer_tasks]),
            targets=",".join(tree_list_names),
            directory=trees_list_dir,
            ref=f"{dataset_name}_trees",
            depends=",".join((task.ref for task in writer_tasks)))
        trees_tasks[dataset_name] = trees_list_task
        trees_lists[dataset_name] = os.path.join(trees_list_task.directory, "trees.list")
        dataset_labels[dataset_name] = dataset.get("label", dataset_name)
        dataset_configs[dataset_name] = dataset
        dataset_fractions[dataset_name] = 1

        if dataset["acqt_datatype"] == "MC":
            triggers_task = cookbook.new_task("interactive",
                comment=f"Join MC triggers files for {dataset_name}",
                command="join_mc_triggers_files.py",
                args=f"--charge {dataset['charge']} " + " ".join(os.path.join(task.directory, "triggers.txt") for task in triggers_tasks),
                targets="triggers.txt",
                directory=os.path.join(dataset_dir, "triggers"),
                ref=f"{dataset_name}_triggers",
                depends=",".join((task.ref for task in triggers_tasks)))
            mc_triggers_tasks[dataset_name] = triggers_task
            mc_triggers_files[dataset_name] = os.path.join(dataset_dir, "triggers", "triggers.txt")

    cookbook.save_db()

    if args.visualize_dependencies:
        visualize_dependencies(cookbook)
        print_graph(cookbook)

if __name__ == "__main__":
    main()
