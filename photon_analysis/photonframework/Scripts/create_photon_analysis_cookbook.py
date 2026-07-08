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
    parser.add_argument("--samples", required=True, nargs='+', help="The name of the analysis for which trees should be written")
    parser.add_argument("--visualize-dependencies", action="store_true")
    parser.add_argument("--partition", default="amsclx")
    parser.add_argument("--cores", type=int)
    parser.add_argument("--timelimit-iss", default="03:00")
    parser.add_argument("--timelimit-mc", default="01:00")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--jobtag", help="Prefix for jobnames, to distinguish jobs from multiple running cookbooks.")
    parser.add_argument("--no-title", action='store_true', help='set if additional plots without titles are wanted, does not apply to Skymaps')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')

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
    fermi_diffuse_file = os.path.expandvars("$PHOTONFRAMEWORK/data/gll_iem_v07.fits")
    fermi_source_file = os.path.expandvars("$PHOTONFRAMEWORK/data/gll_psc_v32.xml")

    # write trees
    datadir = args.datadir
    if not os.path.isdir(datadir):
        raise ValueError(f"Expected photon trees in {datadir!r}, but it does not exist.")

    if "photon_analyses" in config:
        for photon_analysis_name, photon_analysis_config in config["photon_analyses"].items():
            if photon_analysis_name not in args.samples and 'all' not in args.samples:
                continue
            photon_analysis_dir = os.path.join(workdir, photon_analysis_name)
            os.makedirs(photon_analysis_dir, exist_ok=True)
            gamma_tree_dir = os.path.join(photon_analysis_dir, "trees")
            os.makedirs(gamma_tree_dir, exist_ok=True)
            sample_name = photon_analysis_config["sample"]
            preselection_name = config["samples"][sample_name].get("preselection", None)
            iss_dataset_name = photon_analysis_config["iss_dataset"]
            mc_signal_dataset_name = photon_analysis_config["mc_signal_dataset"]

            iss_gamma_tree_dir = os.path.join(gamma_tree_dir, "iss")
            mc_gamma_tree_dir = os.path.join(gamma_tree_dir, "mc")
            os.makedirs(iss_gamma_tree_dir, exist_ok=True)
            os.makedirs(mc_gamma_tree_dir, exist_ok=True)
            iss_gamma_tree_arguments = []
            mc_gamma_tree_arguments = []

            iss_photon_tree_list = os.path.join(datadir, "data", iss_dataset_name, "trees", f"PhotonTree{preselection_name}.list" if preselection_name is not None else "trees.list")
            additional_variables_args = []
            if photon_analysis_config.get("additional_variables", []):
                additional_variables_args = ["--variables"] + photon_analysis_config["additional_variables"]

            parallel_total = config["datasets"][iss_dataset_name]["parallel_factor"]
            for parallel_index in range(parallel_total):
                iss_gamma_tree_arguments.append(["--config", args.config, datadir, "--tree", iss_photon_tree_list, "--sample", sample_name, "--nprocesses", str(cores), "--parallel", str(parallel_index), str(parallel_total), "--outputprefix", f"GammaTree_{iss_dataset_name}", "--chunk-size", str(args.chunk_size)] + additional_variables_args)

            for dataset_name in [mc_signal_dataset_name] + photon_analysis_config.get("mc_background_datasets", []):
                photon_tree_list = os.path.join(datadir, "data", dataset_name, "trees", f"PhotonTree{preselection_name}.list")
                parallel_total = config["datasets"][dataset_name]["parallel_factor"]
                for parallel_index in range(parallel_total):
                    mc_gamma_tree_arguments.append(["--config", args.config, datadir, "--tree", photon_tree_list, "--sample", sample_name, "--nprocesses", str(cores), "--parallel", str(parallel_index), str(parallel_total), "--outputprefix", f"GammaTree_{dataset_name}", "--is-mc", "--chunk-size", str(args.chunk_size)] + additional_variables_args)

            with open(os.path.join(iss_gamma_tree_dir, "arguments.list"), "w") as arguments_file:
                for line in iss_gamma_tree_arguments:
                    arguments_file.write(" ".join(line) + "\n")

            with open(os.path.join(mc_gamma_tree_dir, "arguments.list"), "w") as arguments_file:
                for line in mc_gamma_tree_arguments:
                    arguments_file.write(" ".join(line) + "\n")

            iss_gamma_tree_task = cookbook.new_task("joblist",
                comment=f"Write ISS Gamma Trees for {photon_analysis_name}",
                job="write_gamma_tree.py",
                args=f"--arguments '%s' --threads_per_job {cores} --partition {args.partition} --timelimit {args.timelimit_iss}",
                jobs_from=os.path.join(iss_gamma_tree_dir, "arguments.list"),
                directory=iss_gamma_tree_dir,
                ref=f"photons_{photon_analysis_name}_trees_iss",
                depends="none")
            mc_gamma_tree_task = cookbook.new_task("joblist",
                comment=f"Write MC Gamma Trees for {photon_analysis_name}",
                job="write_gamma_tree.py",
                args=f"--arguments '%s' --threads_per_job {cores} --partition {args.partition} --timelimit {args.timelimit_mc}",
                jobs_from=os.path.join(mc_gamma_tree_dir, "arguments.list"),
                directory=mc_gamma_tree_dir,
                ref=f"photons_{photon_analysis_name}_trees_mc",
                depends="none")

            effective_area_dir = os.path.join(photon_analysis_dir, "effective_area")
            os.makedirs(effective_area_dir, exist_ok=True)
            effective_area_args = ["--config", args.config, datadir, "--tree", os.path.join(mc_gamma_tree_dir, "results", f"GammaTree_{mc_signal_dataset_name}_*.root"), "--treename", "GammaTree", "--dataset", mc_signal_dataset_name, "--nprocesses", str(4), "--title", f"\"{config['samples'][sample_name]['label']}\""]
            if args.no_title:
                effective_area_args.extend(['--no-title'])
            if args.transparent:
                effective_area_args.extend(['--transparent'])
            if "comparison" in photon_analysis_config:
                effective_area_args.extend(["--comparison", photon_analysis_config["comparison"]])
            effective_area_task = cookbook.new_task("interactive",
                comment=f"Calculate effective area for {photon_analysis_name}",
                command="calculate_effective_area.py",
                args=" ".join(effective_area_args),
                directory=effective_area_dir,
                ref=f"photons_{photon_analysis_name}_effective_area",
                depends=mc_gamma_tree_task.ref)

            effective_area_flux_binning_dir = os.path.join(photon_analysis_dir, "effective_area_flux_binning")
            os.makedirs(effective_area_flux_binning_dir, exist_ok=True)
            effective_area_flux_binning_args = ["--config", args.config, datadir, "--tree", os.path.join(mc_gamma_tree_dir, "results", f"GammaTree_{mc_signal_dataset_name}_*.root"), "--treename", "GammaTree", "--dataset", mc_signal_dataset_name, "--nprocesses", str(4), "--title", f"\"{config['samples'][sample_name]['label']}\"", "--energy-binning", "flux"]
            if args.no_title:
                effective_area_flux_binning_args.extend(['--no-title'])
            if args.transparent:
                effective_area_flux_binning_args.extend(['--transparent'])
            if "comparison" in photon_analysis_config:
                effective_area_flux_binning_args.extend(["--comparison", photon_analysis_config["comparison"]])
            effective_area_flux_binning_task = cookbook.new_task("interactive",
                comment=f"Calculate effective area with flux binning for {photon_analysis_name}",
                command="calculate_effective_area.py",
                args=" ".join(effective_area_flux_binning_args),
                directory=effective_area_flux_binning_dir,
                ref=f"photons_{photon_analysis_name}_effective_area_flux_binning",
                depends=mc_gamma_tree_task.ref)

            effective_area_background_tasks = []
            for mc_background_dataset_name in photon_analysis_config.get("mc_background_datasets", []):
                effective_area_bkg_dir = os.path.join(photon_analysis_dir, f"effective_area_{mc_background_dataset_name}")
                os.makedirs(effective_area_bkg_dir, exist_ok=True)
                effective_area_args = ["--config", args.config, datadir, "--tree", os.path.join(mc_gamma_tree_dir, "results", f"GammaTree_{mc_background_dataset_name}_*.root"), "--treename", "GammaTree", "--dataset", mc_signal_dataset_name, "--nprocesses", str(4), "--title", f"\"{mc_background_dataset_name} background for {config['samples'][sample_name]['label']}\""]
                if args.no_title:
                    effective_area_args.extend(['--no-title'])
                if args.transparent:
                    effective_area_args.extend(['--transparent'])
                effective_area_background_tasks.append(cookbook.new_task("interactive",
                    comment=f"Calculate effective area for {mc_background_dataset_name} for {photon_analysis_name}",
                    command="calculate_effective_area.py",
                    args=" ".join(effective_area_args),
                    directory=effective_area_bkg_dir,
                    ref=f"photons_{photon_analysis_name}_{mc_background_dataset_name}_effective_area",
                    depends=mc_gamma_tree_task.ref))

            point_spread_function_dir = os.path.join(photon_analysis_dir, "psf")
            os.makedirs(point_spread_function_dir, exist_ok=True)
            point_spread_function_args = ["--config", args.config, datadir, "--tree", os.path.join(mc_gamma_tree_dir, "results", f"GammaTree_{mc_signal_dataset_name}_*.root"), "--treename", "GammaTree", "--dataset", mc_signal_dataset_name, "--nprocesses", str(4), "--title", f"\"{config['samples'][sample_name]['label']}\"", "--n-alpha-bins", str(config["analysis"].get("alpha_bins", 200))]
            if "energy_range" in config["analysis"]:
                point_spread_function_args.extend(["--fit-energy-range", str(config["analysis"]["energy_range"]["min"]), str(config["analysis"]["energy_range"]["max"])])
            if args.no_title:
                point_spread_function_args.extend(['--no-title'])
            if args.transparent:
                point_spread_function_args.extend(['--transparent'])
            if "comparison" in photon_analysis_config:
                point_spread_function_args.extend(["--comparison", photon_analysis_config["comparison"]])
            point_spread_function_task = cookbook.new_task("interactive",
                comment=f"Calculate point spread function for {photon_analysis_name}",
                command="calculate_point_spread_function.py",
                args=" ".join(point_spread_function_args),
                directory=point_spread_function_dir,
                ref=f"photons_{photon_analysis_name}_psf",
                depends=mc_gamma_tree_task.ref)

            energy_resolution_dir = os.path.join(photon_analysis_dir, "energy_resolution")
            os.makedirs(energy_resolution_dir, exist_ok=True)
            energy_resolution_args = ["--config", args.config, workdir, "--tree", os.path.join(mc_gamma_tree_dir, "results", f"GammaTree_{mc_signal_dataset_name}_*.root"), "--treename", "GammaTree", "--dataset", mc_signal_dataset_name, "--nprocesses", str(4), "--title", f"\"{config['samples'][sample_name]['label']}\""]
            if args.no_title:
                energy_resolution_args.extend(['--no-title'])
            if args.transparent:
                energy_resolution_args.extend(['--transparent'])
            if "comparison" in photon_analysis_config:
                energy_resolution_args.extend(["--comparison", photon_analysis_config["comparison"]])
            energy_resolution_task = cookbook.new_task("interactive",
                comment=f"Calculate energy resolution for {photon_analysis_name}",
                command="calculate_energy_resolution.py",
                args=" ".join(energy_resolution_args),
                directory=energy_resolution_dir,
                ref=f"photons_{photon_analysis_name}_energy_resolution",
                depends=mc_gamma_tree_task.ref)
            
            energy_resolution_flux_binning_dir = os.path.join(photon_analysis_dir, "energy_resolution_flux_binning")
            os.makedirs(energy_resolution_flux_binning_dir, exist_ok=True)
            energy_resolution_flux_binning_args = ["--config", args.config, workdir, "--tree", os.path.join(mc_gamma_tree_dir, "results", f"GammaTree_{mc_signal_dataset_name}_*.root"), "--treename", "GammaTree", "--dataset", mc_signal_dataset_name, "--nprocesses", str(4), "--title", f"\"{config['samples'][sample_name]['label']}\"", "--energy-binning", "flux"]
            if args.no_title:
                energy_resolution_flux_binning_args.extend(['--no-title'])
            if args.transparent:
                energy_resolution_flux_binning_args.extend(['--transparent'])
            if "comparison" in photon_analysis_config:
                energy_resolution_flux_binning_args.extend(["--comparison", photon_analysis_config["comparison"]])
            energy_resolution_flux_binning_task = cookbook.new_task("interactive",
                comment=f"Calculate energy resolution with flux binning for {photon_analysis_name}",
                command="calculate_energy_resolution.py",
                args=" ".join(energy_resolution_flux_binning_args),
                directory=energy_resolution_flux_binning_dir,
                ref=f"photons_{photon_analysis_name}_energy_resolution_flux_binning",
                depends=mc_gamma_tree_task.ref)

            skymap_types = {
                "full": ["--outputprefix", "SkyMap_full", "--normalization", "events"],
                "standard": ["--vmin", str(4.94), "--vmax", str(98.678), "--normalization", "events-per-square-degree", "--outputprefix", "SkyMap_standard"],
                "sources": ["--vmin", str(4.94), "--vmax", str(98.678), "--normalization", "events-per-square-degree", "--sources", "--outputprefix", "SkyMap_sources"],
                "full_equatorial": ["--coord", "G", "C", "--normalization", "events", "--outputprefix", "SkyMap_full_equatorial"],
                "standard_equatorial": ["--coord", "G", "C", "--vmin", str(4.94), "--vmax", str(98.678), "--normalization", "events-per-square-degree", "--outputprefix", "SkyMap_standard_equatorial"],
                "flux_binning_skymaps": ["--vmin", str(4.94), "--vmax", str(98.678), "--normalization", "events-per-square-degree", "--outputprefix", "SkyMap_flux_binning", "--energy-binning", "flux"],
            }
            if "energy_range" in config["analysis"]:
                energy_range_min = config["analysis"]["energy_range"]["min"]
                energy_range_max = config["analysis"]["energy_range"]["max"]
                skymap_types["energy_range"] = ["--vmin", str(4.94), "--vmax", str(98.678), "--normalization", "events-per-square-degree", "--outputprefix", "SkyMap_energy_range", "--energy-range", str(conf), str(energy_range_max)]
                skmap_types["energy_range_equatorial"] = ["--coord", "G", "C","--vmin", str(4.94), "--vmax", str(98.678), "--normalization", "events-per-square-degree", "--outputprefix", "SkyMap_energy_range_equatorial", "--energy-range", str(energy_range_min), str(energy_range_max)],
            if "comparison" in photon_analysis_config:
                if photon_analysis_config["comparison"] == "Vertex":
                    skymap_types["bastian"] = ["--vmin", str(0.5), "--vmax", str(10), "--normalization", "events-per-bastian-pixel", "--time-range", "bastian", "--energy-range", str(0.5), str(10), "--outputprefix", "SkyMap_Bastian"]
                    skymap_types["bastian_equatorial"] = ["--coord", "G", "C", "--vmin", str(0.5), "--vmax", str(10), "--normalization", "events-per-bastian-pixel", "--time-range", "bastian", "--energy-range", str(0.5), str(10), "--outputprefix", "SkyMap_Bastian_equatorial"]
                elif photon_analysis_config["comparison"] == "Ecal":
                    skymap_types["bastian"] = ["--vmin", str(0.5), "--vmax", str(7.5), "--normalization", "events-per-bastian-pixel", "--time-range", "bastian", "--energy-range", str(2), str(1000), "--outputprefix", "SkyMap_Bastian"]
                    skymap_types["bastian_equatorial"] = ["--coord", "G", "C", "--vmin", str(0.5), "--vmax", str(7.5), "--normalization", "events-per-bastian-pixel", "--time-range", "bastian", "--energy-range", str(2), str(1000), "--outputprefix", "SkyMap_Bastian_equatorial"]

            skymap_dir = os.path.join(photon_analysis_dir, f"skymaps")
            os.makedirs(skymap_dir, exist_ok=True)
            skymap_args_filename = os.path.join(skymap_dir, "arguments.list")
            nside = config["analysis"].get("nside", 256)
            skymap_title = f"\"{config['samples'][sample_name]['label']}\""
            skymap_general_args = ["--config", args.config, datadir, "--tree", os.path.join(iss_gamma_tree_dir, "results", f"GammaTree_{iss_dataset_name}_*.root"), "--treename", "GammaTree", "--nside", str(nside), "--title", skymap_title, "--nprocesses", str(1), "--psf", os.path.join(point_spread_function_dir, "results", f"PointSpreadFunction.npz"), "--s-bg-regions", os.path.join(repo_region_dir, config["regions"]["signal"]), os.path.join(repo_region_dir, config["regions"]["background"])]#"Signal_Region", "Background_Region"]
            if args.no_title:
                skymap_general_args.extend(['--no-title'])
            if args.transparent:
                skymap_general_args.extend(['--transparent'])
            with open(skymap_args_filename, "w") as skymap_args_file:
                for skymap_typ, skymap_specific_args in skymap_types.items():
                    skymap_args = skymap_general_args + skymap_specific_args
                    skymap_args_file.write(" ".join(skymap_args) + "\n")
            skymap_task = cookbook.new_task("joblist",
                comment=f"Plot the Gamma Ray Skymaps for {photon_analysis_name}",
                job="plot_skymap.py",
                args="--arguments '%s' --batchsystem local --max_jobs 4",
                jobs_from=skymap_args_filename,
                directory=skymap_dir,
                ref=f"photons_{photon_analysis_name}_skymaps",
                depends=",".join([iss_gamma_tree_task.ref, point_spread_function_task.ref]))

            comparison_bastian_plots_dir = os.path.join(photon_analysis_dir, "comparison_bastian")
            os.makedirs(comparison_bastian_plots_dir, exist_ok=True)
            comparison_bastian_plots_args = ["--overallinputdir", photon_analysis_dir, '--effareafile', os.path.join(effective_area_dir, "results", "EffectiveArea.npz"), '--energyfile', os.path.join(energy_resolution_dir, "results", "EnergyResolution.npz"), "--psffile", os.path.join(point_spread_function_dir, "results", "PointSpreadFunction.npz")]
            if args.no_title:
                comparison_bastian_plots_args.extend(['--no-title'])
            if args.transparent:
                comparison_bastian_plots_args.extend(['--transparent'])
            comparison_bastian_plots_task = cookbook.new_task("interactive",
                comment = f"Creates plots to compare to Bastians plots for {photon_analysis_name}",
                command = "make_plots_for_comparison_to_bastian.py",
                args = " ".join(comparison_bastian_plots_args),
                directory=comparison_bastian_plots_dir,
                ref=f"photons_{photon_analysis_name}_comparison_to_bastian_plots",
                depends= ','.join([effective_area_task.ref, point_spread_function_task.ref, energy_resolution_task.ref]))

            fermi_convert_diffuse_model_dir = os.path.join(photon_analysis_dir, "fermi_diffuse")
            os.makedirs(fermi_convert_diffuse_model_dir, exist_ok=True)
            fermi_convert_diffuse_args = ["--fermi-file", fermi_diffuse_file, "--nside", str(nside)]
            if args.no_title:
                fermi_convert_diffuse_args.extend(['--no-title'])
            if args.transparent:
                fermi_convert_diffuse_args.extend(['--transparent'])
            fermi_convert_diffuse_task = cookbook.new_task("interactive", 
                comment = f"Convertes fermi diffuse model from fits to our histograms for {photon_analysis_name}",
                command = "convert_fermi_diffuse.py",
                args = " ".join(fermi_convert_diffuse_args),
                directory = fermi_convert_diffuse_model_dir,
                ref=f"photons_{photon_analysis_name}_convert_fermi_diffuse",
                depends="none")
            
            exposure_maps_dir = os.path.join(photon_analysis_dir, "exposure_maps")
            os.makedirs(exposure_maps_dir, exist_ok=True)
            exposure_maps_args = ["--altitude-histograms", "/hpcwork/jara0052/robin/ph/exposure/altitude_with_pileup/Altitude_with_pileup.npz", "--effective-area", os.path.join(effective_area_flux_binning_dir, "results", "EffectiveArea.npz"), "--nside", str(nside),"--nprocesses", str(4), "--plot-individual", "--colormap", "jet"]
            if args.no_title:
                exposure_maps_args.extend(['--no-title'])
            if args.transparent:
                exposure_maps_args.extend(['--transparent'])
            exposure_maps_task = cookbook.new_task("interactive",
                comment = f"Makes exposure maps for {photon_analysis_name}",
                command = "calculate_exposure_map.py",
                args = " ".join(exposure_maps_args),
                directory=exposure_maps_dir,
                ref=f"photon_{photon_analysis_name}_exposure_maps",
                depends=effective_area_flux_binning_task.ref)
            
            exposure_eq_maps_dir = os.path.join(photon_analysis_dir, "exposure_maps_equatorial")
            os.makedirs(exposure_eq_maps_dir, exist_ok=True)
            exposure_eq_maps_args = ["--altitude-histograms", "/hpcwork/jara0052/robin/ph/exposure/altitude_with_pileup/Altitude_with_pileup.npz", "--effective-area", os.path.join(effective_area_flux_binning_dir, "results", "EffectiveArea.npz"), "--nside", str(nside),"--nprocesses", str(4), "--plot-individual", "--colormap", "jet", "--coord", "G", "C", "--outputprefix", "Equatorial_Exposure"]
            if args.no_title:
                exposure_eq_maps_args.extend(['--no-title'])
            if args.transparent:
                exposure_eq_maps_args.extend(['--transparent'])
            exposure_eq_maps_task = cookbook.new_task("interactive",
                comment = f"Makes exposure maps for {photon_analysis_name} in equatorial coordinates",
                command = "calculate_exposure_map.py",
                args = " ".join(exposure_eq_maps_args),
                directory=exposure_eq_maps_dir,
                ref=f"photon_{photon_analysis_name}_exposure_eq_maps",
                depends=effective_area_flux_binning_task.ref)
            
            fermi_diffuse_convoluted_dir = os.path.join(photon_analysis_dir, "fermi_diffuse_convoluted")
            os.makedirs(fermi_diffuse_convoluted_dir, exist_ok=True)
            fermi_diffuse_convoluted_args = ["--diffuse-file", os.path.join(fermi_convert_diffuse_model_dir, "results", "Fermi_fluxes.npz"), "--exposure", os.path.join(exposure_maps_dir,"results","Exposure_exposure.npz"), "--psf", os.path.join(point_spread_function_dir, "results", "PointSpreadFunction.npz"), "--nside", str(nside)]
            if args.no_title:
                fermi_diffuse_convoluted_args.extend(['--no-title'])
            if args.transparent:
                fermi_diffuse_convoluted_args.extend(['--transparent'])
            fermi_diffuse_convoluted_task = cookbook.new_task("interactive", 
                comment = f"calculates convoluted diffuse fermi model for {photon_analysis_name}",
                command = "create_fermi_diffuse_model.py",
                args = " ".join(fermi_diffuse_convoluted_args),
                directory = fermi_diffuse_convoluted_dir,
                ref=f"photons_{photon_analysis_name}_fermi_diffuse_convoluted",
                depends=",".join([fermi_convert_diffuse_task.ref, point_spread_function_task.ref, exposure_maps_task.ref]))
            
            fermi_source_xml_dir = os.path.join(photon_analysis_dir, "fermi_sources_xml")
            os.makedirs(fermi_source_xml_dir, exist_ok=True)
            fermi_source_xml_args = ["--fermi-file", fermi_source_file, "--exposure", os.path.join(exposure_maps_dir,"results","Exposure_exposure.npz"), "--psf", os.path.join(point_spread_function_dir, "results", "PointSpreadFunction.npz"), "--nside", str(nside)]
            if args.no_title:
                fermi_source_xml_args.extend(['--no-title'])
            if args.transparent:
                fermi_source_xml_args.extend(['--transparent'])
            input_args = " ".join(fermi_source_xml_args)
            fermi_source_xml_task = cookbook.new_task("generic",
                comment = f"Make femri source model for {photon_analysis_name}",
                job = "fermi_source_flux_xml.py",
                args = f'--arguments "{input_args}"',
                nmin = str(1),
                nmax = str(1),
                directory = fermi_source_xml_dir,
                ref=f"photons_{photon_analysis_name}_fermi_sources_xml",
                depends=",".join([point_spread_function_task.ref, exposure_maps_task.ref]))
            
            fermi_bg_full_dir = os.path.join(photon_analysis_dir, "bg_full_model")
            os.makedirs(fermi_bg_full_dir, exist_ok=True)
            fermi_bg_full_args = ["--skymap", os.path.join(skymap_dir, "results","SkyMap_flux_binning_histograms.npz"), "--source-model", os.path.join(fermi_source_xml_dir,"results","Fermi_fluxes.npz"), "--diffuse-model", os.path.join(fermi_diffuse_convoluted_dir, "results", "Fermi_diffuse_model.npz"), "--normalization", "events-per-square-degree", "--s-bg-regions", os.path.join(repo_region_dir, config["regions"]["signal"]), os.path.join(repo_region_dir, config["regions"]["background"]), "--migration", os.path.join(energy_resolution_flux_binning_dir, "results", "EnergyResolution.npz"), "--psf", os.path.join(point_spread_function_dir, "results", "PointSpreadFunction.npz")]
            if args.no_title:
                fermi_bg_full_args.extend(['--no-title'])
            if args.transparent:
                fermi_bg_full_args.extend(['--transparent'])
            fermi_bg_full_task = cookbook.new_task('interactive',
                comment = f"Make femri bg model for {photon_analysis_name}",
                command = "make_bg_and_full_model.py",
                args = " ".join(fermi_bg_full_args),
                directory = fermi_bg_full_dir,
                ref=f"photons_{photon_analysis_name}_fermi_bg",
                depends=",".join([fermi_diffuse_convoluted_task.ref, point_spread_function_task.ref, fermi_source_xml_task.ref, energy_resolution_flux_binning_task.ref, skymap_task.ref]))
            
            


    cookbook.save_db()

    if args.visualize_dependencies:
        visualize_dependencies(cookbook)
        print_graph(cookbook)

if __name__ == "__main__":
    main()
