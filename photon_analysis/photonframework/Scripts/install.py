#!/usr/bin/env python3

import os
import subprocess as sp

PYTHON_DEPENDENCIES = ["xgboost", "uncertainties", "sympy", "PyUnfold", "iminuit", "gammapy", "astropy", "PyYAML", "healpy"]

def determine_toolchain():
    module_cmd = os.environ["LMOD_CMD"]
    if sp.run([module_cmd, "is-loaded", "foss"], stdout=sp.DEVNULL).returncode == 0:
        return "foss"
    elif sp.run([module_cmd, "is-loaded", "intel"], stdout=sp.DEVNULL).returncode == 0:
        return "intel"
    raise ValueError("Cannot determine loaded toolchain, please load either foss or intel first!")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repository_directory", help="Path to the PhotonFramework repository.")
    parser.add_argument("--virtualenv-directory", default=os.path.join(os.getenv("HOME"), "Software", "virtualenvs", "photonframework"), help="Path to the virtual environment for this repository.")
    parser.add_argument("--module-directory", default=os.path.join(os.getenv("HOME"), ".local", "modules", "all"), help="Path to the user specific module directory.")
    parser.add_argument("--version", default="1.0", help="Version to install the framework module under.")

    args = parser.parse_args()
    
    toolchain = determine_toolchain()

    this_path = os.path.join(args.repository_directory, "Scripts", "install.py")
    if not os.path.exists(this_path):
        raise ValueError(f"Expected to find this script under {this_path!r}, but it's not there.")

    os.makedirs(args.module_directory, exist_ok=True)
    os.makedirs(os.path.join(args.module_directory, "photonframework"), exist_ok=True)
    with open(os.path.join(args.module_directory, "photonframework", f"{args.version}.lua"), "w") as module_file:
        with open(os.path.join(args.repository_directory, "Configuration", "photonframework.lua.tpl")) as template_file:
            content = (template_file.read()
                .replace("%REPOSITORY_DIRECTORY%", args.repository_directory)
                .replace("%VIRTUALENV_DIRECTORY%", args.virtualenv_directory))
            module_file.write(content)

    with open(os.path.join(args.repository_directory, "Configuration", "analysis-job-settings.cfg"), "w") as config_file:
        with open(os.path.join(args.repository_directory, "Configuration", "analysis-job-settings.cfg.tpl")) as template_file:
            content = (template_file.read()
                .replace("%MODULE_DIRECTORY%", args.module_directory)
                .replace("%TOOLCHAIN%", toolchain))
            config_file.write(content)

    with open(os.path.join(args.repository_directory, "Configuration", "analysis-job-settings-no-acsoft.cfg"), "w") as config_file:
        with open(os.path.join(args.repository_directory, "Configuration", "analysis-job-settings-no-acsoft.cfg.tpl")) as template_file:
            content = (template_file.read()
                .replace("%MODULE_DIRECTORY%", args.module_directory)
                .replace("%TOOLCHAIN%", toolchain))
            config_file.write(content)

    os.makedirs(args.virtualenv_directory, exist_ok=True)
    sp.run(["python", "-m", "venv", args.virtualenv_directory])
    sp.run([os.path.join(args.virtualenv_directory, "bin", "pip"), "install"] + PYTHON_DEPENDENCIES)

    print(f"Photon Framework has been installed to {args.module_directory!r}.")
    init_lines = [
        f"module use {args.module_directory}",
        f"module load photonframework",
    ]
    print("Please ensure the following lines are part of your initialization file (e.g. $HOME/.environment.sh):")
    print("\n".join(init_lines))


if __name__ == "__main__":
    main()
