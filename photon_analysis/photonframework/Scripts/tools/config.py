
import yaml

import os

def get_config(filename=None):
    if filename is None:
        filename = os.environ.get("PHOTON_CONFIG", None)
    if filename is None:
        raise ValueError("$PHOTON_CONFIG not set.")
    with open(filename) as config_file:
        config = yaml.safe_load(config_file)
    if isinstance(config["datasets"], str):
        with open(os.path.join(os.environ["PHOTONFRAMEWORK"], "Configuration", config["datasets"])) as data_config_file:
            config["datasets"] = yaml.safe_load(data_config_file)
    return config


def get_default_energy_estimator(config):
    return config["analysis"]["energy_estimator"]
