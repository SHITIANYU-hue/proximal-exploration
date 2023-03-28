import yaml
from typing import Dict

def load_configs_from_yaml(args, main_cfg_name="main_cfg"):
    cfg = {}
    for arg in vars(args):
        yaml_path = getattr(args, arg)
        if arg == main_cfg_name:
            with open(yaml_path) as f:
                main_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            for main_k, main_v in main_cfg.items():
                cfg[main_k] = main_v
        else:
            with open(yaml_path) as f:
                cfg[arg] = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg