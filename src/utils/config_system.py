"""
config_system.py:  
    Functions for initializing config
        - Read json/jsonnet config files
        - Parse args and override parameters in config files
"""

import os
import argparse

import json
import _jsonnet
from easydict import EasyDict
import time
from pathlib import Path

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided

    try:
        config_dict = json.loads(_jsonnet.evaluate_file(json_file))
        # EasyDict allows to access dict values as attributes (works recursively).
        config = EasyDict(config_dict)
        return config, config_dict
    except ValueError:
        print("INVALID JSON file.. Please provide a good json file")
        exit(-1)

def process_config(args):
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    path = Path(script_dir).parent
    config, _ = get_config_from_json(args.config)

    # Some default paths
    if not config.DATA_FOLDER:
        # Default path
        config.DATA_FOLDER = os.path.join(str(path), 'Data')
    if not config.EXPERIMENT_FOLDER:
        # Default path
        config.EXPERIMENT_FOLDER = os.path.join(str(path), 'Experiments')
    if not config.TENSORBOARD_FOLDER:
        # Default path
        config.TENSORBOARD_FOLDER = os.path.join(str(path), 'Data_TB', 'tb_logs')

    # Some args to configs defined specifically for this project

    # Override config data using passed parameters
    config.reset = args.reset

    if args.experiment_name != '':
        config.experiment_name = args.experiment_name



    config = parse_optional_args(config, args)

    # Generated Paths
    config.log_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.experiment_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.checkpoint_path = os.path.join(config.experiment_path, "models")
    config.results_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.results_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.tensorboard_path = os.path.join(config.TENSORBOARD_FOLDER, config.experiment_name)
    config.WANDB.tags = config.WANDB.tags + args.tags

    # change args to dict, and save to config
    def namespace_to_dict(namespace):
        return EasyDict({
            k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
            for k, v in vars(namespace).items()
        })
    config.args = namespace_to_dict(args)
    return config

def parse_optional_args(config, args):
    """Parse optional arguments and override the config file

    Args:
        config (dict): config dict to be used
        args (argparser Namespace): arguments from command-line input

    Returns:
        config: updated config dict
    """
    
    opts=args.opts

    for opt in opts:
        path, value = opt.split('=')
        try:
            value = eval(value)
        except:
            value = str(value)
            print('input value {} is not a number, parse to string.')
        
        config_path_list = path.split('.')
        depth = len(config_path_list)
        if depth == 1:
            config[config_path_list[0]] = value
        elif depth == 2:
            config[config_path_list[0]][config_path_list[1]] = value
        elif depth == 3:
            config[config_path_list[0]][config_path_list[1]][config_path_list[2]] = value
        elif depth == 4:
            config[config_path_list[0]][config_path_list[1]][config_path_list[2]][config_path_list[3]] = value
        elif depth == 5:
            config[config_path_list[0]][config_path_list[1]][config_path_list[2]][config_path_list[3]][config_path_list[4]] = value
        elif depth == 6:
            config[config_path_list[0]][config_path_list[1]][config_path_list[2]][config_path_list[3]][config_path_list[4]][config_path_list[5]] = value
        else:
            raise('Support up to depth=6. Please do not hierarchy the config file too deep.')
            
    return config
