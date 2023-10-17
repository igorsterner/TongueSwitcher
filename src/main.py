"""
main.py:  
    Start functions
        - Read json/jsonnet config files
        - Parse args and override parameters in config files
        - Find selected data and initialize
        - Run processors
"""

import argparse
import json
import logging
import os
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint

import wandb

logger = logging.getLogger(__name__)


from transformers import (AutoModelForMaskedLM,
                          AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from data_loader_manager import *
from tongueswitcher import *
from utils.config_system import process_config
from utils.dirs import *
from utils.metrics_log_callback import MetricsHistoryLogger
from utils.seed import set_seed


def get_checkpoint_model_path(
    saved_model_path, load_epoch=-1, load_best_model=False, load_model_path=""
):
    if load_model_path:
        path_save_model = load_model_path
        if not os.path.exists(path_save_model):
            raise FileNotFoundError(
                "Model file not found: {}".format(path_save_model)
            )
    else:
        if load_best_model:
            file_name = "best.ckpt"
        else:
            if load_epoch == -1:
                file_name = "last.ckpt"
            else:
                file_name = "epoch_{}.ckpt".format(load_epoch)

        path_save_model = os.path.join(saved_model_path, file_name)
        if not os.path.exists(path_save_model):
            logger.warning(
                "No checkpoint exists from '{}'. Skipping...".format(
                    path_save_model
                )
            )
            logger.info("**First time to train**")
            return (
                ""  # return empty string to indicate that no model is loaded
            )
        else:
            logger.info("Loading checkpoint from '{}'".format(path_save_model))
    return path_save_model

def reset_folders(dirs):
    for dir in dirs:
        try:
            delete_dir(dir)
        except Exception as e:
            print(e)

def reset_wandb_runs(all_runs):
    for run in all_runs:
        logger.info(f'Deleting wandb run: {run}')
        run.delete()

def main(config):

    pprint(config)
    if config.seed:
        set_seed(config.seed)
        logger.info(f"All seeds have been set to {config.seed}")

    # Wandb logger
    logger.info(
        "init wandb logger with the following settings: {}".format(
            config.WANDB
        )
    )

    metrics_history_logger = MetricsHistoryLogger()

    additional_args = {
        "logger": [metrics_history_logger],
    }

    DataLoaderWrapper = globals()[config.data_loader.type]

    if DataLoaderWrapper is not None:
        if args.mode == "tongueswitcher":
            data_loader_manager = DataLoaderWrapper(config)
        elif args.mode == "bert_pretraining" or args.mode == "tsbert_classification":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            print(f"Loading tokenizer from {config.bert_model_checkpoint}")
            tokenizer = AutoTokenizer.from_pretrained(config.bert_model_checkpoint, use_fast=True)
            data_loader_manager = DataLoaderWrapper(config, tokenizer)
        else:
            print("Unrecognized mode. Please choose either 'tongueswitcher',  'bert_pretraining' or 'tsbert_classification'")
    else:
        raise ValueError(f"Data loader {config.data_loader.type} not found")

    data_loader_manager.build_dataset()

    # After Initialization, save config files
    with open(os.path.join(config.experiment_path, "config.jsonnet"), "w") as config_file:
        save_config = config.copy()
        json.dump(save_config, config_file, indent=4)
        logger.info(f"config file was successfully saved to {config.experiment_path} for future use.")

    run = wandb.init(
        project = config.WANDB.project,
        entity = config.WANDB.entity
    )
    wandb.run.name = config.experiment_name
    
    if args.mode == 'tongueswitcher':

        Executor = globals()[config.executor.type]
        executor = Executor(config, data_loader_manager)

        executor.build_corpus()

    elif args.mode == 'bert_pretraining':

        model = AutoModelForMaskedLM.from_pretrained(config.bert_model_checkpoint)

        training_args = TrainingArguments(
            output_dir = config.checkpoint_path,
            evaluation_strategy="steps",
            eval_steps = 10000,
            per_device_train_batch_size=config.batch_size,  # Adjust batch size as desired
            per_device_eval_batch_size=config.batch_size,  # Adjust batch size as desired
            report_to='wandb',  # Integration with Weights & Biases
            save_strategy = 'epoch',
            run_name=args.experiment_name,  # Name of the run (experiment)
            num_train_epochs=config.train_epochs,  # Set the number of training epochs to 1
            learning_rate=config.learning_rate,
            warmup_steps=10000,
            seed=config.seed,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_loader_manager.data.train_tokenized_datasets,
            eval_dataset=data_loader_manager.data.eval_tokenized_datasets,  # Add the eval_dataset parameter
            data_collator=data_loader_manager.data.data_collator,
            tokenizer=tokenizer
        )

        trainer.train()

    elif args.mode == "tsbert_classification":

        id2label = {0: 'E', 1: 'M', 2: 'D'}
        label2id = {'E': 0, 'M': 2, 'D': 2}
        
        id2label = {0: 'D', 1: 'M', 2: 'E', 3: 'O', 4: 'SE', 5: 'SD', 6: 'SO'}
        label2id = {'D': 0, 'M': 1, 'E': 2, 'O': 3, 'SE': 4, 'SD': 5, 'SO': 6}

        model = AutoModelForTokenClassification.from_pretrained(config.bert_model_checkpoint, num_labels=7, id2label=id2label, label2id=label2id)

        training_args = TrainingArguments(
            output_dir = config.checkpoint_path,
            evaluation_strategy="no",
            per_device_train_batch_size=config.batch_size,  # Adjust batch size as desired
            per_device_eval_batch_size=config.batch_size,  # Adjust batch size as desired
            report_to='wandb',  # Integration with Weights & Biases
            save_strategy='epoch',
            run_name=args.experiment_name,  # Name of the run (experiment)
            num_train_epochs=config.train_epochs,  # Set the number of training epochs to 1
            learning_rate=config.learning_rate,
            save_total_limit=1,
            seed=config.seed,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_loader_manager.data.data_collator,
            train_dataset=data_loader_manager.data.train_encodings,
            tokenizer=tokenizer
        )

        trainer.train()

    else:
        print("Unrecognized mode. Please choose either 'tongueswitcher', 'mbert_pretraining' or 'mbert_classification'")


def initialization(args):
    # ===== Process Config =======
    config = process_config(args)
    # print(config)
    if config is None:
        return None
    # Create Dirs

    dirs = [
        config.results_path,
    ]

    delete_confirm = "n"

    if config.reset and config.mode == "train":
        # Reset all the folders
        print("You are deleting following dirs: ", dirs, "input y to continue")
        delete_confirm = input()
        if delete_confirm == "y":
            reset_folders(dirs)
        else:
            print("reset cancelled.")

    create_dirs(dirs)
    print(dirs)

    # ====== Set Logger =====
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s (in %(pathname)s:%(lineno)d)"
    log_console_format = "[%(levelname)s] - %(name)s : %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))
    from utils.color_logging import CustomFormatter

    custom_output_formatter = CustomFormatter(custom_format=log_console_format)
    # console_handler.setFormatter(custom_output_formatter)

    info_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "info.log"),
        maxBytes=10**6,
        backupCount=5,
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(Formatter(log_file_format))

    exp_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "debug.log"),
        maxBytes=10**6,
        backupCount=5,
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "error.log"),
        maxBytes=10**6,
        backupCount=5,
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(info_file_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

    # setup a hook to log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            if wandb.run is not None:
                logger.error(f"Attempting to stop the wandb run {wandb.run}")
                wandb.finish()  # stop wandb if keyboard interrupt is raised
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        logger.error(
            f"Uncaught exception: {exc_type} --> {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception

    # setup wandb
    WANDB_CACHE_DIR = config.WANDB.pop("CACHE_DIR")
    if WANDB_CACHE_DIR:
        os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE_DIR

    all_runs = wandb.Api(timeout=19).runs(
        path=f"{config.WANDB.entity}/{config.WANDB.project}",
        filters={"config.experiment_name": config.experiment_name},
    )
    if config.reset and config.mode == "train" and delete_confirm == "y":
        reset_wandb_runs(all_runs)
        config.WANDB.name = config.experiment_name
    else:
        if len(all_runs) > 0:
            config.WANDB.id = all_runs[0].id
            config.WANDB.resume = "must"
            config.WANDB.name = config.experiment_name
        else:
            config.WANDB.name = config.experiment_name

    # logger.info(f"Initialization done with the config: {str(config)}")
    return config


def parse_args_sys(args_list=None):
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "config",
        metavar="config_json_file",
        default="None",
        help="The Configuration file in json format",
    )
    arg_parser.add_argument(
        "--mode",
        default="",
        help="Supported modes: mbert_pretraining, mbert_classification, tongueswitcher",
    )
    arg_parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.",
    )
    arg_parser.add_argument(
        "--playground_input",
        action="store_true",
        default=False,
        help="Use custom tweets",
    )
    arg_parser.add_argument(
        "--csw_cache",
        default=False,
        help="Use csw tweets",
    )
    arg_parser.add_argument(
        "--tags", 
        nargs="*", 
        default=[], 
        help="Add tags to the wandb logger"
    )
    arg_parser.add_argument(
        "--log_prediction_tables",
        action="store_true",
        default=False,
        help="Log prediction tables.",
    )
    arg_parser.add_argument(
        "--reset_dictionaries",
        action="store_true",
        default=False,
        help="Reform dictionaries",
    )    
    arg_parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Reset the corresponding folder under the experiment_name",
    )
    arg_parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    return args


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    config = initialization(args)
    if config is None:
        exit(0)
    main(config)
