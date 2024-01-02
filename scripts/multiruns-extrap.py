"""
This experiment is to test generalization capabalities of the model to unseen "heavier" nuclei.
We train on the inner island of nuclei, and test on outer nuclei (nearest neighbors in the training are 1,2, or 3 neutrons away).
"""
import torch
import yaml
import os
import sys
from lib.utils import IO, Slurm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--root", "-r", type=str, help="root directory of project", default=IO.get_root()
)
parser.add_argument("--debug", "-d", action="store_true", help="debug mode")
parser.add_argument("--slurm", "-s", action="store_true", help="run on slurm")


config = {
    "DEPTH": 4,
    "DEV": "cuda",
    "EPOCHS": 25_000,
    "HIDDEN_DIM": 2048,
    "LR": 0.0001,
    "MODEL": "baseline",
    "SIGMOID_READOUT": "false",
    "TMS": "remove",
    "WD": 0.01,
    "DEV": "cuda",
    "TARGETS_CLASSIFICATION": {},
    "TARGETS_REGRESSION": {
        "binding": 100,
        # "binding_bw2": 100,
        # "binding_semf": 1,
        # "z": 1,
        # "n": 1,
        "radius": 0.02,
        "qa": 200,
        "qbm": 200,
        "qbm_n": 200,
        "qec": 200,
        "sn": 200,
        "sp": 200,
    },
    "TRAIN_FRAC": 1,
    "LIPSCHITZ": "false",
    "TRAIN_SET": "all_data",  # random, all_data, extrap_1, extrap_2, extrap_3
    "BATCH_SIZE": 1024,
    "LOG_TIMES": 10,
    "NUCLEI_GE": 0,
    "NUCLEI_HIGH_UNC": "keep",
    "PER_NUCLEON": "true",
    "SAVE_CKPT": True,
}


if __name__ == "__main__":
    experiment_name = "multiruns-extrap"
    args = parser.parse_args()
    ROOT = args.root
    DEBUG = args.debug
    SLURM = args.slurm

    train_sets = ["extrap_1", 
                #   "extrap_2", "extrap_3"
                  ]
    seeds = [0]

    for seed in seeds:
        for train_set in train_sets:
            if DEBUG:
                experiment_name = "DEBUG"

            config["SEED"] = seed
            config["TRAIN_SET"] = train_set
            config["SAVE_CKPT"] = True
            config["DEV"] = "cuda:1" if torch.cuda.is_available() else "cpu"

            # save args to disk
            args_path = os.path.join(ROOT, experiment_name, "args.yaml")
            os.makedirs(os.path.dirname(args_path), exist_ok=True)
            yaml.dump(config, open(args_path, "w"), sort_keys=False)

            # run the pipeline
            try:
                pipeline_cmd = f"python -m scripts.pipeline -exp {experiment_name} --train -r {ROOT}"
                print("Running:", pipeline_cmd)
                if SLURM:
                    Slurm.create_job(pipeline_cmd, experiment_name)
                else:
                    os.system(pipeline_cmd)
            except KeyboardInterrupt:
                print("Interrupted")
                sys.exit()
