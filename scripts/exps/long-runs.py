"""
This experiment is to generate representations based on all the training data.
We train for a very long time with small learning rate.
"""
import torch
import yaml
import os
import subprocess
import sys
from lib.utils import IO, Slurm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--root", "-r", type=str, help="root directory of project", default=IO.get_root()
)
parser.add_argument("--debug", "-d", action="store_true", help="debug mode")
parser.add_argument("--slurm", "-s", action="store_true", help="run on slurm")
parser.add_argument("--wandb", "-w", action="store_true", help="save to wandb")


config = {
    "DEPTH": 2,
    "DEV": "cuda",
    "EPOCHS": 100_000,
    "HIDDEN_DIM": 1024,
    "LR": 3e-4,
    "MODEL": "baseline",
    "SIGMOID_READOUT": "false",
    "TMS": "remove",
    "WD": 0.01,
    "TARGETS_CLASSIFICATION": {},
    "TARGETS_REGRESSION": {},
    # "TRAIN_FRAC": .7,
    "LIPSCHITZ": "false",
    "TRAIN_SET": "random-all_same",  # random, all_data, extrap_1, extrap_2, extrap_3, random-all_same
    "BATCH_SIZE": 0.2, # if less than one then it's a fraction of the dataset, otherwise it's the batch size
    "NUCLEI_GE": 8,
    "NUCLEI_HIGH_UNC": "keep",
    "PER_NUCLEON": "false",
    "LOG_TIMES": 100,
    "SAVE_CKPT": 100,
    "VERBOSITY": 1,
    "TAGS": ["long-run"],
}


if __name__ == "__main__":
    args = parser.parse_args()
    ROOT = args.root
    DEBUG = args.debug
    SLURM = args.slurm

    targets = [
        # {"binding_semf": 100},
        # {"binding": 100},
        # {"z": 1},
        # {"n": 1},
        # {"radius": 0.02},
        # {"qa": 200},
        # {"qbm": 200},
        # {"qbm_n": 200},
        # {"qec": 200},
        # {"sn": 200},
        # {"sp": 200},
        {
            "binding": 100,
            "z": 1,
            "n": 1,
            "radius": 0.02,
            "qa": 200,
            "qbm": 200,
            "qbm_n": 200,
            "qec": 200,
            "sn": 200,
            "sp": 200,
        },
    ]
    seeds = [0,1,2]
    train_fracs = [i*0.1 for i in range(1,10)]

    for train_frac in train_fracs:
      for seed in seeds:
          for target in targets:
              target_name = "+".join([f"{k}{v}" for k, v in target.items()])
              config["SEED"] = seed
              config["TARGETS_REGRESSION"] = target
              config["VERBOSITY"] = 1 if SLURM else 2
              config["WANDB"] = "true" if args.wandb else "false"
              config["TRAIN_FRAC"] = train_frac
              experiment_name = f"epochs-{config['EPOCHS']}"
              experiment_name += f"-{target_name}-seed{seed}-dataseed{seed}-trainfrac{train_frac:.1f}-nuclei_ge{config['NUCLEI_GE']}-hiddendim{config['HIDDEN_DIM']}-LR{config['LR']}"

              # save args to disk
              args_path = os.path.join(ROOT, experiment_name, "args.yaml")
              os.makedirs(os.path.dirname(args_path), exist_ok=True)
              yaml.dump(config, open(args_path, "w"), sort_keys=False)

              # run the pipeline
              which_python = (
                  subprocess.check_output("which python", shell=True).decode("ascii").strip()
              )
              try:
                  pipeline_cmd = f"{which_python} -m scripts.pipeline -exp {experiment_name} --train -r {ROOT} --name {experiment_name}"
                  print("Running:", pipeline_cmd)
                  if not DEBUG:
                      if SLURM:
                          Slurm.create_job(pipeline_cmd, experiment_name)
                      else:
                          os.system(pipeline_cmd)
              except KeyboardInterrupt:
                  print("Interrupted")
                  sys.exit()
