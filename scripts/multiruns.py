import torch
import yaml
import os
import sys
from lib.utils import IO

DEBUG = False
ROOT = IO.get_root()

args = {
    "DEPTH": 4,
    "DEV": "cuda",
    "EPOCHS": 20_000,
    "HIDDEN_DIM": 2048,
    "LR": 0.0001,
    "MODEL": "baseline",
    "SIGMOID_READOUT": "false",
    "TMS": "remove",
    "WD": 0.001,
    "DEV": "cuda",
    "TARGETS_CLASSIFICATION": {},
    "TARGETS_REGRESSION": {},
    "TRAIN_FRAC": 1,
    "LIPSCHITZ": "false",
    "TRAIN_SET": "all_data", # random, all_data, extrap_1, extrap_2, extrap_3
    "BATCH_SIZE": 1024,
    "LOG_TIMES": 10,
    "NUCLEI_GE": 0,
    "NUCLEI_HIGH_UNC": "keep",
    "PER_NUCLEON": "false",
}



if __name__ == "__main__":
    # load args from disk
    # seeds = [0, 1, 2, 3, 4]
    seeds = [2]
    targets_regression = [
        {
            "binding": 1,
            # "binding_semf": 1,
            # "z": 1,
            # "n": 1,
            "radius": 1,
            "qa": 1,
            "qbm": 1,
            "qbm_n": 1,
            "qec": 1,
            # "sn": 1,
            # "sp": 1,
        },
        # {
        #     "binding_semf": 1,
        # },
        # {
        #     "binding": 1,
        # },
        # {
        #     "radius": 1,
        # },
        # {
        #     "qa": 1,
        # },
        # {
        #     "sp": 1,
        # },
    ]

    train_sets = [
        "extrap_1", "extrap_2", "extrap_3",
        # "all_data",
        ]

    # for seed in seeds:
    seed = 0
    for targets in targets_regression:
        for train_set in train_sets:
            if len(targets) == 1:
                target_name = list(targets.keys())[0]
                experiment_name = f"{target_name}_{seed}"
            else:
                experiment_name = f"all_{seed}"
            experiment_name += f"_{train_set}"
            experiment_name += f"_{args['EPOCHS']//1000}k"
            experiment_name += f"_no_zn_{args['DEPTH']}depth"
            experiment_name += f"_per_nucleon" if args["PER_NUCLEON"] == "true" else ""


            if DEBUG:
                experiment_name = "DEBUG"


            args["TARGETS_REGRESSION"] = targets
            args["SEED"] = seed
            args["TRAIN_SET"] = train_set
            args["SAVE_CKPT"] = True
            args["DEV"] = "cuda:1" if torch.cuda.is_available() else "cpu"
        

            # save args to disk
            args_path = os.path.join(ROOT, experiment_name, "args.yaml")
            os.makedirs(os.path.dirname(args_path), exist_ok=True)
            yaml.dump(args, open(args_path, "w"), sort_keys=False)
            
            # run the pipeline
            try:
                pipeline_cmd = f"python -m scripts.pipeline -exp {experiment_name} --train -r {ROOT}"
                print("Running:", pipeline_cmd)
                os.system(pipeline_cmd)
            except KeyboardInterrupt:
                print("Interrupted")
                sys.exit()

