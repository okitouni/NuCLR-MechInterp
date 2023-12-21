import torch
import yaml
import os
import sys
from lib.utils import IO

ROOT = IO.get_root()

args = {
    "DEPTH": 2,
    "DEV": "cuda",
    "EPOCHS": 50000,
    "HIDDEN_DIM": 2048,
    "LR": 0.0001,
    "MODEL": "baseline",
    "ROOT": "/work/submit/kitouni/ai-nuclear",
    "SIGMOID_READOUT": "false",
    "TMS": "remove",
    "WD": 0.01,
    "DEV": "cuda",
    "TARGETS_CLASSIFICATION": {},
    "TARGETS_REGRESSION": {},
    "TRAIN_FRAC": 1,
    "LIPSCHITZ": "false",
}



if __name__ == "__main__":
    # load args from disk
    # seeds = [0, 1, 2, 3, 4]
    seeds = [2]
    targets_regression = [
        # {
        #     "binding": 1,
        #     # "binding_semf": 1,
        #     # "z": 1,
        #     # "n": 1,
        #     "radius": 1,
        #     "qa": 1,
        #     "qbm": 1,
        #     "qbm_n": 1,
        #     "qec": 1,
        #     "sn": 1,
        #     "sp": 1,
        # },
        # {
        #     "binding": 1,
        # },
        # {
        #     "radius": 1,
        # },
        {
            "qa": 1,
        },
        {
            "sp": 1,
        },
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in seeds:
        for targets in targets_regression:
            if len(targets) == 1:
                target_name = list(targets.keys())[0]
                experiment_name = f"{target_name}_{seed}"
            else:
                experiment_name = f"all_{seed}"

            args["TARGETS_REGRESSION"] = targets
            args["SEED"] = seed
            args["SAVE_CKPT"] = True

            # save args to disk
            args_path = os.path.join(ROOT, experiment_name, "args.yaml")
            os.makedirs(os.path.dirname(args_path), exist_ok=True)
            yaml.dump(args, open(args_path, "w"), sort_keys=False)
            
            # run the pipeline
            try:
                pipeline_cmd = f"python -m scripts.pipeline -exp {experiment_name} --train --plot -r {ROOT}"
                print("Running:", pipeline_cmd)
                os.system(pipeline_cmd)
            except KeyboardInterrupt:
                print("Interrupted")
                sys.exit()

