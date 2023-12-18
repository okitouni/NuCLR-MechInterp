from argparse import ArgumentParser
import torch
import yaml
import os


parser = ArgumentParser()
parser.add_argument(
    "--experiment_name", "-exp", type=str, help="name of experiment to load"
)
parser.add_argument(
    "--train", "-t", action="store_true", help="train a new model", required=False
)
parser.add_argument(
    "--plot", "-p", action="store_true", help="plot the embeddings", required=False
)

pipeline_args = parser.parse_args()
experiment_name = pipeline_args.experiment_name

if pipeline_args.train:
    # train the model
    train_cmd = f"python -m scripts.train -exp {experiment_name}"
    print("Running:", train_cmd)
    os.system(train_cmd)

if pipeline_args.plot:
    plot_cmd = f"python -m scripts.plot -exp {experiment_name} -a"
    print("Running:", plot_cmd)
    os.system(plot_cmd)
