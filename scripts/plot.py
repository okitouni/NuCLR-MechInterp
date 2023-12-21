import os
from lib.utils import IO, PlottingContext
from lib.data import prepare_nuclear_data
from lib.model import get_model_and_optim
import torch
from argparse import ArgumentParser
from glob import glob

parser = ArgumentParser()
parser.add_argument(
    "--experiment_name", "-exp", type=str, help="name of experiment to load"
)
parser.add_argument(
    "--all_ckpts",
    "-a",
    action="store_true",
    help="plot all checkpoints",
)

parser.add_argument("--root", "-r", type=str, help="root directory of project", default="./results")


if __name__ == "__main__":
    scripts_args = parser.parse_args()
    experiment_name = scripts_args.experiment_name

    run_args = IO.load_args(os.path.join(scripts_args.root, experiment_name, "args.yaml"))
    data = prepare_nuclear_data(run_args)
    new_model, _ = get_model_and_optim(data, run_args)

    models = {"final": os.path.join(scripts_args.root, experiment_name, "model.pt")}
    if scripts_args.all_ckpts:
        ckpts = glob(os.path.join(scripts_args.root, experiment_name, "ckpts", "*.pt"))
        # names are of the form model-epoch.pt
        epoch = lambda x: int(x.split("-")[-1].split(".")[0])
        models.update({epoch(ckpt): ckpt for ckpt in ckpts})
        
    for name, ckpt in models.items():
        new_model.load_state_dict(torch.load(ckpt))
        for i, type in enumerate(["Z", "N"]):
            embed = new_model.emb[i].detach().cpu().numpy()
            fig = PlottingContext.plot_embedding(embed, num_components=10)

            img_path = os.path.join(
                scripts_args.root, experiment_name, "plots", f"embedding_{type}_{name}.png"
            )
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            fig.savefig(img_path)
            print("Saved embedding to", img_path)