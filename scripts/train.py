"""This is supposed to be run from the root directory of the project (where scripts/ is)"""
import torch
import yaml
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import os
import wandb

# print current working directory

from lib.utils import preds_targets_zn, IO, get_rms, get_rms_no_outliers
from lib.model import get_model_and_optim
from lib.data import prepare_nuclear_data


parser = ArgumentParser()
parser.add_argument(
    "--experiment_name", "-exp", type=str, help="name of experiment to load"
)
parser.add_argument("--device", "-dev", type=str, help="device to run on")
parser.add_argument(
    "--root", "-r", type=str, help="logdir", default="./results"
)
parser.add_argument("--name", "-n", type=str, help="name of run", default=None)

if __name__ == "__main__":
    # root is the parent of the directory containing this script
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_args = parser.parse_args()

    result_dir = f"{script_args.root}/{script_args.experiment_name}"
    run_args_file = f"{result_dir}/args.yaml"
    with open(run_args_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**args)
    print("Loaded args:", args, "\n")
    main_task_name = ( 
        "binding"
        if "binding" in args.TARGETS_REGRESSION
        else list(args.TARGETS_REGRESSION.keys())[0]
    )

    if args.WANDB == "true":
        tags = args.TAGS if hasattr(args, "TAGS") else []
        wandb.init(project="nuclr-mechinterp", tags=tags, dir='/tmp/kitouni', name=script_args.name)
        wandb.config.update(args)
        # save code as artifact
        # wandb.save(f"{root}/scripts")
        # wandb.save(f"{root}/lib")

    data = prepare_nuclear_data(args)

    torch.manual_seed(args.SEED)
    # setup training data
    X_train = data.X[data.train_mask]
    y_train = data.y[data.train_mask]
    non_nan_targets = ~torch.isnan(y_train.view(-1))
    X_train = X_train[non_nan_targets]
    y_train = y_train[non_nan_targets]

    def quick_eval(model, task="binding", train=False):
        preds, targets, zn = preds_targets_zn(data, model, task, train=train)
        rms = get_rms(preds, targets, zn, scale_by_A=args.PER_NUCLEON == "true")
        rms_clip = get_rms_no_outliers(
            preds, targets, zn, scale_by_A=args.PER_NUCLEON == "true"
        )
        return rms, rms_clip

    def transform(tensor):
        min_ = torch.tensor(
            data.regression_transformer.data_min_.tolist(), device=tensor.device
        )
        max_ = torch.tensor(
            data.regression_transformer.data_max_.tolist(), device=tensor.device
        )
        return (tensor * (max_ - min_)) + min_

    new_model, optim = get_model_and_optim(data, args)

    # shuffle indices to make batches
    indices = torch.arange(X_train.shape[0])

    # train the new model
    loss_weights = 1 / torch.tensor(
        list(args.TARGETS_REGRESSION.values()), device=X_train.device
    ).view(-1)

    pbar = range(args.EPOCHS)
    if args.VERBOSITY > 1:
        pbar = tqdm(pbar)
    bs = args.BATCH_SIZE if args.BATCH_SIZE > 1 else int(X_train.shape[0] * args.BATCH_SIZE)
    for epoch in pbar:
        torch.randperm(X_train.shape[0], out=indices)
        for batch_idx in range(0, X_train.shape[0]//bs * bs, bs):
            batch = indices[batch_idx : batch_idx + bs]
            X_batch = X_train[batch]
            y_batch = y_train[batch]

            optim.zero_grad()
            preds = transform(new_model(X_batch))
            preds = preds.gather(1, X_batch[:, 2].long().view(-1, 1))
            loss = torch.nn.functional.mse_loss(preds, y_batch, reduction="none")
            weights = loss_weights[X_batch[:, [2]].long()]
            loss = (loss * weights).mean()
            loss.backward()
            optim.step()
            if args.VERBOSITY > 1:
                pbar.set_description(f"Epoch {epoch}: {loss.item():.2e}")

            if args.WANDB == "true":
                wandb.log({"train/loss": loss.item()})

        if epoch % (args.EPOCHS // args.LOG_TIMES) == 0 and args.VERBOSITY > 0:
            print(f"Epoch {epoch}: {loss.item():.2f}")
            train_rms = quick_eval(new_model, main_task_name, train=True)
            print(f"Train RMS: {train_rms[0]:.2f} ({train_rms[1]:.2f})", end=" ")
            if args.WANDB == "true":
                wandb.log({f"train/{main_task_name}/rms": train_rms[0]})
            if data.val_mask.sum() > 0:
                val_rms = quick_eval(
                    new_model, main_task_name, train=False
                )
                print(f"Val RMS: {val_rms[0]:.2f} ({val_rms[1]:.2f})")
                if args.WANDB == "true":
                    wandb.log({f"val/{main_task_name}/rms": val_rms[0]})
        if epoch % (args.EPOCHS // args.SAVE_CKPT) ==0:
            os.makedirs(f"{result_dir}/ckpts", exist_ok=True)
            torch.save(
                new_model.state_dict(), f"{result_dir}/ckpts/model-{epoch}.pt"
            )

    # save model
    torch.save(new_model.state_dict(), f"{result_dir}/ckpts/model.pt")
    IO.save_args(args, f"{result_dir}/args.yaml")
