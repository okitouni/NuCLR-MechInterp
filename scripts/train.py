"""This is supposed to be run from the root directory of the project (where scripts/ is)"""
import torch
import yaml
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import os
# print current working directory

from lib.utils import preds_targets_zn, IO, get_rms, get_rms_no_outliers
from lib.model import get_model_and_optim
from lib.data import prepare_nuclear_data


parser = ArgumentParser()
parser.add_argument("--experiment_name", "-exp", type=str, help="name of experiment to load")
parser.add_argument("--device", "-dev", type=str, help="device to run on")


def fix_mask(data, all_same=True):
    """all_same makes every task have the same mask, otherwise each task has it's own mask given by the order of the task"""
    gen = torch.Generator().manual_seed(42)
    if all_same:
        new_train_mask = torch.rand(data.train_mask.shape[0]//len(data.output_map), generator=gen) < 1
        new_train_mask = new_train_mask.repeat_interleave(len(data.output_map))
    else:
        new_train_mask = [torch.rand(data.train_mask.shape[0]//len(data.output_map), generator=gen) < 0.8 for _ in range(len(data.output_map))]
        new_train_mask = torch.cat(new_train_mask)
    new_train_mask = new_train_mask.to(data.train_mask.device)
    data = data._replace(train_mask=new_train_mask)
    new_val_mask = ~new_train_mask
    data = data._replace(val_mask=new_val_mask)
    return data


if __name__ == "__main__":
    script_args = parser.parse_args()

    result_dir = f"results/{script_args.experiment_name}"
    run_args_file = f"{result_dir}/args.yaml"
    with open(run_args_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**args)
    print("Loaded args:", args, "\n")

    data = prepare_nuclear_data(args)
    print("Loaded Data:", data._fields, data.output_map, "\n")
    # changing the data
    print("Fixing mask")
    data = fix_mask(data, all_same=True)
    args.TRAIN_FRAC = 1.0

    torch.manual_seed(args.SEED)
    

    # setup training data
    X_train = data.X[data.train_mask]
    y_train = data.y[data.train_mask]
    non_nan_targets = ~torch.isnan(y_train.view(-1))
    X_train = X_train[non_nan_targets]
    y_train = y_train[non_nan_targets]

    def quick_eval(model, task="binding", verbose=True, train=False):
        """helper to get the rms for the franken model"""
        preds, targets, zn = preds_targets_zn(data, model, task, train=train)
        rms = get_rms(preds, targets, zn, scale_by_A=True)
        rms_clip = get_rms_no_outliers(preds, targets, zn, scale_by_A=True)
        if verbose:
            print(f"RMS for franken model: {rms:.2f}")
            print(f"RMS for franken model (clipped): {rms_clip:.2f}")
        return rms, rms_clip

    new_model, optim = get_model_and_optim(data, args)


    # train the new model
    for epoch in (pbar:=tqdm(range(args.EPOCHS))):
        optim.zero_grad()
        preds = new_model(X_train)
        preds = preds.gather(1, X_train[:, 2].long().view(-1, 1))
        loss = torch.nn.functional.mse_loss(preds, y_train)
        loss.backward()
        optim.step()
        if epoch % (args.EPOCHS//10) == 0:
            print(f"Epoch {epoch}: {loss.item():.2f}")
            train_rms = quick_eval(new_model, "binding", verbose=False, train=True)
            print(f"Train RMS: {train_rms[0]:.2f} ({train_rms[1]:.2f})", end=" ")
            if data.val_mask.sum() > 0:
                val_rms = quick_eval(new_model, "binding", verbose=False, train=False)
                print(f"Val RMS: {val_rms[0]:.2f} ({val_rms[1]:.2f})")
            if args.SAVE_CKPT:
                os.makedirs(f"{result_dir}/ckpts", exist_ok=True)
                torch.save(new_model.state_dict(), f"{result_dir}/ckpts/model-{epoch}.pt")
        pbar.set_description(f"Epoch {epoch}: {loss.item():.2e}")
        
    # save model
    torch.save(new_model.state_dict(), f"{result_dir}/model.pt")
    IO.save_args(args, f"{result_dir}/args.yaml")
        