from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import yaml
from argparse import Namespace
import os
import subprocess
from pathlib import Path



def preds_targets_zn(data, model, task_name, train=True):
    # the data has an admittedly weird structure
    # data.X is a tensor of shape (N, 3) where N is the number of nuclei
    # TIMES the number of tasks. The first column is the number of protons,
    # the second column is the number of neutrons, and the third column is
    # the task index.
    task_names = list(
        data.output_map.keys()
    )  # get a list of names of tasks (e.g. binding)
    task_idx = task_names.index(task_name)

    mask = data.train_mask if train else data.val_mask
    X_train = data.X[mask]
    scatter = X_train[:, 2].cpu().numpy() == task_idx  # get only rows relevant to task

    # get the targets and predictions for the task
    # first, we need to undo the preprocessing
    # data.regresion_transformer is a sklearn transformer that does the preprocessing
    # we can use its inverse_transform method to undo the preprocessing
    # it expects a numpy array, of shape (samples, features) where features is the number
    # of tasks we have.
    targets = data.y.view(-1, len(data.output_map)).cpu().numpy()
    # targets = data.regression_transformer.inverse_transform(targets)
    targets = targets.flatten()[mask.cpu().numpy()]
    targets = targets[scatter]

    # Predictions on the other hand are shape (samples, tasks)
    # each row has one correct prediction, and the rest are not useful
    # this is not optimal but not worth optimizing for now
    preds = model(data.X[mask])
    preds = preds.cpu().detach().numpy()
    preds = data.regression_transformer.inverse_transform(preds)[scatter, task_idx]
    return preds, targets, X_train[scatter, :2].cpu().numpy()


def _get_residuals(preds, targets, zn, scale_by_A=False):
    non_nan_targets = ~np.isnan(targets)
    targets = targets[non_nan_targets]
    preds = preds[non_nan_targets]
    zn = zn[non_nan_targets]
    if scale_by_A:
        factor = zn.sum(1)
    else:
        factor = 1
    diff = (preds - targets) * factor
    return diff


def get_rms(preds, targets, zn, scale_by_A=False):
    diff = _get_residuals(preds, targets, zn, scale_by_A)
    rms = np.sqrt(np.mean(diff**2))
    return rms


def get_rms_no_outliers(preds, targets, zn, percentile=99, scale_by_A=False):
    diff = _get_residuals(preds, targets, zn, scale_by_A)
    diff = np.abs(diff)
    thresh = np.percentile(diff, percentile)
    diff = diff[diff < thresh]
    rms = np.sqrt(np.mean(diff**2))
    return rms


class PlottingContext:
    def plot_predictions(data, model, task_name, train=True):
        preds, targets, zn = preds_targets_zn(data, model, task_name, train)
        rms = get_rms(preds, targets, zn, scale_by_A="binding"in task_name)

        plt.plot(targets, preds, "o")
        plt.xlabel("Target")
        plt.ylabel("Prediction")
        plt.title(f"Predictions vs Targets for {task_name}, RMS={rms:.2f}")
        plt.show()
        print(f"RMS: {rms:.2f}")

    def scatter_text(strings, x, y, colors=None, ax=None, fontsize=12, **kwargs):
        if "ha" not in kwargs.keys():
            kwargs["ha"] = "left"
        if "va" not in kwargs.keys():
            kwargs["va"] = "bottom"

        strings = [str(s) for s in strings]
        colors = colors if colors is not None else ["k"] * len(x)
        if not ax:
            ax = plt.gca()
        for s, x, y, c in zip(strings, x, y, colors):
            ax.text(x, y, s, fontsize=fontsize, c=c, **kwargs)
        return ax

    def plot_embedding(embed, type="PCA", num_components=2, figscale=1):
        if type == "PCA":
            pca = PCA(n_components=num_components)
            embed = pca.fit_transform(embed)
        fig = plt.figure(figsize=(15*figscale, 3 * num_components*figscale))
        index = range(len(embed))
        for i in range(num_components - 1):
            colors = embed[:, i + 1]
            norm = mpl.colors.Normalize(vmin=min(colors), vmax=max(colors))
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
            colors = sm.to_rgba(colors)

            # PC vs PC on first column
            plt.subplot(num_components - 1, 2, 2 * i + 1)
            plt.scatter(embed[:, 0], embed[:, i + 1], c=colors, s=5 * figscale)
            PlottingContext.scatter_text(
                index, embed[:, 0], embed[:, i + 1], colors=colors, fontsize=8*figscale
            )
            plt.xlabel("PC 0")
            plt.ylabel(f"PC {i + 1}")

            # PC vs index
            plt.subplot(num_components - 1, 2, 2 * i + 2)
            plt.scatter(index, embed[:, i], c=colors, s=5 * figscale)
            PlottingContext.scatter_text(
                index, index, embed[:, i], colors=colors, fontsize=8*figscale
            )
            plt.xlabel("Index")
            plt.ylabel(f"PC {i}")

            plt.colorbar(sm, label=f"PC {i + 1}")

        fig.tight_layout()
        return fig


class IO:
    def namespace_to_dict(args):
        return {k: v for k, v in args.__dict__.items() if not k.startswith("_")}

    def save_args(args, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            yaml.dump(IO.namespace_to_dict(args), f, sort_keys=False)

    def load_args(path):
        with open(path, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
            args = Namespace(**args)
        return args

    def load_latest_model(path, verbose=True):
        final_model = os.path.join(path, "ckpts/model.pt")
        if os.path.exists(final_model):
            model = final_model
        else:
            path = os.path.join(path, "ckpts")
            ckpts = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")]
            ckpts.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1]) if x.split('/')[-1].split('.')[0].split('-')[-1].isdigit() else float('-inf'))
            model = ckpts[-1]
        if verbose:
            print(f"Loading model from {model}")
        return model

    def get_root():
        return "/export/d0/kitouni/NuCLR-MechInterp-results"


class Physics:
    all_funcs = ['pairing', 'volume', 'surface', 'coulomb', 'asymmetry', 'SEMF', 'shell', 'rotational', 'exchange', 'wigner', 'strutinsky', 'BW2']

    def pairing(Z, N):
        A = Z + N
        aP = 11.18
        delta = aP * A ** (-1 / 2)
        delta[(Z % 2 == 1) & (N % 2 == 1)] *= -1
        delta[A % 1 == 1] = 0
        return delta


    def volume(Z, N):
        A = Z + N
        aV = 15.75
        return aV * A


    def surface(Z, N):
        A = Z + N
        aS = 17.8
        return aS * A ** (2 / 3)


    def coulomb(Z, N):
        A = Z + N
        aC = 0.711
        return aC * Z * (Z - 1) / (A ** (1 / 3))


    def asymmetry(Z, N):
        A = Z + N
        aA = 23.7
        return aA * (N - Z) ** 2 / A


    def SEMF(Z, N):
        Eb = Physics.volume(Z, N) - Physics.surface(Z, N) - Physics.coulomb(Z, N) - Physics.asymmetry(Z, N) + Physics.pairing(Z, N)
        Eb[Eb < 0] = 0
        return Eb / (Z + N) * 1000  # keV


    def shell(Z, N):
        # calculates the shell effects according to "Mutual influence of terms in a semi-empirical" Kirson
        alpham = -1.9
        betam = 0.14
        magic = [2, 8, 20, 28, 50, 82, 126, 184]

        def find_nearest(lst, target):
            return min(lst, key=lambda x: abs(x - target))

        nup = np.array([abs(x - find_nearest(magic, x)) for x in Z])
        nun = np.array([abs(x - find_nearest(magic, x)) for x in N])
        P = np.array([nup * nun / (nup + nun) if nup + nun != 0 else 0 for nup, nun in zip(nup, nun)])
        return alpham * P + betam * P**2


    def rotational(Z, N):
        aR = 14.77
        return aR * (Z + N) ** (1 / 3)


    def exchange(Z, N):
        aX = 2.22
        return aX * Z ** (4 / 3) / (N + Z) ** (1 / 3)


    def wigner(Z, N):
        aW = -43.4
        return aW * (N - Z) ** 2 / (N + Z)


    def strutinsky(Z, N):
        aS = 55.62
        return aS * (N - Z) ** 2 / (N + Z) ** (4 / 3)


    def BW2(Z, N):
        A = N + Z

        aV = 16.58
        aS = -26.95
        aC = -0.774
        aA = -31.51
        axC = 2.22
        aW = -43.4
        ast = 55.62
        aR = 14.77

        Eb = (
            aV * A  # volume
            + aS * A ** (2 / 3) # surface
            + aC * Z**2 / (A ** (1 / 3)) # coulomb
            + aA * (N - Z) ** 2 / A # asymmetry
            + Physics.pairing(Z, N)  # pairing
            + Physics.shell(Z, N) # shell
            + aR * A ** (1 / 3) # rotational
            + axC * Z ** (4 / 3) / A ** (1 / 3) # exchange
            + aW * abs(N - Z) / A # Wigner
            + ast * (N - Z) ** 2 / A ** (4 / 3) # Strutinsky
        )

        Eb[Eb < 0] = 0
        return Eb / A * 1000  # keV


class Slurm:
    # SLURM job
    # run hostname to get the name of the machine you're on
    slurm_job = """#!/bin/zsh
#SBATCH --job-name={name}
#SBATCH --output=logs/{name}-%j.out
#SBATCH --error=logs/{name}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition={partition}

module load conda
conda activate {conda_env}
"""

    @classmethod
    def create_job(cls, cmd_base, name):
        host = subprocess.run(["hostname"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
        if "submit" in host:
            partition = "submit-gpu"
            conda_env = "sandbox"
        else:
            partition = "gpu,iaifi_gpu"
            conda_env = "torchdos"
        slurm_scripts_path = Path("logs/scripts")
        os.makedirs(slurm_scripts_path, exist_ok=True)

        job = cls.slurm_job.format(name=name, partition=partition, conda_env=conda_env) 
        job += cmd_base
        job_name = slurm_scripts_path / f"{name}.sh"
        with open(job_name, "w") as f:
            f.write(job)
        subprocess.run(["sbatch", job_name])