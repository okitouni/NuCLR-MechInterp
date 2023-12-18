from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import yaml
from argparse import Namespace
import os


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
    targets = data.regression_transformer.inverse_transform(targets)
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
        rms = get_rms(preds, targets, zn, scale_by_A="binding_semf" == task_name)

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

    def plot_embedding(embed, type="PCA", num_components=2):
        if type == "PCA":
            pca = PCA(n_components=num_components)
            embed = pca.fit_transform(embed)
        fig = plt.figure(figsize=(15, 3 * num_components))
        index = range(len(embed))
        for i in range(num_components - 1):
            colors = embed[:, i + 1]
            norm = mpl.colors.Normalize(vmin=min(colors), vmax=max(colors))
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
            colors = sm.to_rgba(colors)

            # PC vs PC on first column
            plt.subplot(num_components - 1, 2, 2 * i + 1)
            plt.scatter(embed[:, 0], embed[:, i + 1], c=colors)
            PlottingContext.scatter_text(
                index, embed[:, 0], embed[:, i + 1], colors=colors, fontsize=8
            )
            plt.xlabel("PC 0")
            plt.ylabel(f"PC {i + 1}")

            # PC vs index
            plt.subplot(num_components - 1, 2, 2 * i + 2)
            plt.scatter(index, embed[:, i], c=colors)
            PlottingContext.scatter_text(
                index, index, embed[:, i], colors=colors, fontsize=8
            )
            plt.xlabel("Index")
            plt.ylabel(f"PC {i}")

            plt.colorbar(sm, label=f"PC {i + 1}")

        plt.show()
        plt.close(fig)


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
