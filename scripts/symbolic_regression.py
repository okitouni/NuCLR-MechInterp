# %%
# Make sure to install the requirements with pip install -r requirements.txt
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from lib.model import get_model_and_optim
from lib.data import prepare_nuclear_data
from lib.utils import  PlottingContext, IO, get_rms
from sklearn.decomposition import PCA
from lib.data import semi_empirical_mass_formula, BW2_mass_formula
import seaborn as sns
sns.set_style('white')
sns.set_context('paper')

import glob
locations = glob.glob('/export/d0/kitouni/data/experiments/long-runs/*')

keys = [l.split('/')[-1].split('-')[1] for l in locations]
def cleanup(name):
    if '+' in name: return 'all'
    while name[-1].isdigit() or name[-1] == '.':
        name = name[:-1]
    return name
keys = [cleanup(k) for k in keys]
locations = {key:l for key, l in zip(keys, locations)}
task_rms_values = {}

# %%
model_name = 'all'
location = locations[model_name]
args = IO.load_args(f"{location}/args.yaml")
args.DEV = "cpu"
data = prepare_nuclear_data(args)
model = get_model_and_optim(data, args)[0]
model.load_state_dict(torch.load(IO.load_latest_model(location), map_location=args.DEV))

# %%
def get_penultimate_acts(model, X):
    # Save the last layer activations
    acts = torch.zeros(len(X), args.HIDDEN_DIM, device=args.DEV)
    # save the activations fed into the readout layer
    hook = model.readout.register_forward_pre_hook(
        lambda m, i: acts.copy_(i[0].detach())
    )
    model(X)
    hook.remove()
    return acts

X = data.X 
y = data.y.view(-1)
mask = (X[:, 2] == 0) & ~torch.isnan(y)
X, y = X[mask], y[mask]

acts = get_penultimate_acts(model, X)

# %%
pca = PCA(n_components=10)
pca.fit(acts.detach().cpu().numpy())
acts_pca = pca.transform(acts.detach().cpu().numpy())
print(pca.explained_variance_ratio_)

# %%
# visualize PCs as a function of X[0] and X[1]
for i in range(pca.n_components_):
    plt.scatter(X[:, 0], X[:, 1], c=acts_pca[:, i], s=5, marker='s')
    plt.xlabel(f"X[0]")
    plt.ylabel(f"X[1]")
    plt.show()

# %%
from pysr import PySRRegressor
from functools import partial

# max number of procs
import multiprocessing
nprocs = multiprocessing.cpu_count()

sr_factory = partial(PySRRegressor,
    niterations=2000,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "sin",
        # "exp",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
        "parity(x) = x % 2",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x, "parity": lambda x: x % 2},
    nested_constraints={
        # "cos": {"cos": 0,"sin": 0}, 
        "sin": {
            # "cos": 0,
                "sin": 0}, 
        "parity": {"parity": 0}},
    # ^ Define operator for SymPy as well
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    ncyclesperiteration=1000,
    population_size=500,
    procs=nprocs,
    parsimony=0.001,
)

# %%
# catch warnigns
import warnings
import tqdm
import pickle
import time
FORCE = False

os.makedirs(f"plots/long-runs/{model_name}/equations", exist_ok=True)
inputs = X[:, :2].detach().cpu().numpy()
# fit the symbolic regression model to the data
equations = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i, feature in enumerate(tqdm.tqdm(acts_pca.T)):
        time.sleep(2)
        fname = f"plots/long-runs/{model_name}/equations/{i}.pkl"
        if not os.path.exists(fname) or FORCE:
            sr = sr_factory()
            sr.fit(inputs, feature)
            equations.append(sr)
            pickle.dump(sr, open(fname, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            equations.append(pickle.load(open(fname, 'rb')))

        # make predictions
        lowest_loss_idx = equations[-1].equations_.loss.argmin()
        y_pred = equations[-1].predict(inputs, lowest_loss_idx)
        recomputed_loss = ((feature - y_pred)**2).mean()

        loss = equations[-1].equations_.loss[lowest_loss_idx]
        print(f"Loss: {loss} vs {recomputed_loss}, var: {np.var(feature)}")
print("Done")

# %%
def inverse_transform(tensor, data):
    min_ = torch.from_numpy(data.regression_transformer.data_min_).to(tensor.device)
    max_ = torch.from_numpy(data.regression_transformer.data_max_).to(tensor.device)
    return tensor * (max_ - min_) + min_

def get_preds_from_acts(acts):
    pred = model.readout(acts)
    pred = inverse_transform(pred, data)
    return pred[:, 0]


# %%
equations[-1].equations_

# %%
equations[1].equations_.loss.values.argsort()[0]

# %%
((equations[-1].predict(inputs, 13) - acts_pca[:, -1])**2).mean()

# %%
# combine the predictions of the symbolic regression models
# into a vector of features and use the same linear regression parameters from before
# to predict the output
sr_features = np.zeros((len(inputs), len(equations)))

for i, (sr, feat) in enumerate(zip(equations, acts_pca.T)):
    # idx = sr.equations_.score.values.argsort()[::-1][0] # select the best pareto front
    idx = sr.equations_.loss.values.argsort()[0] # select the best equation by loss
    pred = sr.predict(inputs, idx)
    print((((pred - feat)**2).mean()), feat.std()**2)
    sr_features[:, i] = pred
sr_features = torch.tensor(sr_features, device=args.DEV).float()

# %%
# We will get predictions in three different ways:
# 1. use the model readout layer
# 2. use a linear regression directly to the target
# 3. use a linear regression from the original PCA features

# 1. use the model readout layer
acts = pca.inverse_transform(sr_features.detach().cpu().numpy())
acts = torch.tensor(acts, device=args.DEV).float()
pred = get_preds_from_acts(acts)
print((pred - y).pow(2).mean().sqrt())


# %%
acts = get_penultimate_acts(model, X)
acts = acts.detach().cpu().numpy()
acts_pca = pca.transform(acts)
acts = pca.inverse_transform(acts_pca)

acts = torch.tensor(acts, device=args.DEV).float()
pred = get_preds_from_acts(acts)
print((pred - y).pow(2).mean().sqrt())


