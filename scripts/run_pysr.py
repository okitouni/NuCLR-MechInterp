# %%
# Make sure to install the requirements with pip install -r requirements.txt
import os
os.chdir(f"{os.path.dirname(__file__)}/..")
import lib
from matplotlib import pyplot as plt
import numpy as np
from pysr import PySRRegressor
from functools import partial
import torch
import multiprocessing
from lib.model import get_model_and_optim
from lib.data import prepare_nuclear_data
from lib.utils import  PlottingContext, IO, get_rms
from sklearn.decomposition import PCA
import seaborn as sns
from lib.data import semi_empirical_mass_formula, BW2_mass_formula, semi_empirical_mass_formula_individual
# catch warnigns
import warnings
import tqdm
import pickle
import time
sns.set_style('white')
sns.set_context('paper')


import glob
locations = glob.glob('/checkpoint/nolte/nuclr/long-runs/*')


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
Z, N = X[:, :2].T
semf_parts = tuple(x * (N+Z) for x  in semi_empirical_mass_formula_individual(Z, N))

# sr_factory = partial(PySRRegressor,
#     niterations=500,  # < Increase me for better results
#     populations=100,
#     binary_operators=["+", "*", "-", "/", "^"],
#     unary_operators=[
#         # "log",
#         # "exp",
#         # "square",
#         "sqrt",
#         "sin",
#         "inv(x) = 1/x",
#         "parity(x) = x % 2",
#     ],
#     nested_constraints={
#       "sin" : {"sin" : 0},
#       # "exp" : {"exp" : 0},
#       # "log" : {"log" : 0},
#     },
#     constraints={"parity": 1},
#     extra_sympy_mappings={"inv": lambda x: 1 / x, "parity": lambda x: x % 2},
#     loss="loss(prediction, target) = (prediction - target)^2",
#     ncyclesperiteration=800,
#     maxsize=20,
# )
sr_factory = partial(
    PySRRegressor,
    niterations=2000,  # < Increase me for better results
    binary_operators=["+", "*", "^"],
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
            "sin": 0
        },
        "parity": {"parity": 0},
    },
    # ^ Define operator for SymPy as well
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    ncyclesperiteration=1000,
    population_size=500,
    procs=multiprocessing.cpu_count(),
    parsimony=0.001,
    maxsize=30,
    model_selection="accuracy",
)

# %%
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
        # print(np.var(feature))
print("Done")
