import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.preprocessing import (
    QuantileTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import torch
import argparse
from collections import namedtuple, OrderedDict


def delta(Z, N):
    A = Z + N
    aP = 11.18
    delta = aP * A ** (-1 / 2)
    delta[(Z % 2 == 1) & (N % 2 == 1)] *= -1
    delta[(Z % 2 == 0) & (N % 2 == 1)] = 0
    delta[(Z % 2 == 1) & (N % 2 == 0)] = 0
    return delta


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
    # P = nup * nun / (nup + nun)
    # P[np.isnan(P)] = 0
    return alpham * P + betam * P**2


# TODO move all the physics functions to a separate file
def BW2_mass_formula(Z, N):
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
        + delta(Z, N)  # pairing
        + shell(Z, N) # shell
        + aR * A ** (1 / 3) # rotational
        + axC * Z ** (4 / 3) / A ** (1 / 3) # exchange
        + aW * abs(N - Z) / A # Wigner
        + ast * (N - Z) ** 2 / A ** (4 / 3) # Strutinsky
    )

    Eb[Eb < 0] = 0
    return Eb / A * 1000  # keV


def semi_empirical_mass_formula(Z, N):
    A = N + Z
    aV = 15.75
    aS = 17.8
    aC = 0.711
    aA = 23.7
    Eb = (
        aV * A
        - aS * A ** (2 / 3)
        - aC * Z * (Z - 1) / (A ** (1 / 3))
        - aA * (N - Z) ** 2 / A
        + delta(Z, N)
    )
    Eb[Eb < 0] = 0
    return Eb / A * 1000  # keV


def apply_to_df_col(column):
    def wrapper(fn):
        return lambda df: df[column].astype(str).apply(fn)

    return wrapper


@apply_to_df_col(column="jp")
def get_spin_from(string):
    string = (
        string.replace("(", "")
        .replace(")", "")
        .replace("+", "")
        .replace("-", "")
        .replace("]", "")
        .replace("[", "")
        .replace("GE", "")
        .replace("HIGH J", "")
        .replace(">", "")
        .replace("<", "")
        .strip()
        .split(" ")[0]
    )
    if string == "":
        return float("nan")
    else:
        return float(eval(string))  # eval for 1/2 and such


@apply_to_df_col("jp")
def get_parity_from(string):
    # find the first + or -
    found_plus = string.find("+")
    found_minus = string.find("-")

    if found_plus == -1 and found_minus == -1:
        return float("nan")
    elif found_plus == -1:
        return 0  # -
    elif found_minus == -1:
        return 1  # +
    elif found_plus < found_minus:
        return 1  # +
    elif found_plus > found_minus:
        return 0  # -
    else:
        raise ValueError("something went wrong")


def get_half_life_from(df):
    # selection excludes unknown lifetimes and ones where lifetimes are given as bounds
    series = df.half_life_sec.copy()
    series[(df.half_life_sec == " ") | (df.operator_hl != " ")] = float("nan")
    series = series.astype(float)
    series = series.apply(np.log10)
    return series


@apply_to_df_col("qa")
def get_qa_from(string):
    # ~df.qa.isna() & (df.qa != " ")
    if string == " ":
        return float("nan")
    else:
        return float(string)


@apply_to_df_col("qbm")
def get_qbm_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("qbm_n")
def get_qbm_n_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("qec")
def get_qec_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("sn")
def get_sn_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("sp")
def get_sp_from(string):
    return float(string.replace(" ", "nan"))


def get_abundance_from(df):
    # abundance:
    # assumes that " " means 0
    return df.abundance.replace(" ", "0").astype(float)


@apply_to_df_col("half_life")
def get_stability_from(string):
    if string == "STABLE":
        return 1.0
    elif string == " ":
        return float("nan")
    else:
        return 0.0


@apply_to_df_col("isospin")
def get_isospin_from(string):
    return float(eval(string.replace(" ", "float('nan')")))


def get_binding_energy_from(df):
    binding = df.binding.replace(" ", "nan").astype(float)
    return binding


def get_radius_from(df):
    return df.radius.replace(" ", "nan").astype(float)


def get_targets(df, per_nucleon=False):
    # place all targets into targets an empty copy of df
    A = df.z + df.n
    scale = 1 if per_nucleon else A
    targets = df[["z", "n"]].copy()
    # binding energy per nucleon
    targets["binding"] = get_binding_energy_from(df) * scale
    # binding energy per nucleon minus semi empirical mass formula
    targets["binding_semf"] = targets.binding - semi_empirical_mass_formula(df.z, df.n) * scale
    targets["binding_bw2"] = targets.binding - BW2_mass_formula(df.z, df.n) * scale
    # radius in fm
    targets["radius"] = get_radius_from(df)
    # half life in log10(sec)
    targets["half_life_sec"] = get_half_life_from(df)
    # stability in {0, 1, nan}
    targets["stability"] = get_stability_from(df)
    # spin as float
    targets["spin"] = get_spin_from(df)
    # parity as {0 (-),1 (+), nan}
    targets["parity"] = get_parity_from(df)
    # isotope abundance in %
    targets["abundance"] = get_abundance_from(df)
    # qa = alpha decay energy in keV
    targets["qa"] = get_qa_from(df)
    # qbm = beta minus decay energy in keV
    targets["qbm"] = get_qbm_from(df)
    # qbm_n = beta minus + neutron emission energy in keV
    targets["qbm_n"] = get_qbm_n_from(df)
    # qec = electron capture energy in keV
    targets["qec"] = get_qec_from(df)
    # sn = neutron separation energy in keV
    targets["sn"] = get_sn_from(df)
    # sp = proton separation energy in keV
    targets["sp"] = get_sp_from(df)
    # isospin as float
    targets["isospin"] = get_isospin_from(df)
    # These are semi-empirical mass formula terms
    targets["volume"] = targets.z + targets.n  # volume
    targets["surface"] = targets.volume ** (2 / 3)  # surface
    targets["symmetry"] = ((targets.z - targets.n) ** 2) / targets.volume  # symmetry
    targets["delta"] = delta(targets.z, targets.n)  # delta
    targets["coulomb"] = (targets.z**2 - targets.z) / targets.volume ** (
        1 / 3
    )  # coulomb
    targets['control'] = np.random.Generator(np.random.PCG64(42)).normal(size=len(targets))
    return targets


def get_nuclear_data(recreate=False, ge=8):
    def lc_read_csv(url):
        req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
        )
        return pd.read_csv(urllib.request.urlopen(req))

    if not os.path.exists("data"):
        os.mkdir("data")
    if recreate or not os.path.exists("data/ground_states.csv"):
        df = lc_read_csv("fields=ground_states&nuclides=all")
        df.to_csv("data/ground_states.csv", index=False)
    else:
        # df = pd.read_csv("data/ground_states.csv")
        df2 = pd.read_csv("data/ame2020.csv").set_index(["z", "n"])
        df2 = df2[~df2.index.duplicated(keep="first")]
        df = pd.read_csv("data/ground_states.csv").set_index(["z", "n"])
        df["binding_unc"] = df2.binding_unc
        df["binding_sys"] = df2.binding_sys
        df.reset_index(inplace=True)

    # TODO removed this for mech interp model
    df = df[(df.z > ge) & (df.n > ge)]
    return df


Data = namedtuple(
    "Data",
    [
        "X",
        "y",
        "vocab_size",
        "output_map",
        "regression_transformer",
        "train_mask",
        "val_mask",
        "binding_unc",
    ],
)


def _train_test_split_exact(X, train_frac, n_embedding_inputs, seed=1):
    """
    Take exactly train_frac of the data as training data.
    """
    # TODO shuffle data when using SGD
    torch.manual_seed(seed)
    while True:
        train_mask = torch.ones(X.shape[0], dtype=torch.bool)
        train_mask[int(train_frac * X.shape[0]) :] = False
        train_mask = train_mask[torch.randperm(X.shape[0])]
        for i in range(n_embedding_inputs):
            if len(X[train_mask][:, i].unique()) != len(X[:, i].unique()):
                print("resampling train mask")
                break
        else:
            break
    test_mask = ~train_mask
    return train_mask, test_mask


def _train_test_split_sampled(X, train_frac, n_embedding_inputs, seed=1):
    """
    Samples are assigned to train by a bernoulli distribution with probability train_frac.
    """
    torch.manual_seed(seed)
    # assert that we have each X at least once in the training set
    while True:
        train_mask = torch.rand(X.shape[0]) < train_frac
        for i in range(n_embedding_inputs):
            if len(X[train_mask][:, i].unique()) != len(X[:, i].unique()):
                print("Resampling train mask")
                break
        else:
            break
    test_mask = ~train_mask
    return train_mask, test_mask


def _train_test_split(size, train_frac, seed=1):
    torch.manual_seed(seed)
    train_mask = torch.zeros(size, dtype=torch.bool)
    train_mask[: int(train_frac * size)] = True
    train_mask = train_mask[torch.randperm(size)]
    return train_mask, ~train_mask


def prepare_nuclear_data(config: argparse.Namespace, recreate: bool = False):
    """Prepare data to be used for training. Transforms data to tensors, gets tokens X,targets y,
    vocab size and output map which is a dict of {target:output_shape}. Usually output_shape is 1 for regression
    and n_classes for classification.

    Args:
        columns (list, optional): List of columns to use as targets. Defaults to None.
        recreate (bool, optional): Force re-download of data and save to csv. Defaults to False.
    returns (Data): namedtuple of X, y, vocab_size, output_map, quantile_transformer
    """
    df = get_nuclear_data(recreate=recreate, ge=config.NUCLEI_GE if hasattr(config, "NUCLEI_GE") else 8)
    targets = get_targets(df, per_nucleon=config.PER_NUCLEON == "true")

    X = torch.tensor(targets[["z", "n"]].values)
    vocab_size = (
        targets.z.max() + 1,
        targets.n.max() + 1,
        len(config.TARGETS_CLASSIFICATION) + len(config.TARGETS_REGRESSION),
    )

    # classification targets increasing integers
    for col in config.TARGETS_CLASSIFICATION:
        targets[col] = targets[col].astype("category").cat.codes
        # put nans back
        targets[col] = targets[col].replace(-1, np.nan)

    output_map = OrderedDict()
    for target in config.TARGETS_CLASSIFICATION:
        output_map[target] = targets[target].nunique()

    for target in config.TARGETS_REGRESSION:
        output_map[target] = 1

    reg_columns = list(config.TARGETS_REGRESSION)
    # feature_transformer = QuantileTransformer(
    #     output_distribution="uniform", random_state=config.SEED
    # )
    feature_transformer = MinMaxScaler()
    if len(reg_columns) > 0:
        # targets[reg_columns] = feature_transformer.fit_transform(
        #     targets[reg_columns].values
        # )
        feature_transformer.fit(
            targets[reg_columns].values
        )

    # don't consider nuclei with high uncertainty in binding energy
    # TODO removed this for mech interp model
    binding_unc = torch.tensor(df.binding_unc.values)
    if hasattr(config, "NUCLEI_HIGH_UNC") and config.NUCLEI_HIGH_UNC == "remove":
        except_binding = (df.binding_unc * (df.z + df.n) > 100).values
        targets.loc[except_binding, "binding"] = np.nan

    y = torch.tensor(targets[list(output_map.keys())].values).float()

    # Time to flatten everything
    X = torch.vstack(
        [torch.tensor([*x, task]) for x in X for task in torch.arange(len(output_map))]
    )
    y = y.flatten().view(-1, 1)
    train_mask, test_mask = _train_test_split(
        len(y), config.TRAIN_FRAC, seed=config.SEED
    )

    data =  Data(
        X.to(config.DEV),
        y.to(config.DEV),
        vocab_size,
        output_map,
        feature_transformer,
        train_mask.to(config.DEV),
        test_mask.to(config.DEV),
        binding_unc
    )

    if config.TRAIN_SET == "all":
        data = fix_mask(data, all_same=True, rate=1)
        config.TRAIN_FRAC = 1.0
    elif config.TRAIN_SET == "random-all_same":
        data = fix_mask(data, all_same=True, rate=config.TRAIN_FRAC)
    elif "extrap" in config.TRAIN_SET:
        distance = int(config.TRAIN_SET.split("_")[1])
        data = remove_edges(data, distance)
        config.TRAIN_FRAC = (data.train_mask.sum() / data.train_mask.shape[0]).item()
    return data



def fix_mask(data, all_same=True, rate=1):
    """all_same makes every task have the same mask, otherwise each task has it's own mask given by the order of the task"""
    gen = torch.Generator().manual_seed(42)
    if all_same:
        new_train_mask = torch.rand(data.train_mask.shape[0]//len(data.output_map), generator=gen) < rate
        new_train_mask = new_train_mask.repeat_interleave(len(data.output_map))
    else:
        new_train_mask = [torch.rand(data.train_mask.shape[0]//len(data.output_map), generator=gen) < rate for _ in range(len(data.output_map))]
        new_train_mask = torch.cat(new_train_mask)
    new_train_mask = new_train_mask.to(data.train_mask.device)
    data = data._replace(train_mask=new_train_mask)
    new_val_mask = ~new_train_mask
    data = data._replace(val_mask=new_val_mask)
    return data

def remove_edges(data, distance):
    """removes edges from the data that are within distance"""
    Z = data.X[:, 0]
    N = data.X[:, 1]
    new_train_mask = torch.ones_like(data.train_mask, dtype=torch.bool)
    for z in range(28, Z.max()+1):
        all_ns = N[Z == z]
        min_n, max_n = all_ns.min(), all_ns.max()
        if max_n - min_n < 2 * distance:
            continue
        new_train_mask[(Z == z) & ((N < min_n + distance) | (N > max_n - distance))] = False
    data =  data._replace(train_mask=new_train_mask)
    data = data._replace(val_mask=~new_train_mask.clone())
    return data
