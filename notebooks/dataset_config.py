"""
dataset_config.py  --  the ONLY place that knows about the individual datasets.

Put this file in  WOA_Project/  (next to the folders datasets/ and results/).
To switch dataset, change ACTIVE_DATASET below (one line) and re-run the notebooks.

Nothing here touches the Whale Optimization Algorithm itself: the WOA functions
(sigmoid, population_to_binary, fitness_function, binary_woa) are unchanged.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedGroupKFold
)

# ----------------------------------------------------------------------------
ACTIVE_DATASET = "tcga"        # "colon" | "wdbc" | "parkinson" | "tcga"
# ----------------------------------------------------------------------------

DATASETS = {
    "colon":     {"file": "colon_dataset.arff",  "groups": False},   # 62 x 2000, 2 classes
    "wdbc":      {"file": "wdbc.data",           "groups": False},   # 569 x 30,  2 classes
    "parkinson": {"file": "pd_speech_features.csv", "groups": True},  # 756 x 753, 2 classes, 3 rows/subject
    "tcga":      {"file": "data.csv",            "groups": False},   # 801 x 20531, 5 classes
}

# Run settings (used only by notebook 05). Same values as the original code.
WOA_SETTINGS = {
    name: {"population_size": 10, "max_iterations": 20,
           "random_seeds": [42, 100, 2026]}
    for name in DATASETS
}

CV_FOLDS = 5
RANDOM_STATE = 42


def _find(project_path, filename):
    hits = sorted((Path(project_path) / "datasets").rglob(filename))
    if not hits:
        raise FileNotFoundError(
            f"{filename} not found under {Path(project_path) / 'datasets'}")
    return hits[0]


def load_dataset(project_path, name=None):
    """Return one DataFrame: all feature columns + 'class' (+ 'group' for Parkinson)."""
    name = name or ACTIVE_DATASET
    cache = Path(project_path) / "datasets" / f"{name}_dataset.csv"
    if cache.exists():                       # written by notebook 01
        return pd.read_csv(cache)

    path = _find(project_path, DATASETS[name]["file"])
    if name == "colon":                      # ARFF -> DataFrame (as in notebook 01)
        lines = path.read_text(encoding="utf-8").splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data")
        df = pd.read_csv(path, skiprows=start + 1, header=None)
        df.columns = [f"att_{i}" for i in range(1, df.shape[1])] + ["class"]
    elif name == "wdbc":                     # no header: id, diagnosis, 30 features
        raw = pd.read_csv(path, header=None)
        df = raw.drop(columns=[0]).rename(columns={1: "class"})
        df.columns = [f"f_{c}" if c != "class" else c for c in df.columns]
        df = df[[c for c in df.columns if c != "class"] + ["class"]]
    elif name == "parkinson":                # 2 header rows, id, ..., class
        df = pd.read_csv(path, header=1).rename(columns={"id": "group"})
        df = df[[c for c in df.columns if c not in ("class", "group")] + ["group", "class"]]
    elif name == "tcga":                     # data.csv + labels.csv
        X = pd.read_csv(path, index_col=0)
        lab = pd.read_csv(_find(project_path, "labels.csv"), index_col=0)["Class"]
        df = X.copy()
        df["class"] = lab.loc[X.index].values
        df = df.reset_index(drop=True)
    else:
        raise ValueError(name)
    return df


def split_xy(df):
    """X (features only), y (class), groups (subject ids or None)."""
    has_groups = "group" in df.columns
    X = df.drop(columns=["class"] + (["group"] if has_groups else []))
    return X, df["class"], (df["group"] if has_groups else None)


def split_train_test(X, y, groups=None, test_size=0.2, random_state=RANDOM_STATE):
    """80/20 split. Identical to the original train_test_split for datasets without
    groups; subject-aware (no subject in both sets) when groups are given."""
    if groups is None:
        a, b, c, d = train_test_split(X, y, test_size=test_size,
                                      random_state=random_state, stratify=y)
        return a, b, c, d, None
    sg = StratifiedGroupKFold(n_splits=int(round(1 / test_size)), shuffle=True,
                              random_state=random_state)
    tr, te = next(sg.split(X, y, groups))
    return (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te],
            groups.iloc[tr].to_numpy())


class _GroupedCV:
    """Carries the subject ids so that cross_val_score(cv=...) keeps working unchanged."""
    def __init__(self, groups, n_splits, random_state):
        self.groups, self.n_splits, self.random_state = groups, n_splits, random_state

    def split(self, X, y=None, groups=None):
        return StratifiedGroupKFold(self.n_splits, shuffle=True,
                                    random_state=self.random_state).split(X, y, self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def make_cv(groups_train=None, n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    if groups_train is None:       # exactly the original CV object
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return _GroupedCV(groups_train, n_splits, random_state)


def drop_near_constant(X_train, X_test, min_std=1e-8):
    """Remove columns that are constant in the training data (label-free, train-only)."""
    keep = X_train.std() > min_std
    print("Constant features removed:", int((~keep).sum()))
    return X_train.loc[:, keep], X_test.loc[:, keep]


def get_results_path(project_path, name=None):
    p = Path(project_path) / "results" / (name or ACTIVE_DATASET)
    p.mkdir(parents=True, exist_ok=True)
    return p
