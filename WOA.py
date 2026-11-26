import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, cross_val_score, cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:   
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:                        
    pass

# config file code
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

ACTIVE_DATASET = "tcga"       

DATASETS = {
    "colon":     {"file": "colon_dataset.arff",     "groups": False},  # 62 x 2000,  2 classes
    "wdbc":      {"file": "wdbc.data",              "groups": False},  # 569 x 30,   2 classes
    "parkinson": {"file": "pd_speech_features.csv", "groups": True},   # 756 x 753,  2 classes, 3 rows/subject
    "tcga":      {"file": "data.csv",               "groups": False},  # 801 x 20531, 5 classes (+ labels.csv)
}

WOA_SETTINGS = {
    name: {"population_size": int(np.random.randint(20, 31)), "max_iterations": 20,
           "random_seeds": [42, 100, 2026]}
    for name in DATASETS
}

CV_FOLDS = 5
RANDOM_STATE = 42
ALPHA = 0.9

def _find(data_dir, filename):
    """Look for `filename` inside data_dir (also in sub-folders)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Datasets folder not found: {data_dir}\n"
            f"Create it and put the dataset files inside, or pass --data-dir <folder>.")
    hits = sorted(data_dir.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"'{filename}' not found under {data_dir}")
    return hits[0]


def load_dataset(data_dir, name):
    """Return one DataFrame: all feature columns + 'class' (+ 'group' for Parkinson)."""
    path = _find(data_dir, DATASETS[name]["file"])
    print(f"Loading '{name}' from: {path}")

    if name == "colon":                      # ARFF -> DataFrame
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
        lab = pd.read_csv(_find(data_dir, "labels.csv"), index_col=0)["Class"]
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
    """80/20 split. Stratified when there are no groups; subject-aware
    (no subject in both sets) when groups are given."""
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
    if groups_train is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return _GroupedCV(groups_train, n_splits, random_state)


def drop_near_constant(X_train, X_test, min_std=1e-8):
    """Remove columns that are constant in the training data (label-free, train-only)."""
    keep = X_train.std() > min_std
    print("Constant features removed:", int((~keep).sum()))
    return X_train.loc[:, keep], X_test.loc[:, keep]


# WOA part of the notebooks
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def continuous_to_binary(position, rng):
    probabilities = sigmoid(position)
    random_values = rng.random(size=position.shape)
    binary_position = (random_values < probabilities).astype(int)
    return binary_position


def calculate_coefficients(iteration, max_iterations, rng):
    a = 2 - iteration * (2 / max_iterations)
    r1 = rng.random()
    r2 = rng.random()
    A = 2 * a * r1 - a
    C = 2 * r2
    return A, C


def update_whale_positions(
    positions,
    best_position,
    iteration,
    max_iterations,
    rng,
    lower_bound=-6,
    upper_bound=6
):
    new_positions = np.zeros_like(positions)

    a = 2 - iteration * (2 / max_iterations)

    n_whales = positions.shape[0]

    for i in range(n_whales):

        X = positions[i].copy()

        r1 = rng.random()
        r2 = rng.random()

        A = 2 * a * r1 - a
        C = 2 * r2

        p = rng.random()

        if p < 0.5:

            if abs(A) < 1:

                # Encircling the best solution
                D = np.abs(C * best_position - X)
                new_position = best_position - A * D

            else:

                # Search for a random whale
                random_index = rng.integers(0, n_whales)
                random_position = positions[random_index]

                D = np.abs(C * random_position - X)
                new_position = random_position - A * D

        else:

            # Spiral movement
            b = 1
            l = rng.uniform(-1, 1)

            D_prime = np.abs(best_position - X)

            new_position = (
                D_prime * np.exp(b * l) * np.cos(2 * np.pi * l)
                + best_position
            )

        # Keep positions within the search boundaries
        new_positions[i] = np.clip(
            new_position,
            lower_bound,
            upper_bound
        )

    return new_positions


def population_to_binary(positions, rng):
    probabilities = sigmoid(positions)

    random_values = rng.random(size=positions.shape)

    binary_population = (
        random_values < probabilities
    ).astype(int)

    return binary_population


def fitness_function(
    binary_position,
    X_train,
    y_train,
    classifier,
    cv,
    alpha=0.9
):
    binary_position = np.asarray(binary_position)
    selected_indices = np.where(binary_position == 1)[0]
    n_selected = len(selected_indices)

    if n_selected == 0:
        return 1.0

    X_selected = X_train.iloc[:, selected_indices]

    cv_scores = cross_val_score(
        classifier,
        X_selected,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    mean_f1 = cv_scores.mean()

    feature_ratio = n_selected / X_train.shape[1]

    fitness = (
        alpha * (1 - mean_f1)
        + (1 - alpha) * feature_ratio
    )

    return fitness


def binary_woa(
    X_train,
    y_train,
    classifier,
    cv,
    population_size=10,
    max_iterations=20,
    alpha=0.9,
    lower_bound=-6,
    upper_bound=6,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    n_features = X_train.shape[1]

    #Initialize continuous whale positions
    positions = rng.uniform(
        lower_bound,
        upper_bound,
        size=(population_size, n_features)
    )

    #Convert initial population to binary
    binary_population = population_to_binary(
        positions,
        rng
    )

    #Evaluate initial population
    fitness_values = np.array([
        fitness_function(
            binary_population[i],
            X_train,
            y_train,
            classifier,
            cv,
            alpha
        )
        for i in range(population_size)
    ])

    #Find initial global best
    best_index = np.argmin(fitness_values)

    best_fitness = fitness_values[best_index]
    best_position = positions[best_index].copy()
    best_binary_solution = binary_population[best_index].copy()

    convergence_curve = [best_fitness]

    #Main WOA iterations
    for iteration in range(max_iterations):

        # Update continuous positions
        positions = update_whale_positions(
            positions=positions,
            best_position=best_position,
            iteration=iteration,
            max_iterations=max_iterations,
            rng=rng,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        # Convert updated positions to binary
        binary_population = population_to_binary(
            positions,
            rng
        )

        # Evaluate new population
        fitness_values = np.array([
            fitness_function(
                binary_population[i],
                X_train,
                y_train,
                classifier,
                cv,
                alpha
            )
            for i in range(population_size)
        ])

        # Find best candidate in current iteration
        current_best_index = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        # Update global best if improvement is found
        if current_best_fitness < best_fitness:

            best_fitness = current_best_fitness

            best_position = positions[current_best_index].copy()

            best_binary_solution = (
                binary_population[current_best_index].copy()
            )

        #Store best fitness so far
        convergence_curve.append(best_fitness)

        print(
            f"Iteration {iteration + 1}/{max_iterations} "
            f"| Best Fitness: {best_fitness:.6f} "
            f"| Selected Features: {best_binary_solution.sum()}"
        )

    return (
        best_binary_solution,
        best_fitness,
        convergence_curve
    )


def make_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ])


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# data exploration from notebook1
def stage_explore(df, name, results_path):
    banner(f"[1/6] Data exploration  -  {name}")
    X, y, groups = split_xy(df)
    y = y.astype(str)

    print("Dataset shape:", df.shape)
    print("Number of samples:", df.shape[0])
    print("Number of features:", X.shape[1])
    print("\nClass counts:\n", y.value_counts())
    print("\nClass percentages:\n", (y.value_counts(normalize=True) * 100).round(2))

    missing = df.isnull().sum()
    print("\nTotal missing values:", int(missing.sum()))
    print("Columns with missing values:", int((missing > 0).sum()))
    print("Dtypes:\n", df.dtypes.value_counts())

    n_const = int((X.nunique() <= 1).sum())
    print("\nConstant features:", n_const)
    print("Mean of feature means:", X.mean().mean())
    print("Mean of feature stds :", X.std().mean())
    print("Min / Max value      :", X.min().min(), "/", X.max().max())

    # class distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values,
           color=[plt.get_cmap("Set2")(i) for i in range(len(counts))])
    ax.set_title("Class Distribution"); ax.set_xlabel("Class"); ax.set_ylabel("Number of Samples")
    fig.tight_layout(); fig.savefig(results_path / "class_distribution.png", dpi=150); plt.close(fig)

    # six sample feature histograms
    cols = [X.columns[i] for i in np.linspace(0, X.shape[1] - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.flatten(), cols):
        ax.hist(X[c].dropna(), bins=30)
        ax.set_title(f"Distribution of {str(c)[:30]}")
    fig.tight_layout(); fig.savefig(results_path / "feature_histograms.png", dpi=120); plt.close(fig)
    print("Saved: class_distribution.png, feature_histograms.png")


# preprocessing from notebook 2
def prepare_data(df):
    banner("[2/6] Preprocessing")
    X, y, groups = split_xy(df)
    print("Features shape:", X.shape, "| Target shape:", y.shape)
    print("Missing values in features:", int(X.isnull().sum().sum()),
          "| in target:", int(y.isnull().sum()))

    X_train, X_test, y_train, y_test, groups_train = split_train_test(X, y, groups)
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Train class distribution:\n", y_train.value_counts().to_string())
    print("Test class distribution:\n", y_test.value_counts().to_string())

    # duplicate features (decided on training data only)
    duplicate_columns = X_train.columns[X_train.T.duplicated()].tolist()
    print("Duplicate features removed:", len(duplicate_columns))
    X_train = X_train.drop(columns=duplicate_columns)
    X_test = X_test.drop(columns=duplicate_columns)

    # near-constant features (training data only)
    X_train, X_test = drop_near_constant(X_train, X_test)
    print("Shape after cleaning -> train:", X_train.shape, "| test:", X_test.shape)
    print("(StandardScaler is applied inside each model pipeline, fitted on training folds only.)")

    return dict(X=X, y=y, groups=groups,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                groups_train=groups_train, duplicate_columns=duplicate_columns)


# baseline models code from notebook 3
def stage_baselines(d, results_path):
    banner("[3/6] Baseline models (all features)")
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", class_weight="balanced"))
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", class_weight="balanced"))
        ]),
    }
    cv = make_cv(d["groups_train"])
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}

    cv_results = {}
    for name, model in models.items():
        s = cross_validate(model, d["X_train"], d["y_train"], cv=cv, scoring=scoring,
                           return_train_score=False)
        cv_results[name] = {"Accuracy": s["test_accuracy"].mean(),
                            "Balanced Accuracy": s["test_balanced_accuracy"].mean(),
                            "F1 Macro": s["test_f1_macro"].mean()}
    results_df = pd.DataFrame(cv_results).T
    print("\nCross-validation (train):\n", results_df.round(4))
    best = results_df["F1 Macro"].idxmax()
    print("\nBest baseline by CV F1 Macro:", best)

    test_results, fitted = {}, {}
    for name, model in models.items():
        model.fit(d["X_train"], d["y_train"])
        yp = model.predict(d["X_test"])
        fitted[name] = model
        test_results[name] = {
            "Accuracy": accuracy_score(d["y_test"], yp),
            "Balanced Accuracy": balanced_accuracy_score(d["y_test"], yp),
            "Precision Macro": precision_score(d["y_test"], yp, average="macro", zero_division=0),
            "Recall Macro": recall_score(d["y_test"], yp, average="macro", zero_division=0),
            "F1 Macro": f1_score(d["y_test"], yp, average="macro", zero_division=0),
        }
    test_df = pd.DataFrame(test_results).T
    print("\nTest set:\n", test_df.round(4))

    yp_best = fitted[best].predict(d["X_test"])
    print(f"\nClassification report ({best}):\n", classification_report(d["y_test"], yp_best))
    print("Confusion matrix:\n", confusion_matrix(d["y_test"], yp_best))

    results_df.to_csv(results_path / "baseline_cv_results.csv")
    test_df.to_csv(results_path / "baseline_test_results.csv")
    print("Saved: baseline_cv_results.csv, baseline_test_results.csv")


# WOA feature selection
def stage_woa(d, name, results_path, population_size, max_iterations, seeds):
    banner("[4/6] WOA feature selection")
    X_train, y_train = d["X_train"], d["y_train"]
    classifier = make_classifier()
    cv = make_cv(d["groups_train"])
    print(f"Features: {X_train.shape[1]} | population: {population_size} | "
          f"iterations: {max_iterations} | seeds: {seeds} | alpha: {ALPHA}")

    all_runs = []
    for seed in seeds:
        print(f"\nStarting WOA run with seed {seed}")
        t0 = time.time()
        best_solution, best_fitness, curve = binary_woa(
            X_train=X_train, y_train=y_train, classifier=classifier, cv=cv,
            population_size=population_size, max_iterations=max_iterations,
            alpha=ALPHA, lower_bound=-6, upper_bound=6, random_state=seed)
        print(f"Seed {seed} finished in {time.time() - t0:.1f}s")
        all_runs.append({"seed": seed, "best_solution": best_solution.copy(),
                         "best_fitness": best_fitness,
                         "selected_features": int(best_solution.sum()),
                         "convergence_curve": list(curve)})

    runs_summary = pd.DataFrame([{"Seed": r["seed"], "Best Fitness": r["best_fitness"],
                                  "Selected Features": r["selected_features"]} for r in all_runs])
    print("\nRuns summary:\n", runs_summary)

    best_run = all_runs[int(np.argmin([r["best_fitness"] for r in all_runs]))]
    final_solution = best_run["best_solution"].copy()
    selected = X_train.columns[final_solution == 1].tolist()
    print(f"\nBest seed: {best_run['seed']} | fitness {best_run['best_fitness']:.6f} "
          f"| selected {len(selected)} features")
    assert final_solution.shape == (X_train.shape[1],) and final_solution.sum() > 0

    #convergence plot + csv files
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in all_runs:
        ax.plot(range(len(r["convergence_curve"])), r["convergence_curve"], marker="o",
                label=f"seed {r['seed']}", lw=2.2 if r is best_run else 1, alpha=1 if r is best_run else .6)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Fitness")
    ax.set_title("Binary WOA Convergence Curve"); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(results_path / "woa_convergence.png", dpi=150); plt.close(fig)

    pd.DataFrame({"Iteration": range(len(best_run["convergence_curve"])),
                  "Best_Fitness": best_run["convergence_curve"]}
                 ).to_csv(results_path / "woa_convergence.csv", index=False)
    pd.DataFrame({"Feature": selected}).to_csv(results_path / "final_woa_selected_features.csv", index=False)
    runs_summary.to_csv(results_path / "woa_multiple_runs_summary.csv", index=False)
    print("Saved: woa_convergence.csv/.png, final_woa_selected_features.csv, woa_multiple_runs_summary.csv")
    return selected


# notebook 6 evaluaion
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def stage_final_evaluation(d, selected_features, results_path):
    banner("[5/6] Final evaluation: all features vs WOA-selected features")
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["y_train"], d["y_test"]
    class_labels = sorted(y_train.unique())

    assert len(selected_features) > 0
    assert len(set(selected_features)) == len(selected_features)
    assert set(selected_features).issubset(set(X_train.columns))

    base_model, woa_model = make_classifier(), make_classifier()
    base_model.fit(X_train, y_train)
    woa_model.fit(X_train[selected_features], y_train)
    y_pred_base = base_model.predict(X_test)
    y_pred_woa = woa_model.predict(X_test[selected_features])

    comparison_df = pd.DataFrame(
        [evaluate_model(y_test, y_pred_base), evaluate_model(y_test, y_pred_woa)],
        index=[f"Baseline ({X_train.shape[1]} Features)", f"WOA ({len(selected_features)} Features)"])
    print(comparison_df.round(4))

    for title, yp in (("Baseline", y_pred_base), ("WOA", y_pred_woa)):
        print(f"\n{title} classification report")
        print(classification_report(y_test, yp, labels=class_labels, zero_division=0))

    n0, n1 = X_train.shape[1], len(selected_features)
    print(f"Original features: {n0} | selected: {n1} | reduction: {(n0 - n1) / n0 * 100:.2f}%")

    comparison_df.to_csv(results_path / "final_model_comparison.csv")
    base_report = classification_report(y_test, y_pred_base, labels=class_labels,
                                        output_dict=True, zero_division=0)
    woa_report = classification_report(y_test, y_pred_woa, labels=class_labels,
                                       output_dict=True, zero_division=0)
    pd.DataFrame(base_report).to_csv(results_path / "baseline_classification_report.csv")
    pd.DataFrame(woa_report).to_csv(results_path / "woa_classification_report.csv")

    #small comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    comparison_df["F1 Macro"].plot(kind="bar", ax=axes[0], title="F1 Macro Comparison", ylim=(0, 1))
    axes[0].set_ylabel("F1 Macro"); axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(["Baseline", "WOA"], [n0, n1])
    axes[1].set_title("Number of Features"); axes[1].set_ylabel("Number of Features")
    fig.tight_layout(); fig.savefig(results_path / "final_comparison.png", dpi=150); plt.close(fig)
    print("Saved: final_model_comparison.csv, *_classification_report.csv, final_comparison.png")

    return dict(comparison_df=comparison_df, class_labels=class_labels,
                baseline_report=base_report, woa_report=woa_report,
                y_pred_baseline=y_pred_base, y_pred_woa=y_pred_woa,
                selected_features=selected_features)


# notebook 6 report pdf stuff
def build_pdf_report(name, results_path, d, ev, settings):
    banner("[6/6] PDF report")
    RESULTS_PATH, ACTIVE = results_path, name
    X, y, groups = d["X"], d["y"], d["groups"]
    X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    duplicate_columns = d["duplicate_columns"]
    comparison_df, class_labels = ev["comparison_df"], ev["class_labels"]
    baseline_report, woa_report = ev["baseline_report"], ev["woa_report"]
    y_pred_baseline, y_pred_woa = ev["y_pred_baseline"], ev["y_pred_woa"]
    selected_features = ev["selected_features"]

    PDF_PATH = RESULTS_PATH / f"{ACTIVE}_full_report.pdf"
    PAGE = (11.69, 8.27)                      # A4 landscape
    C_BASE, C_WOA = "#4C78A8", "#F58518"      # baseline = blue, WOA = orange
    TAB10 = plt.get_cmap("tab10")
    page_no = [0]

    #helpers
    def fmt(df):
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                v = out[c].dropna()
                whole = len(v) > 0 and (v == v.round()).all()
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else (f"{x:.0f}" if whole else f"{x:.4f}"))
        return out.astype(str)

    def new_page(title):
        fig = plt.figure(figsize=PAGE)
        fig.text(0.04, 0.965, title, fontsize=17, fontweight="bold", va="top")
        fig.text(0.96, 0.965, f"Dataset: {ACTIVE}", fontsize=10, color="#666666", va="top", ha="right")
        fig.add_artist(plt.Line2D([0.04, 0.96], [0.915, 0.915], color="#cccccc", lw=0.8))
        page_no[0] += 1
        fig.text(0.96, 0.02, f"Page {page_no[0]}", ha="right", fontsize=8, color="#888888")
        return fig

    def block(fig, rect, title):
        ax = fig.add_axes(rect)
        fig.text(rect[0], rect[1] + rect[3] + 0.012, title, fontsize=11, fontweight="bold", va="bottom")
        return ax

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)

    def draw_table(ax, df, fontsize=9, scale=1.45, index=True):
        ax.axis("off")
        dd = fmt(df)
        if index:
            dd.insert(0, "", [str(i) for i in df.index])
        cols = [str(c) for c in dd.columns]
        lens = np.array([max([len(cols[j])] + [len(str(v)) for v in dd.iloc[:, j]])
                         for j in range(len(cols))], float) + 3
        widths = (lens / lens.sum()).tolist()
        ax_h_in = ax.get_position().height * PAGE[1]
        h = min(1.0, (len(dd) + 1) * 0.30 * scale / ax_h_in / 1.45)
        tab = ax.table(cellText=dd.values, colLabels=cols, colWidths=widths, cellLoc="center",
                       bbox=[0, 1 - h, 1, h])
        tab.auto_set_font_size(False); tab.set_fontsize(fontsize)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#BBBBBB")
            if r == 0:
                cell.set_facecolor("#E8EEF4"); cell.set_text_props(weight="bold")
            elif index and c == 0:
                cell.set_facecolor("#F6F6F6"); cell.set_text_props(weight="bold")

    def not_found(ax, nm):
        ax.axis("off")
        ax.text(0, 0.9, f"{nm} not found", color="#b00020", fontsize=9, va="top")

    def grouped_bars(ax, df, cols, colors=None, ylim=(0, 1.25), labels=True):
        n = len(cols); w = 0.8 / n; x = np.arange(len(df.index))
        for j, c in enumerate(cols):
            bars = ax.bar(x + (j - (n - 1) / 2) * w, df[c].values, w, label=str(c),
                          color=(colors[j] if colors else TAB10(j)))
            if labels:
                for b in bars:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                            ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels([str(i) for i in df.index], fontsize=8)
        ax.set_ylim(*ylim); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(fontsize=8, frameon=False, ncol=min(n, 3), loc="upper center"); style(ax)

    def read_result(fname, index_col=None):
        p = RESULTS_PATH / fname
        return pd.read_csv(p, index_col=index_col) if p.exists() else None

    def heatmap(ax, cm, title, fmt_str="d", cmap="Blues"):
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1e-9))
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt_str), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    #numbers
    class_names = [str(c) for c in class_labels]
    n_samples, n_feat_raw = X.shape
    n_train, n_test = len(X_train), len(X_test)
    n_dup = len(duplicate_columns)
    n_feat_clean = X_train.shape[1]
    n_const = n_feat_raw - n_dup - n_feat_clean
    n_sel = len(selected_features)
    reduction = (n_feat_clean - n_sel) / n_feat_clean * 100
    counts_all = y.value_counts().reindex(class_labels).fillna(0).astype(int)
    counts_tr = y_train.value_counts().reindex(class_labels).fillna(0).astype(int)
    counts_te = y_test.value_counts().reindex(class_labels).fillna(0).astype(int)

    Z = StandardScaler().fit(X_train).transform(X_train)

    with PdfPages(PDF_PATH) as pdf:

        #cover
        fig = plt.figure(figsize=PAGE); page_no[0] += 1
        fig.text(0.06, 0.88, "WOA Feature Selection", fontsize=30, fontweight="bold", va="top")
        fig.text(0.06, 0.79, "Full Results Report", fontsize=20, va="top", color="#444444")
        fig.text(0.06, 0.71, f"Dataset: {ACTIVE}", fontsize=14, va="top")
        fig.text(0.06, 0.67, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", fontsize=10, va="top", color="#666666")
        ax = block(fig, [0.06, 0.34, 0.50, 0.20], "Key Results (Test Set)")
        key = comparison_df[["Accuracy", "Balanced Accuracy", "F1 Macro"]].copy()
        key.insert(0, "Features", [n_feat_clean, n_sel])
        draw_table(ax, key, fontsize=10, scale=1.7)
        fig.text(0.66, 0.60, "Contents", fontsize=13, fontweight="bold", va="top")
        for i, t in enumerate(["1. Dataset Overview", "2. Feature Statistics", "3. Preprocessing",
                               "4. Baseline Models", "5. WOA Optimisation", "6. Final Comparison",
                               "7. Per-Class Results", "8. Confusion Matrices", "9. Selected Features"]):
            fig.text(0.66, 0.56 - i * 0.045, t, fontsize=11, va="top")
        pdf.savefig(fig); plt.close(fig)

        #dataset overview
        fig = new_page("1. Dataset Overview")
        ax = block(fig, [0.04, 0.12, 0.40, 0.70], "Dataset Dimensions")
        dims = [("Samples", n_samples), ("Features (raw)", n_feat_raw), ("Classes", len(class_labels)),
                ("Class labels", ", ".join(class_names)),
                ("Subjects (groups)", int(groups.nunique()) if groups is not None else "-"),
                ("Missing values (features)", int(X.isnull().sum().sum())),
                ("Missing values (target)", int(y.isnull().sum())),
                ("Train samples", n_train), ("Test samples", n_test),
                ("Duplicate features removed", n_dup), ("Constant features removed", n_const),
                ("Features after cleaning", n_feat_clean), ("Features selected by WOA", n_sel)]
        draw_table(ax, pd.DataFrame(dims, columns=["Item", "Value"]).set_index("Item"), fontsize=10, scale=1.6)
        ax = block(fig, [0.55, 0.50, 0.41, 0.32], "Class Distribution (All Samples)")
        bars = ax.bar(class_names, counts_all.values, color=[TAB10(i) for i in range(len(class_names))])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Samples"); style(ax)
        ax = block(fig, [0.55, 0.12, 0.41, 0.28], "Class Counts")
        ct = pd.DataFrame({"All": counts_all, "Percent": [f"{v:.2f}" for v in counts_all / counts_all.sum() * 100],
                           "Train": counts_tr, "Test": counts_te})
        ct.index = class_names
        draw_table(ax, ct, fontsize=9, scale=1.4)
        pdf.savefig(fig); plt.close(fig)

        #feature statistics
        fm, fs, fmin, fmax = X.mean(), X.std(), X.min(), X.max()
        fig = new_page("2. Feature Statistics")
        ax = block(fig, [0.04, 0.60, 0.55, 0.24], "Per-Feature Statistics (summary across all features)")
        summ = pd.DataFrame({"Min": [s.quantile(0) for s in (fm, fs, fmin, fmax)],
                             "25%": [s.quantile(.25) for s in (fm, fs, fmin, fmax)],
                             "Median": [s.quantile(.5) for s in (fm, fs, fmin, fmax)],
                             "75%": [s.quantile(.75) for s in (fm, fs, fmin, fmax)],
                             "Max": [s.quantile(1) for s in (fm, fs, fmin, fmax)]},
                            index=["Feature mean", "Feature std", "Feature min", "Feature max"])
        summ = summ.map(lambda v: f"{v:.4g}") if hasattr(summ, "map") else summ.applymap(lambda v: f"{v:.4g}")
        draw_table(ax, summ, fontsize=9, scale=1.5)
        ax = block(fig, [0.66, 0.60, 0.30, 0.24], "Overall Scale")
        overall = pd.DataFrame({"Value": [f"{fm.mean():.4g}", f"{fs.mean():.4g}",
                                          f"{fmin.min():.4g}", f"{fmax.max():.4g}"]},
                               index=["Mean of feature means", "Mean of feature std",
                                      "Global minimum", "Global maximum"])
        draw_table(ax, overall, fontsize=9, scale=1.5)
        ax = block(fig, [0.06, 0.10, 0.40, 0.38], "Distribution of Feature Means")
        ax.hist(np.sign(fm) * np.log10(1 + fm.abs()), bins=40, color=C_BASE)
        ax.set_xlabel("sign(mean) x log10(1 + |mean|)"); ax.set_ylabel("Features"); style(ax)
        ax = block(fig, [0.55, 0.10, 0.40, 0.38], "Distribution of Feature Std (log10)")
        pos = fs[fs > 0]
        ax.hist(np.log10(pos), bins=40, color=C_WOA)
        ax.set_xlabel("log10(feature std)"); ax.set_ylabel("Features"); style(ax)
        pdf.savefig(fig); plt.close(fig)

        #sample feature histograms
        fig = new_page("2. Feature Statistics - Sample Features")
        idx = np.linspace(0, X.shape[1] - 1, 6).astype(int)
        for k, j in enumerate(idx):
            r, c = divmod(k, 3)
            ax = block(fig, [0.06 + c * 0.31, 0.50 - r * 0.42, 0.25, 0.30], f"Feature: {str(X.columns[j])[:28]}")
            ax.hist(X.iloc[:, j].dropna(), bins=30, color=C_BASE); ax.set_ylabel("Samples"); style(ax)
        pdf.savefig(fig); plt.close(fig)

        #PCA projection
        n_comp = min(10, Z.shape[0] - 1, Z.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
        P = pca.fit_transform(Z)
        fig = new_page("2. Feature Statistics - PCA of Training Samples")
        ax = block(fig, [0.06, 0.12, 0.50, 0.70], "PCA Projection (first 2 components)")
        ytr = y_train.to_numpy()
        for i, cl in enumerate(class_labels):
            m = ytr == cl
            ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.75, color=TAB10(i), label=f"Class {cl}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.legend(fontsize=8, frameon=False); style(ax)
        ax = block(fig, [0.65, 0.12, 0.30, 0.70], "Explained Variance Ratio")
        ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100, color=C_BASE)
        ax.set_xlabel("Component"); ax.set_ylabel("Explained variance (%)"); style(ax)
        pdf.savefig(fig); plt.close(fig)

        #preprocessing
        fig = new_page("3. Preprocessing")
        ax = block(fig, [0.04, 0.50, 0.42, 0.32], "Train / Test Split")
        split = pd.DataFrame({"Samples": [n_train, n_test],
                              "Share (%)": [f"{n_train / n_samples * 100:.1f}", f"{n_test / n_samples * 100:.1f}"]},
                             index=["Train", "Test"])
        draw_table(ax, split, fontsize=10, scale=1.6)
        ax = block(fig, [0.04, 0.10, 0.42, 0.28], "Standardised Training Data (check)")
        chk = pd.DataFrame({"Value": [f"{Z.mean(axis=0).mean():.4g}", f"{Z.std(axis=0).mean():.4g}",
                                      f"{Z.min():.4g}", f"{Z.max():.4g}"]},
                           index=["Mean of feature means", "Mean of feature std", "Minimum", "Maximum"])
        draw_table(ax, chk, fontsize=10, scale=1.5)
        ax = block(fig, [0.60, 0.15, 0.36, 0.62], "Number of Features at Each Stage")
        stages = ["Raw", "After duplicate\nremoval", "After constant\nremoval", "Selected\nby WOA"]
        vals = [n_feat_raw, n_feat_raw - n_dup, n_feat_clean, n_sel]
        bars = ax.barh(stages[::-1], vals[::-1], color=[C_WOA, C_BASE, C_BASE, C_BASE])
        for b in bars:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {int(b.get_width())}",
                    va="center", fontsize=9)
        ax.set_xlim(0, max(vals) * 1.15); ax.set_xlabel("Features"); style(ax)
        pdf.savefig(fig); plt.close(fig)

        #baseline models
        cv_df = read_result("baseline_cv_results.csv", 0)
        te_df = read_result("baseline_test_results.csv", 0)
        fig = new_page("4. Baseline Models (All Features)")
        ax = block(fig, [0.04, 0.60, 0.34, 0.22], "Cross-Validation Results (Train)")
        draw_table(ax, cv_df, fontsize=9, scale=1.5) if cv_df is not None else not_found(ax, "baseline_cv_results.csv")
        ax = block(fig, [0.42, 0.60, 0.54, 0.22], "Test Set Results")
        draw_table(ax, te_df, fontsize=9, scale=1.5) if te_df is not None else not_found(ax, "baseline_test_results.csv")
        ax = block(fig, [0.06, 0.10, 0.40, 0.38], "Cross-Validation Scores")
        if cv_df is not None:
            grouped_bars(ax, cv_df, list(cv_df.columns))
        ax = block(fig, [0.55, 0.10, 0.40, 0.38], "Test Scores")
        if te_df is not None:
            grouped_bars(ax, te_df, ["Accuracy", "Balanced Accuracy", "F1 Macro"])
        pdf.savefig(fig); plt.close(fig)

        #WOA optimisation
        conv = read_result("woa_convergence.csv")
        runs = read_result("woa_multiple_runs_summary.csv")
        fig = new_page("5. WOA Optimisation")
        ax = block(fig, [0.06, 0.52, 0.50, 0.30], "Convergence Curve")
        if conv is not None:
            ax.plot(conv["Iteration"], conv["Best_Fitness"], marker="o", color=C_WOA)
            ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness"); style(ax)
        else:
            not_found(ax, "woa_convergence.csv")
        ax = block(fig, [0.66, 0.52, 0.30, 0.30], "WOA Parameters")
        par = pd.DataFrame({"Value": [settings["population_size"], settings["max_iterations"],
                                      ", ".join(map(str, settings["random_seeds"])), ALPHA, "-6 to 6",
                                      "Logistic Regression", f"{CV_FOLDS}-fold"]},
                           index=["Population size", "Iterations", "Seeds", "Alpha", "Position bounds",
                                  "Evaluator", "CV"])
        draw_table(ax, par, fontsize=9, scale=1.4)
        ax = block(fig, [0.06, 0.08, 0.40, 0.28], "WOA Runs by Seed")
        draw_table(ax, runs.set_index(runs.columns[0]), fontsize=9, scale=1.5) if runs is not None \
            else not_found(ax, "woa_multiple_runs_summary.csv")
        ax = block(fig, [0.60, 0.08, 0.36, 0.28], "Selected Features per Seed")
        if runs is not None:
            best_i = int(runs["Best Fitness"].idxmin())
            cols = [C_WOA if i == best_i else C_BASE for i in runs.index]
            bars = ax.bar(runs["Seed"].astype(str), runs["Selected Features"], color=cols)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_xlabel("Seed (best run in orange)"); ax.set_ylabel("Features"); style(ax)
        pdf.savefig(fig); plt.close(fig)

        #final comparison
        fig = new_page("6. Final Comparison: All Features vs WOA")
        ax = block(fig, [0.04, 0.62, 0.92, 0.20], "Test Set Metrics")
        draw_table(ax, comparison_df, fontsize=10, scale=1.7)
        ax = block(fig, [0.06, 0.10, 0.55, 0.40], "Metric Comparison")
        cmp_t = comparison_df.T
        cmp_t.columns = ["Baseline (all features)", "WOA (selected features)"]
        grouped_bars(ax, cmp_t, list(cmp_t.columns), colors=[C_BASE, C_WOA])
        ax = block(fig, [0.72, 0.10, 0.24, 0.40], "Number of Features")
        bars = ax.bar(["Baseline", "WOA"], [n_feat_clean, n_sel], color=[C_BASE, C_WOA])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(f"Reduction: {reduction:.2f}%", fontsize=9); style(ax)
        pdf.savefig(fig); plt.close(fig)

        #per-class results
        rb = pd.DataFrame(baseline_report).T.astype(float)
        rw = pd.DataFrame(woa_report).T.astype(float)
        for _r in (rb, rw):
            if "accuracy" in _r.index:
                _r.loc["accuracy", "support"] = _r.loc["macro avg", "support"]
        fig = new_page("7. Per-Class Results")
        ax = block(fig, [0.04, 0.52, 0.44, 0.32], "Classification Report - Baseline")
        draw_table(ax, rb, fontsize=8, scale=1.3)
        ax = block(fig, [0.52, 0.52, 0.44, 0.32], "Classification Report - WOA")
        draw_table(ax, rw, fontsize=8, scale=1.3)
        ax = block(fig, [0.06, 0.08, 0.88, 0.30], "F1 Score per Class")
        f1 = pd.DataFrame({"Baseline (all features)": [rb.loc[c, "f1-score"] for c in class_names],
                           "WOA (selected features)": [rw.loc[c, "f1-score"] for c in class_names]},
                          index=[f"Class {c}" for c in class_names])
        grouped_bars(ax, f1, list(f1.columns), colors=[C_BASE, C_WOA])
        pdf.savefig(fig); plt.close(fig)

        #confusion matrices
        cm_b = confusion_matrix(y_test, y_pred_baseline, labels=class_labels)
        cm_w = confusion_matrix(y_test, y_pred_woa, labels=class_labels)
        nrm = lambda m: m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        fig = new_page("8. Confusion Matrices")
        heatmap(block(fig, [0.07, 0.50, 0.36, 0.34], "Baseline (counts)"), cm_b, "")
        heatmap(block(fig, [0.55, 0.50, 0.36, 0.34], "WOA (counts)"), cm_w, "")
        heatmap(block(fig, [0.07, 0.07, 0.36, 0.34], "Baseline (row-normalised)"), nrm(cm_b), "", ".2f")
        heatmap(block(fig, [0.55, 0.07, 0.36, 0.34], "WOA (row-normalised)"), nrm(cm_w), "", ".2f")
        pdf.savefig(fig); plt.close(fig)

        #selected features
        per_col, n_cols = 52, 4
        per_page = per_col * n_cols
        for start in range(0, max(n_sel, 1), per_page):
            chunk = selected_features[start:start + per_page]
            fig = new_page(f"9. Selected Features ({n_sel} total, {start + 1}-{start + len(chunk)})")
            for c in range(n_cols):
                part = [str(s)[:30] for s in chunk[c * per_col:(c + 1) * per_col]]
                fig.text(0.05 + c * 0.235, 0.89, "\n".join(part), fontsize=7.5, va="top", family="monospace")
            pdf.savefig(fig); plt.close(fig)

    print("Full report saved at:", PDF_PATH)


# pipeline and commandline stuff i guess
def run_pipeline(name, data_dir, results_dir, population_size, max_iterations, seeds, make_report):
    t0 = time.time()
    results_path = Path(results_dir) / name
    results_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_dir, name)
    stage_explore(df, name, results_path)
    d = prepare_data(df)
    stage_baselines(d, results_path)
    selected = stage_woa(d, name, results_path, population_size, max_iterations, seeds)
    ev = stage_final_evaluation(d, selected, results_path)
    if make_report:
        settings = dict(population_size=population_size, max_iterations=max_iterations, random_seeds=seeds)
        build_pdf_report(name, results_path, d, ev, settings)
    print(f"\nDone '{name}' in {(time.time() - t0) / 60:.1f} min. Results: {results_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="WOA feature selection on local medical datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default=ACTIVE_DATASET, choices=list(DATASETS) + ["all"],
                   help="which dataset to run ('all' = every dataset found in --data-dir)")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="folder that contains the dataset files (sub-folders are searched too)")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="folder where results/<dataset>/ will be written")
    p.add_argument("--population", type=int, default=None, help="WOA population size (default: 10)")
    p.add_argument("--iterations", type=int, default=None, help="WOA iterations (default: 20)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="WOA random seeds (default: 42 100 2026)")
    p.add_argument("--no-report", action="store_true", help="skip the PDF report")
    return p.parse_args()


def main():
    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failed = []
    for name in names:
        s = WOA_SETTINGS[name]
        try:
            run_pipeline(
                name, args.data_dir, args.results_dir,
                args.population or s["population_size"],
                args.iterations or s["max_iterations"],
                args.seeds or s["random_seeds"],
                make_report=not args.no_report)
        except FileNotFoundError as e:
            print(f"\n[!] {name}: {e}")
            failed.append(name)
            if args.dataset != "all":
                sys.exit(1)
    if failed:
        print("\nSkipped (files not found):", ", ".join(failed))


if __name__ == "__main__":
    main()
