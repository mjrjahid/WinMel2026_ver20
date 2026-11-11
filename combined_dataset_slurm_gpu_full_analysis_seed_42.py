#!/usr/bin/env python3
"""WinMal2026 evidence-audit experiment suite, revision 2026-11-08 v20.

Primary target: originating report-subject family CONTEXT. Extracted files and
process images are observations, not independently verified family labels.
This script changes the evaluation; old manuscript scores must not be reused.

Run --help and read WinMal2026_Experiment_Guide.txt before the full SLURM run.
Default: three seeds, seven available original classifiers plus a Dummy baseline,
component-grouped CV, fold-fitted preprocessing, no SMOTE, no test-fitted ensemble.
No class is selected by feature purity, separability, or a target accuracy.
Split diagnostics, ablations, collision sensitivity, and blinded review templates
are included. Temporal/external/label audits require real additional evidence.
All inputs remain read-only. No local malware execution or service requests occur.
"""
import argparse
import builtins
import inspect
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from ipaddress import ip_address


def _positive_int_env(name: str, fallback: int) -> int:
    """Read a positive integer environment variable without breaking imports."""
    try:
        return max(1, int(os.environ.get(name, fallback)))
    except (TypeError, ValueError):
        return max(1, int(fallback))


# Configure thread pools before importing NumPy/scikit-learn. SLURM normally
# exports SLURM_CPUS_PER_TASK; respecting it prevents oversubscribing a node.
CPU_THREADS = _positive_int_env("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, str(CPU_THREADS))
os.environ.setdefault("MPLBACKEND", "Agg")
_temp_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp"
_cache_user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_temp_root, f"matplotlib-cache-{_cache_user}"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.multiclass import type_of_target


class _CompatibleWordNgrams:
    """Equivalent word n-grams without sklearn's failing local call aliases.

    Tokenization, preprocessing, stop-word selection, vocabulary fitting,
    counts, and TF-IDF weighting still use the inherited sklearn routines.
    Only word n-gram assembly is replaced; char/char_wb analyzers are untouched.
    Constructors remain inherited, preserving clone and joblib compatibility.
    """

    def _word_ngrams(self, tokens, stop_words=None):
        filtered = [
            token for token in tokens
            if stop_words is None or token not in stop_words
        ]
        minimum, maximum = self.ngram_range
        unigrams = [token for token in filtered] if minimum == 1 else []
        higher = [
            builtins.str.join(' ', filtered[start:start + length])
            for length in range(max(2, minimum), min(maximum, len(filtered)) + 1)
            for start in range(len(filtered) - length + 1)
        ]
        return unigrams + higher


class CompatibleCountVectorizer(_CompatibleWordNgrams, CountVectorizer):
    """CountVectorizer with equivalent, locally implemented word n-grams."""


class CompatibleTfidfVectorizer(_CompatibleWordNgrams, TfidfVectorizer):
    """TfidfVectorizer with equivalent, locally implemented word n-grams."""


def validate_text_vectorizers() -> dict:
    """Check installed and compatible routines before expensive experiments."""
    documents = ['alpha beta gamma', 'alpha delta']
    expected_features = ['alpha', 'alpha beta', 'alpha delta', 'beta', 'beta gamma', 'delta', 'gamma']
    report = {
        'sklearn_version': __import__('sklearn').__version__,
        'selected_backend': 'compatible word-ngram subclasses',
        'checks': {},
    }
    vectorizers = [
        ('installed_count', CountVectorizer),
        ('installed_tfidf', TfidfVectorizer),
        ('compatible_count', CompatibleCountVectorizer),
        ('compatible_tfidf', CompatibleTfidfVectorizer),
    ]
    for name, vectorizer_class in vectorizers:
        try:
            vectorizer = vectorizer_class(ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(documents)
            features = vectorizer.get_feature_names_out().tolist()
            if matrix.shape != (2, 7) or features != expected_features:
                raise ValueError(f'Unexpected vocabulary or matrix shape: {features}, {matrix.shape}')
            if not np.isfinite(matrix.data).all():
                raise ValueError('Non-finite vectorizer values.')
            report['checks'][name] = {'passed': True}
        except Exception as exc:
            report['checks'][name] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
            if name.startswith('compatible_'):
                raise RuntimeError(f'Text vectorizer compatibility check failed: {name}: {exc}') from exc
    return report

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_VERSION = getattr(xgb, "__version__", "0")
except Exception:
    xgb = None
    XGBClassifier = None
    XGBOOST_VERSION = "not installed"

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


import hashlib
import sys
import traceback
import warnings
import importlib.metadata
from decimal import Decimal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from urllib.parse import urlsplit

DISK_SCHEMA = [
    "sha256",
    "size",
    "type",
    "parent_sha256",
    "parent_size",
    "parent_type",
    "parent_imphash",
    "file_name",
    "file_path",
    "target",
]
MEMORY_SCHEMA = [
    "sha256",
    "normalized_path",
    "command_line",
    "parent_sha256",
    "total_processes",
    "created_files",
    "registry",
    "modules_count",
    "target",
]
MEMORY_SOURCE_PREDICTOR_COLUMNS = [
    "normalized_path",
    "command_line",
    "total_processes",
    "created_files",
    "registry",
    "modules_count",
]
MEMORY_SOURCE_NUMERIC_COLUMNS = [
    "total_processes",
    "modules_count",
]
MEMORY_SOURCE_TEXT_COLUMNS = [
    "normalized_path",
    "command_line",
    "created_files",
    "registry",
]
NETWORK_SCHEMA = [
    "sha256_sample",
    "avg_domain_len",
    "mean_domain_entropy",
    "n_public_service_domains",
    "n_unique_tlds",
    "n_public_ips",
    "n_unique_ips",
    "domains_per_conn",
    "target",
]
RANDOM_STATE = 42
SCRIPT_VERSION = "2026-11-08-evidence-audit-grouped-v20-gaussian-float64"
DEFAULT_SLURM_DATA_DIR = Path("/nfs/fjahiduzzaman/thesis/Feedback-September")
DEFAULT_SLURM_OUTPUT_DIR = DEFAULT_SLURM_DATA_DIR / "results" / "winmal_evidence_audit_v20"
DISK_PREDICTORS = ["size", "type", "parent_size", "parent_type", "parent_imphash", "file_name", "file_path"]
PREDICTORS = {"disk": DISK_PREDICTORS, "memory": MEMORY_SOURCE_PREDICTOR_COLUMNS,
              "network": NETWORK_SCHEMA[1:-1]}
SCHEMAS = {"disk": DISK_SCHEMA, "memory": MEMORY_SCHEMA, "network": NETWORK_SCHEMA}
NUMERIC = {"disk": ["size", "parent_size"], "memory": MEMORY_SOURCE_NUMERIC_COLUMNS,
           "network": NETWORK_SCHEMA[1:-1]}
CATEGORICAL = {"disk": ["type", "parent_type", "parent_imphash"], "memory": [], "network": []}
SUBJECT_COLUMN = {"disk": "parent_sha256", "memory": "parent_sha256", "network": "sha256_sample"}
OBJECT_COLUMN = {"disk": "sha256", "memory": "sha256", "network": "sha256_sample"}
IDENTIFIER_COLUMNS = {"sha256", "parent_sha256", "sha256_sample", "target", "_component", "_subject", "_row_id"}
PUBLIC_SERVICE_DOMAINS = ("google.com", "microsoft.com", "windows.com", "bing.com", "cloudflare.com", "amazonaws.com")

@dataclass(frozen=True)
class AnalysisOptions:
    """Controls optional RF SHAP and t-SNE analyses and sample sizes."""

    run_xai: bool = True
    run_tsne: bool = True
    run_tsne_3d: bool = True
    xai_background_size: int = 200
    xai_explain_size: int = 300
    tsne_max_points: int = 5000
    tsne_iterations: int = 1500
    tsne_perplexity: float = 30.0


def print_header(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def _cuda_allocation_visible() -> tuple[bool, str]:
    """Return whether this process may use a CUDA GPU and why."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        cleaned = cuda_visible.strip().lower()
        visible = cleaned not in {"", "-1", "none", "nodevfiles"}
        return visible, f"CUDA_VISIBLE_DEVICES={cuda_visible!r}"

    if os.environ.get("SLURM_JOB_ID"):
        allocation_vars = {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS")
            if os.environ.get(key)
        }
        if allocation_vars:
            return True, ", ".join(f"{key}={value}" for key, value in allocation_vars.items())
        return False, "SLURM job has no visible GPU allocation variable"

    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return True, probe.stdout.strip().splitlines()[0]
        return False, probe.stderr.strip() or "nvidia-smi found no GPU"
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi unavailable: {exc}"


def resolve_gpu_mode(requested: bool, require_gpu: bool) -> bool:
    """Validate the SLURM/CUDA allocation before building GPU estimators."""
    if require_gpu and not requested:
        raise ValueError("--require-gpu must be used together with --use-gpu.")
    if not requested:
        print("GPU acceleration: disabled by command line.")
        return False

    visible, reason = _cuda_allocation_visible()
    if visible:
        print(f"GPU acceleration: enabled ({reason}).")
        return True
    if require_gpu:
        raise RuntimeError(
            "A GPU was required, but no CUDA device is visible to this process. "
            f"Details: {reason}. Submit the supplied SLURM script with a GPU allocation."
        )
    print(f"[WARN] GPU requested but unavailable; using CPU. Details: {reason}")
    return False


def xgboost_tree_kwargs(use_gpu: bool) -> dict:
    """Support both XGBoost 1.x and 2.x GPU parameter conventions."""
    if not use_gpu:
        return {"tree_method": "hist"}
    try:
        major = int(str(XGBOOST_VERSION).split(".", 1)[0])
    except (TypeError, ValueError):
        major = 2
    if major >= 2:
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}


def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in str(s)).strip().replace(" ", "_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_clone(model):
    """Return a fresh unfitted estimator without copying fitted metadata.

    Some XGBoost/scikit-learn combinations expose ``feature_names_in_`` as a
    read-only property.  Falling back to ``deepcopy`` can then try to restore
    that fitted property and fail with ``AttributeError: ... has no setter``.
    Reconstruct third-party estimators from their constructor parameters and
    use sklearn's clone for native estimators so every CV fold remains clean.
    """
    module_name = model.__class__.__module__.lower()
    if "xgboost" in module_name and XGBClassifier is not None:
        return XGBClassifier(**model.get_params(deep=False))
    if "catboost" in module_name and CatBoostClassifier is not None:
        return CatBoostClassifier(**model.get_params())
    try:
        return clone(model)
    except Exception:
        try:
            return model.__class__(**model.get_params(deep=False))
        except Exception as rebuild_error:
            raise RuntimeError(
                f"Could not create a fresh {model.__class__.__name__} estimator."
            ) from rebuild_error


def model_input(estimator, values):
    """Use unnamed numeric matrices for XGBoost sklearn compatibility.

    Feature names are retained separately by the analysis for tables and
    plots.  This prevents newer sklearn validation code from assigning to
    XGBoost's read-only ``feature_names_in_`` property.
    """
    if "xgboost" in estimator.__class__.__module__.lower():
        if isinstance(values, pd.DataFrame):
            return values.to_numpy(dtype=np.float32, copy=False)
        if isinstance(values, pd.Series):
            return values.to_numpy(copy=False)
    return values


def add_bar_labels(bars, fmt: str = "{:.3f}", fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def add_count_labels(bars, fontsize: int = 9) -> None:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def build_search_dirs(cli_data_dir: str | None) -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    dirs = []
    if cli_data_dir:
        dirs.append(Path(cli_data_dir))
    dirs.extend([
        script_dir / "Dataset",
        script_dir,
        DEFAULT_SLURM_DATA_DIR,
        Path.cwd() / "Dataset",
        Path.cwd(),
    ])
    # unique and existing first; if not existing keep last resort paths too
    unique = []
    seen = set()
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def find_existing_file(search_dirs: list[Path], *filenames: str) -> Path:
    checked = []
    for d in search_dirs:
        for name in filenames:
            candidate = d / name
            checked.append(str(candidate))
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Could not find any of these files:\n  - "
        + "\n  - ".join(filenames)
        + "\n\nChecked locations:\n  - "
        + "\n  - ".join(checked)
    )


def canonical_target(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_process_count(value) -> int:
    """Parse either an integer count or a pipe-separated process list."""
    if pd.isna(value):
        return 0
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return max(0, int(numeric))
    return len([item for item in str(value).split("|") if item.strip()])


def align_proba_to_all_classes(proba, est_classes, all_classes):
    aligned = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    for j, c in enumerate(est_classes):
        if c in class_to_idx:
            aligned[:, class_to_idx[c]] = proba[:, j]
    return aligned


def plot_class_counts(y_names, out_path: Path, title: str) -> None:
    counts = pd.Series(y_names).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    colors = plt.cm.Greys(np.linspace(0.25, 0.85, len(counts)))
    bars = plt.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="black", linewidth=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    add_count_labels(bars)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_binary_or_multiclass_roc(y_true_all, y_prob_all, model_name: str, class_names, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    unique_classes = np.unique(y_true_all)

    if len(unique_classes) <= 2 and y_prob_all.shape[1] >= 2:
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    else:
        Y = label_binarize(y_true_all, classes=np.arange(len(class_names)))
        fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), y_prob_all.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, linewidth=2.2, label=f"micro-average (AUC = {auc_micro:.4f})")

        max_classes_to_plot = min(len(class_names), 10)
        for i in range(max_classes_to_plot):
            try:
                fpr_i, tpr_i, _ = roc_curve(Y[:, i], y_prob_all[:, i])
                auc_i = auc(fpr_i, tpr_i)
                plt.plot(fpr_i, tpr_i, linewidth=1.1, label=f"{class_names[i]} (AUC = {auc_i:.3f})")
            except Exception:
                pass

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_binary_or_multiclass_pr(y_true_all, y_prob_all, model_name: str, class_names, out_path: Path) -> None:
    """Save a binary or one-vs-rest multiclass precision-recall curve."""
    y_true_all = np.asarray(y_true_all)
    y_prob_all = np.asarray(y_prob_all)
    unique_classes = np.unique(y_true_all)
    fig, ax = plt.subplots(figsize=(7, 6))

    if len(unique_classes) <= 2 and y_prob_all.ndim == 2 and y_prob_all.shape[1] >= 2:
        precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all[:, 1])
        ap = average_precision_score(y_true_all, y_prob_all[:, 1])
        ax.plot(recall, precision, linewidth=2.2, label=f"PR (AP = {ap:.4f})")
        positive_rate = float(np.mean(y_true_all == unique_classes[-1]))
        ax.axhline(positive_rate, linestyle="--", linewidth=1, color="grey", label="No-skill baseline")
    else:
        classes = np.arange(len(class_names))
        Y = label_binarize(y_true_all, classes=classes)
        precision_micro, recall_micro, _ = precision_recall_curve(Y.ravel(), y_prob_all.ravel())
        ap_micro = average_precision_score(Y, y_prob_all, average="micro")
        ax.plot(recall_micro, precision_micro, linewidth=2.4, label=f"micro-average (AP = {ap_micro:.4f})")
        for idx in range(min(len(class_names), 10)):
            try:
                precision_i, recall_i, _ = precision_recall_curve(Y[:, idx], y_prob_all[:, idx])
                ap_i = average_precision_score(Y[:, idx], y_prob_all[:, idx])
                ax.plot(recall_i, precision_i, linewidth=1.1, label=f"{class_names[idx]} (AP = {ap_i:.3f})")
            except Exception:
                pass

    ax.set(xlim=(0, 1.0), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision", title=f"Precision-Recall Curves - {model_name}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _tsne_iteration_keyword(iterations: int) -> dict:
    """Support both old and new scikit-learn TSNE parameter names."""
    parameters = inspect.signature(TSNE).parameters
    if "max_iter" in parameters:
        return {"max_iter": iterations}
    return {"n_iter": iterations}


def run_tsne_analysis(
    X: pd.DataFrame,
    y,
    class_names: list[str],
    dataset_slug: str,
    out_dir: Path,
    options: AnalysisOptions,
) -> None:
    """Create the notebook-equivalent t-SNE plots and save their coordinates."""
    if not options.run_tsne:
        return
    print_header(f"{dataset_slug.upper()} t-SNE ANALYSIS")
    X_num = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y_arr = np.asarray(y)
    if len(X_num) < 4:
        print("[WARN] t-SNE skipped because fewer than four samples are available.")
        return

    if len(X_num) > options.tsne_max_points:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = np.sort(rng.choice(len(X_num), size=options.tsne_max_points, replace=False))
        X_num = X_num.iloc[selected].reset_index(drop=True)
        y_arr = y_arr[selected]
        print(f"Subsampled to {len(X_num):,} rows for t-SNE.")

    X_scaled = StandardScaler().fit_transform(X_num)
    X_embed = X_scaled
    pca_dims = min(50, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if X_scaled.shape[1] > pca_dims and pca_dims >= 2:
        X_embed = PCA(n_components=pca_dims, random_state=RANDOM_STATE).fit_transform(X_scaled)
        print(f"PCA pre-step: {X_scaled.shape[1]} -> {pca_dims} features")

    perplexity = min(float(options.tsne_perplexity), max(2.0, (len(X_embed) - 1) / 3.0))
    perplexity = min(perplexity, len(X_embed) - 1.0)
    common = dict(
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
        verbose=1,
        **_tsne_iteration_keyword(options.tsne_iterations),
    )

    z2 = TSNE(n_components=2, **common).fit_transform(X_embed)
    names = [str(class_names[int(v)]) if 0 <= int(v) < len(class_names) else str(v) for v in y_arr]
    emb2 = pd.DataFrame({
        "tSNE_1": z2[:, 0],
        "tSNE_2": z2[:, 1],
        "Class_ID": y_arr,
        "Class_Name": names,
    })
    emb2.to_csv(out_dir / f"{dataset_slug}_tsne_2d_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(z2[:, 0], z2[:, 1], c=y_arr, s=10, cmap="tab20", alpha=0.8)
    ax.set(title=f"t-SNE (2D) — {dataset_slug.title()} Dataset", xlabel="t-SNE-1", ylabel="t-SNE-2")
    fig.colorbar(scatter, ax=ax, label="Encoded Class")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"tsne_2d_{dataset_slug}.png", dpi=300)
    plt.close(fig)

    if options.run_tsne_3d:
        z3 = TSNE(n_components=3, **common).fit_transform(X_embed)
        emb3 = pd.DataFrame({
            "tSNE_1": z3[:, 0],
            "tSNE_2": z3[:, 1],
            "tSNE_3": z3[:, 2],
            "Class_ID": y_arr,
            "Class_Name": names,
        })
        emb3.to_csv(out_dir / f"{dataset_slug}_tsne_3d_coordinates.csv", index=False)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=y_arr, s=9, cmap="tab20", alpha=0.8)
        ax.set(
            title=f"t-SNE (3D) — {dataset_slug.title()} Dataset",
            xlabel="t-SNE-1",
            ylabel="t-SNE-2",
            zlabel="t-SNE-3",
        )
        fig.colorbar(scatter, ax=ax, label="Encoded Class", shrink=0.7)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"tsne_3d_{dataset_slug}.png", dpi=300)
        plt.close(fig)


def _normalise_shap_values(values_obj, n_samples: int, n_features: int) -> np.ndarray:
    """Normalise SHAP's version-dependent output to (sample, feature[, class])."""
    if isinstance(values_obj, list):
        arr = np.stack([np.asarray(v) for v in values_obj], axis=2)
    elif hasattr(values_obj, "values"):
        arr = np.asarray(values_obj.values)
    else:
        arr = np.asarray(values_obj)
    if arr.ndim == 3 and arr.shape[0] != n_samples and arr.shape[1:] == (n_samples, n_features):
        arr = np.moveaxis(arr, 0, 2)
    if arr.ndim not in (2, 3) or arr.shape[0] != n_samples or arr.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP array shape {arr.shape}; expected ({n_samples}, {n_features}[, classes]).")
    return arr


def build_disk_models(use_gpu: bool = False):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=CPU_THREADS),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=CPU_THREADS),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=3, n_jobs=CPU_THREADS),
        "Naive Bayes": GaussianNB(),
    }
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            loss_function="MultiClass",
            verbose=False,
            random_seed=RANDOM_STATE,
            thread_count=CPU_THREADS,
            **({"task_type": "GPU", "devices": "0"} if use_gpu else {}),
        )
    else:
        print("[WARN] catboost is not installed. Disk CatBoost model will be skipped.")
    return models


def build_memory_models(use_gpu: bool = False):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            C=3.0,
            class_weight="balanced",
            n_jobs=CPU_THREADS,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=CPU_THREADS,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=1,
            metric="cosine",
            algorithm="brute",
            n_jobs=CPU_THREADS,
        ),
        "Naive Bayes": MultinomialNB(alpha=1e-4),
    }
    if XGBClassifier is not None:
        xgb_kwargs = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            n_jobs=CPU_THREADS,
            random_state=RANDOM_STATE,
        )
        xgb_kwargs.update(xgboost_tree_kwargs(use_gpu))
        models["XGBoost"] = XGBClassifier(**xgb_kwargs)
    else:
        print("[WARN] xgboost is not installed. Memory XGBoost model will be skipped.")

    if CatBoostClassifier is not None:
        cb_kwargs = dict(
            iterations=500,
            learning_rate=0.08,
            depth=6,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=CPU_THREADS,
        )
        if use_gpu:
            cb_kwargs.update({"task_type": "GPU", "devices": "0"})
        models["CatBoost"] = CatBoostClassifier(**cb_kwargs)
    else:
        print("[WARN] catboost is not installed. Memory CatBoost model will be skipped.")
    return models


def build_network_models(use_gpu: bool = False):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, solver="lbfgs", n_jobs=CPU_THREADS),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=CPU_THREADS),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=CPU_THREADS),
        "Naive Bayes": GaussianNB(),
    }
    if XGBClassifier is not None:
        xgb_kwargs = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            n_jobs=CPU_THREADS,
            random_state=RANDOM_STATE,
        )
        xgb_kwargs.update(xgboost_tree_kwargs(use_gpu))
        models["XGBoost"] = XGBClassifier(**xgb_kwargs)
    else:
        print("[WARN] xgboost is not installed. Network XGBoost model will be skipped.")

    if CatBoostClassifier is not None:
        cb_kwargs = dict(
            iterations=800,
            learning_rate=0.08,
            depth=6,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=CPU_THREADS,
        )
        if use_gpu:
            cb_kwargs.update({"task_type": "GPU", "devices": "0"})
        models["CatBoost"] = CatBoostClassifier(**cb_kwargs)
    else:
        print("[WARN] catboost is not installed. Network CatBoost model will be skipped.")
    return models


class RandomForestSourceXAI:
    """Explain the fitted RF in each primary component-grouped fold on test rows.

    MDI rankings use all fitted folds. Global SHAP uses one seeded sample of
    held-out rows across those folds. Memory fields are masked as complete
    encoded groups, using the same exact source-level game in every fold.
    Local examples use the first eligible fold and median-confidence case
    within each outcome, so selection does not favour spectacular examples.
    """

    def __init__(self, X, y, class_names, dataset_slug, out_dir, options,
                 sample_ids=None, display_values=None, groups=None):
        self.enabled = options.run_xai
        if not self.enabled:
            return
        try:
            import shap
        except Exception as exc:
            raise RuntimeError(
                'Random Forest SHAP requires shap. Install it with: pip install shap. '
                'Use --skip-xai only when explanations are intentionally omitted.'
            ) from exc
        self.shap = shap
        self.slug = dataset_slug
        self.out_dir = Path(out_dir)
        self.plots_dir = ensure_dir(self.out_dir / 'plots')
        self.options = options
        self.y = np.asarray(y, dtype=int)
        self.class_names = [str(value) for value in class_names]
        self.features = [str(value) for value in X.columns]
        self.display_values = (X if display_values is None else display_values).reset_index(drop=True)
        self.sample_ids = (
            np.arange(len(X)).astype(str) if sample_ids is None
            else np.asarray(sample_ids).astype(str)
        )
        self.groups = None if groups is None else np.asarray(groups).astype(str)
        if len(self.display_values) != len(self.y) or len(self.sample_ids) != len(self.y):
            raise ValueError('RF explanation rows, labels, and sample identities must align.')
        rng = np.random.default_rng(RANDOM_STATE)
        self.global_indices = set(rng.choice(
            len(X), size=min(options.xai_explain_size, len(X)), replace=False
        ).tolist())
        self.fold_importances = []
        self.global_shap = []
        self.global_values = []
        self.global_row_indices = []
        self.global_methods = []
        self.global_bases = []
        self.global_probabilities = []
        self.global_fold_ids = []
        self.local_rows = {}
        self.local_tables = []
        self.oof_rows = []
        self.seen_indices = set()
        self.fold_audit = []
        self.tree_backend_failed = False
        self.pdp_status = 'not attempted'
        print_header(f'{dataset_slug.upper()} RANDOM FOREST FEATURE IMPORTANCE / SHAP')

    def _model_columns(self, estimator):
        if not isinstance(estimator, Pipeline):
            raise TypeError("XAI requires the evaluated fold-fitted source pipeline")
        forest = estimator.named_steps["classifier"]
        encoder = estimator.named_steps["features"]
        names = [str(v) for v in encoder.get_feature_names_out()]
        groups = {}
        for feature in self.features:
            block = encoder.output_indices_["field_" + feature]
            groups[feature] = list(range(block.start, block.stop))
        if not isinstance(forest, RandomForestClassifier):
            raise TypeError("RF explanations require the evaluated RandomForestClassifier")
        if len(names) != forest.n_features_in_:
            raise ValueError("Feature mapping differs from evaluated RF matrix")
        return forest, encoder, names, groups

    def _dense_view(self, values, encoder):
        matrix = encoder.transform(values) if encoder is not None else values
        if hasattr(matrix, 'toarray'):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def _source_plot_values(self, matrix, groups):
        values = np.zeros((len(matrix), len(self.features)), dtype=float)
        for index, source in enumerate(self.features):
            columns = groups[source]
            if not columns:
                continue
            selected = matrix[:, columns].astype(float)
            values[:, index] = (
                selected[:, 0] if len(columns) == 1
                else np.sqrt(np.mean(np.square(selected), axis=1))
            )
        return values

    def _source_view(self, matrix, shap_values, groups):
        values = self._source_plot_values(matrix, groups)
        grouped = np.zeros((len(matrix), len(self.features), shap_values.shape[2]), dtype=float)
        for index, source in enumerate(self.features):
            grouped[:, index, :] = shap_values[:, groups[source], :].sum(axis=1)
        return values, grouped

    def _exact_source_shapley(self, forest, matrix, background, names, groups,
                              model_probability, audit):
        """Enumerate the empirical-background game using actual RF predictions.

        v(S) = mean_b RF(x_S, b_not_S), with whole source blocks masked jointly.
        All 2**p coalitions are evaluated, then the ordinary Shapley formula
        allocates their marginal differences. Exactness is relative to the
        selected finite training background, not a population expectation.
        Sparse batching bounds Memory's encoded feature-matrix allocation.
        """
        count = len(self.features)
        if not 1 <= count <= 10:
            raise ValueError('Exact source SHAP supports one to ten source predictors.')
        owner = np.full(matrix.shape[1], -1, dtype=int)
        for feature_index, feature in enumerate(self.features):
            columns = groups[feature]
            if np.any(owner[columns] != -1):
                raise ValueError('Source SHAP groups overlap.')
            owner[columns] = feature_index
        if np.any(owner < 0):
            raise ValueError('Source SHAP groups do not cover every evaluated model feature.')
        masks = 1 << count
        coefficient = np.zeros((count, masks), dtype=float)
        for feature_index in range(count):
            bit = 1 << feature_index
            for mask in range(masks):
                if mask & bit:
                    continue
                # |S|! (p-|S|-1)! / p! = 1 / [p * choose(p-1, |S|)].
                weight = 1.0 / (count * math.comb(count - 1, mask.bit_count()))
                coefficient[feature_index, mask] -= weight
                coefficient[feature_index, mask | bit] += weight
        background = np.asarray(background, dtype=np.float32)
        rows_background = len(background)
        sparse_mode = matrix.shape[1] > 64
        sparse_background = csr_matrix(background) if sparse_mode else None
        observed_columns = {
            mask: ((mask >> owner) & 1).astype(bool) for mask in range(1, masks - 1)
        }
        # Cache only when the estimated sparse payload stays below 128 MiB.
        cache_sparse = (
            sparse_mode and masks * sparse_background.nnz * 12 / 2 <= 128 * 1024 * 1024
        )
        missing_background = {}

        def predict(values):
            if hasattr(forest, 'feature_names_in_'):
                values = pd.DataFrame(values, columns=names)
            probabilities = np.asarray(forest.predict_proba(values), dtype=float)
            if not np.isfinite(probabilities).all():
                raise RuntimeError('RF returned non-finite probabilities during exact SHAP.')
            return probabilities

        base = predict(sparse_background if sparse_mode else background).mean(axis=0)
        result = np.empty((len(matrix), count, len(forest.classes_)), dtype=float)
        coalitions_per_batch = max(1, 4096 // rows_background)
        intermediate_masks = list(range(1, masks - 1))
        for row_index, row in enumerate(matrix):
            game = np.empty((masks, len(forest.classes_)), dtype=float)
            game[0] = base
            game[-1] = model_probability[row_index]
            for offset in range(0, len(intermediate_masks), coalitions_per_batch):
                batch_masks = intermediate_masks[offset:offset + coalitions_per_batch]
                blocks = []
                for mask in batch_masks:
                    keep = observed_columns[mask]
                    if sparse_mode:
                        missing = missing_background.get(mask)
                        if missing is None:
                            missing = sparse_background.multiply(~keep).tocsr()
                            if cache_sparse:
                                missing_background[mask] = missing
                        indices = np.flatnonzero(keep & (row != 0))
                        nnz = len(indices)
                        repeated_row = csr_matrix((
                            np.tile(row[indices], rows_background),
                            np.tile(indices, rows_background),
                            np.arange(rows_background + 1, dtype=np.int64) * nnz,
                        ), shape=(rows_background, matrix.shape[1]))
                        blocks.append(missing + repeated_row)
                    else:
                        blocks.append(np.where(keep[None, :], row[None, :], background))
                combined = vstack(blocks, format='csr') if sparse_mode else np.concatenate(blocks, axis=0)
                probabilities = predict(combined).reshape(
                    len(batch_masks), rows_background, len(forest.classes_)
                ).mean(axis=1)
                game[batch_masks] = probabilities
            result[row_index] = coefficient @ game
            if row_index == 0 or (row_index + 1) % 5 == 0 or row_index + 1 == len(matrix):
                print(f'[RF SHAP:{self.slug}] exact source SHAP: row {row_index + 1}/{len(matrix)}', flush=True)
        reconstruction = base[None, :] + result.sum(axis=1)
        error = float(np.max(np.abs(reconstruction - model_probability)))
        if not np.isfinite(result).all() or not np.allclose(
            reconstruction, model_probability, atol=1e-8, rtol=1e-7
        ):
            raise RuntimeError(f'Exact source SHAP failed its independent probability check: {error:.6g}')
        audit['SHAP Method'] = 'exact source-level interventional Shapley'
        audit['SHAP Output Mode'] = 'predict_proba'
        audit['Effective Background Rows'] = rows_background
        audit['Exact Coalitions Per Row'] = masks
        audit['Maximum Probability Reconstruction Error'] = error
        return result, base

    def _verified_interventional_shap(self, forest, matrix, background, names,
                                     model_probability, audit):
        """Use training-only background data and reject invalid attributions.

        Some SHAP versions expose RF class probabilities through 'raw' rather
        than 'probability'. Both routes use the same empirical background, and
        both must reconstruct the actual predict_proba output to be accepted.
        No model is retrained and no additivity tolerance is relaxed on retry.
        """
        failures = []
        # Preserve the RF's float32 input values, but pass a contiguous float64
        # matrix to SHAP's compiled implementation for numerical calculations.
        explain_data = np.ascontiguousarray(matrix, dtype=np.float64)
        background_data = np.ascontiguousarray(background, dtype=np.float64)
        for output_mode in ['probability', 'raw']:
            try:
                masker = self.shap.maskers.Independent(
                    background_data, max_samples=len(background_data)
                )
                explainer = self.shap.TreeExplainer(
                    forest, data=masker, feature_perturbation='interventional',
                    model_output=output_mode,
                )
                raw_values = explainer.shap_values(explain_data, check_additivity=True)
                values = _normalise_shap_values(raw_values, len(matrix), len(names))
                base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
                if values.ndim == 2:
                    if len(forest.classes_) == 2 and len(base) == 1:
                        values = np.stack([-values, values], axis=2)
                        base = np.array([1.0 - base[0], base[0]])
                    elif len(forest.classes_) == 1:
                        values = values[:, :, None]
                    else:
                        raise ValueError('Unexpected Random Forest SHAP class dimensions.')
                expected_shape = (len(matrix), len(names), len(forest.classes_))
                if values.shape != expected_shape or len(base) != len(forest.classes_):
                    raise ValueError(f'RF SHAP shape {values.shape}; expected {expected_shape}.')
                reconstruction = base[None, :] + values.sum(axis=1)
                error = float(np.max(np.abs(reconstruction - model_probability)))
                if not np.isfinite(values).all() or not np.isfinite(base).all():
                    raise ValueError('RF SHAP produced non-finite contributions or base values.')
                if not np.allclose(reconstruction, model_probability, atol=1e-5, rtol=1e-4):
                    raise ValueError(f'RF probability reconstruction error={error:.6g}')
                audit['Maximum Probability Reconstruction Error'] = error
                audit['SHAP Output Mode'] = output_mode
                audit['SHAP Retry Notes'] = ' | '.join(failures)
                audit['Effective Background Rows'] = len(explainer.data)
                audit['SHAP Method'] = 'interventional Tree SHAP'
                return values, base
            except Exception as exc:
                failures.append(f'{output_mode}: {type(exc).__name__}: {exc}')
                if output_mode == 'probability':
                    print(f'[RF SHAP:{self.slug}] Probability route rejected: {str(exc)[:160]}. '
                          'Retrying the RF raw-probability interface with the same background.')
                else:
                    print(f'[RF SHAP:{self.slug}] Both interventional routes failed verification.')
        diagnostic = {
            'source': self.slug, 'fold': audit['Fold'], 'seed': RANDOM_STATE,
            'shap_version': getattr(self.shap, '__version__', 'unknown'),
            'sklearn_version': __import__('sklearn').__version__,
            'model_input_features': int(forest.n_features_in_),
            'explanation_shape': list(matrix.shape),
            'background_shape': list(background.shape),
            'audit': audit, 'errors': failures,
        }
        diagnostic_path = self.out_dir / f'{self.slug}_rf_shap_failure_fold_{audit["Fold"]}.json'
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
        raise RuntimeError(
            f'{self.slug} RF SHAP could not produce verified class-probability '
            f'explanations in fold {audit["Fold"]}. Invalid values were not exported. '
            f'Details: {diagnostic_path}'
        )

    def record_fold(self, estimator, X_test, train_indices, test_indices, predictions,
                    probabilities, training_values=None):
        if not self.enabled:
            return
        train_indices, test_indices = np.asarray(train_indices), np.asarray(test_indices)
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError('RF SHAP received overlapping training and test rows.')
        if self.seen_indices.intersection(test_indices.tolist()):
            raise ValueError('RF SHAP received repeated out-of-fold test rows.')
        if self.groups is not None and np.intersect1d(
            self.groups[train_indices], self.groups[test_indices]
        ).size:
            raise ValueError('RF SHAP detected training/test group overlap.')
        self.seen_indices.update(test_indices.tolist())
        fold = len(self.fold_audit) + 1
        forest, encoder, names, groups = self._model_columns(estimator)
        importance = np.array([
            forest.feature_importances_[groups[source]].sum() for source in self.features
        ])
        self.fold_importances.append(importance)
        predicted = np.asarray(predictions, dtype=int)
        probability = np.asarray(probabilities, dtype=float)
        confidence = probability.max(axis=1)
        correct = predicted == self.y[test_indices]
        for position, row_index in enumerate(test_indices):
            self.oof_rows.append({
                'Dataset Row Index': int(row_index), 'Sample ID': self.sample_ids[row_index],
                'Fold': fold, 'True Class': self.class_names[self.y[row_index]],
                'Predicted Class': self.class_names[predicted[position]],
                'Correct': bool(correct[position]),
                'Prediction Confidence': float(confidence[position]),
            })
        selected_local = {}
        for category, mask in [('correct', correct), ('incorrect', ~correct)]:
            candidates = np.flatnonzero(mask)
            if category not in self.local_rows and len(candidates):
                median_confidence = np.median(confidence[candidates])
                selected_local[category] = min(candidates.tolist(), key=lambda index: (
                    abs(confidence[index] - median_confidence), int(test_indices[index])
                ))
        global_positions = [
            index for index, row in enumerate(test_indices) if int(row) in self.global_indices
        ]
        selected_positions = sorted(set(global_positions) | set(selected_local.values()))
        audit = {
            'Fold': fold, 'Training Rows': len(train_indices), 'Test Rows': len(test_indices),
            'Internal RF Features': len(names), 'Global SHAP Rows': len(global_positions),
            'Local Categories Selected': ' | '.join(selected_local),
            'Maximum Probability Reconstruction Error': None,
        }
        self.fold_audit.append(audit)
        if not selected_positions:
            return
        if training_values is None or len(training_values) != len(train_indices):
            raise ValueError('RF SHAP needs aligned training-fold values for its background.')
        subset = X_test.iloc[selected_positions] if isinstance(X_test, pd.DataFrame) else X_test[selected_positions]
        matrix = self._dense_view(subset, encoder)
        background_rng = np.random.default_rng(RANDOM_STATE + fold)
        background_positions = np.sort(background_rng.choice(
            len(train_indices), size=min(self.options.xai_background_size, len(train_indices)),
            replace=False,
        ))
        background_rows = (
            training_values.iloc[background_positions]
            if isinstance(training_values, pd.DataFrame)
            else training_values[background_positions]
        )
        background = self._dense_view(background_rows, encoder)
        if matrix.shape[1] != forest.n_features_in_ or background.shape[1] != forest.n_features_in_:
            raise ValueError('Training background, held-out rows, and fitted RF feature counts differ.')
        if not np.isfinite(matrix).all() or not np.isfinite(background).all():
            raise ValueError('RF SHAP requires finite training background and held-out feature values.')
        audit['Background Rows'] = len(background)
        audit['Background Dataset Row Indices'] = json.dumps(train_indices[background_positions].tolist())
        audit['SHAP Perturbation'] = 'interventional'
        fitted_classes = np.asarray(forest.classes_, dtype=int)
        prediction_matrix = (
            pd.DataFrame(matrix, columns=names)
            if hasattr(forest, 'feature_names_in_') else matrix
        )
        model_probability = forest.predict_proba(prediction_matrix)
        aligned_probability = align_proba_to_all_classes(
            model_probability, fitted_classes, np.arange(len(self.class_names))
        )
        if not np.allclose(aligned_probability, probability[selected_positions], atol=1e-6):
            raise RuntimeError('RF SHAP model predictions differ from the evaluated CV model.')
        source_values = self._source_plot_values(matrix, groups)
        if self.slug in {'memory', 'disk'} or self.tree_backend_failed:
            # Use the same whole-source game in every Memory fold. Summing
            # separate token SHAP values defines a different game and is not
            # mixed with these source-level values in the global ranking.
            audit['SHAP Retry Notes'] = (
                'Whole-source exact game used consistently for all encoded-text/categorical folds.'
                if self.slug in {'memory', 'disk'} else
                'Tree backend failed earlier; continuing the same interventional game with exact source SHAP.'
            )
            source_shap, base_values = self._exact_source_shapley(
                forest, matrix, background, names, groups, model_probability, audit
            )
        else:
            # Disk and Network have one model dimension per source predictor,
            # so this exact fallback evaluates the same interventional game
            # and the same background as their Tree SHAP calculation.
            try:
                shap_values, base_values = self._verified_interventional_shap(
                    forest, matrix, background, names, model_probability, audit
                )
                _, source_shap = self._source_view(matrix, shap_values, groups)
            except Exception as exc:
                self.tree_backend_failed = True
                audit['Tree SHAP Failure'] = str(exc)
                audit['SHAP Retry Notes'] = str(exc)
                print(f'[RF SHAP:{self.slug}] Tree SHAP unavailable in fold {fold}; '
                      'recovering with exact source SHAP from RF predict_proba.', flush=True)
                source_shap, base_values = self._exact_source_shapley(
                    forest, matrix, background, names, groups, model_probability, audit
                )
                diagnostic_path = self.out_dir / f'{self.slug}_rf_shap_failure_fold_{fold}.json'
                if diagnostic_path.exists():
                    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
                    diagnostic['recovery_status'] = 'recovered using exact source-level interventional Shapley'
                    diagnostic['recovery_max_probability_error'] = audit['Maximum Probability Reconstruction Error']
                    diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
        if not np.allclose(base_values[None, :] + source_shap.sum(axis=1), model_probability,
                           atol=1e-5, rtol=1e-4):
            raise RuntimeError('Source SHAP contributions differ from the evaluated RF probabilities.')
        error = audit['Maximum Probability Reconstruction Error']
        all_class_shap = np.zeros((len(matrix), len(self.features), len(self.class_names)))
        all_class_shap[:, :, fitted_classes] = source_shap
        selected_lookup = {position: index for index, position in enumerate(selected_positions)}
        for position in global_positions:
            index = selected_lookup[position]
            self.global_shap.append(all_class_shap[index])
            self.global_values.append(source_values[index])
            self.global_row_indices.append(int(test_indices[position]))
            self.global_methods.append(audit['SHAP Method'])
            full_base = np.zeros(len(self.class_names))
            full_base[fitted_classes] = base_values
            self.global_bases.append(full_base)
            self.global_probabilities.append(aligned_probability[index])
            self.global_fold_ids.append(fold)
        for category, position in selected_local.items():
            index = selected_lookup[position]
            row_index = int(test_indices[position])
            predicted_class = int(predicted[position])
            true_class = int(self.y[row_index])
            predicted_axis = int(np.flatnonzero(fitted_classes == predicted_class)[0])
            true_axes = np.flatnonzero(fitted_classes == true_class)
            true_values = (
                source_shap[index, :, int(true_axes[0])] if len(true_axes)
                else np.full(len(self.features), np.nan)
            )
            local_values = source_shap[index, :, predicted_axis]
            base = float(base_values[predicted_axis])
            metadata = {
                'Category': category, 'Status': 'available', 'Source': self.slug,
                'Model': 'Random Forest', 'Random Seed': RANDOM_STATE,
                'SHAP Method': audit['SHAP Method'], 'Background Rows': len(background),
                'Fold': fold, 'Dataset Row Index': row_index,
                'Sample ID': self.sample_ids[row_index],
                'True Class': self.class_names[true_class],
                'Predicted Class': self.class_names[predicted_class],
                'Prediction Confidence': float(probability[position, predicted_class]),
                'True Class Probability': float(probability[position, true_class]),
                'Explained Class': self.class_names[predicted_class],
                'Base Probability': base, 'SHAP Sum': float(local_values.sum()),
                'Reconstructed Probability': base + float(local_values.sum()),
                'Selection Rule': (
                    'First CV fold containing this outcome; closest to its median '
                    'predicted-class confidence; row-index tie break'
                ),
            }
            self.local_rows[category] = metadata
            table = pd.DataFrame({
                'Feature': self.features,
                'Source Value': [str(self.display_values.iloc[row_index][feature]) for feature in self.features],
                'Encoded Value or Group RMS': source_values[index],
                'SHAP Value': local_values,
                'SHAP Value for True Class': true_values,
                'Absolute SHAP': np.abs(local_values),
            }).sort_values('Absolute SHAP', ascending=False)
            for key in ['Category', 'Fold', 'Sample ID', 'True Class', 'Predicted Class']:
                table.insert(0, key, metadata[key])
            table.to_csv(self.out_dir / f'{self.slug}_shap_local_{category}.csv', index=False)
            self.local_tables.append(table)
            one = self.shap.Explanation(
                values=local_values, base_values=base, data=source_values[index],
                feature_names=self.features,
            )
            self.shap.plots.waterfall(one, max_display=min(20, len(self.features)), show=False)
            plt.title(
                f'{self.slug.title()} RF — {category.title()} held-out prediction\n'
                f'True: {metadata["True Class"]}; predicted: {metadata["Predicted Class"]}'
            )
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'shap_waterfall_{self.slug}_{category}.png', dpi=300, bbox_inches='tight')
            plt.close()
            try:
                force = self.shap.force_plot(base, local_values, source_values[index],
                                             feature_names=self.features, matplotlib=False)
                self.shap.save_html(str(self.out_dir / f'{self.slug}_shap_force_{category}.html'), force)
            except Exception as exc:
                print(f'[WARN] {self.slug} {category} interactive force plot skipped: {exc}')
        if self.pdp_status == 'not attempted':
            self._pdp(forest, matrix, names, groups, importance)
        print(f'[RF SHAP:{self.slug}] fold {fold}: {len(global_positions)} global rows; '
              f'local={list(selected_local)}; probability error={error:.2g}')

    def _pdp(self, forest, matrix, names, groups, importance):
        eligible = [
            index for index, feature in enumerate(self.features)
            if len(groups[feature]) == 1 and (
                feature in NUMERIC[self.slug]
            )
        ]
        if not eligible or len(matrix) < 2:
            self.pdp_status = 'unavailable: no eligible numeric feature or too few rows'
            return
        source_index = max(eligible, key=lambda index: importance[index])
        source = self.features[source_index]
        feature_index = groups[source][0]
        target_class = int(pd.Series(self.y).value_counts().idxmax())
        if target_class not in forest.classes_:
            target_class = int(forest.classes_[0])
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            pdp_matrix = (
                pd.DataFrame(matrix, columns=names)
                if hasattr(forest, 'feature_names_in_') else matrix
            )
            PartialDependenceDisplay.from_estimator(
                forest, pdp_matrix, [feature_index], kind='both', grid_resolution=30,
                response_method='predict_proba', method='brute', target=target_class,
                feature_names=names, ax=ax,
            )
            ax.set_title(f'{self.slug.title()} RF — held-out fold PDP / ICE: {source}')
            ax.set_xlabel(f'{source} (model input units)')
            fig.tight_layout()
            fig.savefig(self.plots_dir / f'pdp_ice_{self.slug}_{sanitize_filename(source)}.png', dpi=300)
            plt.close(fig)
            self.pdp_status = f'available: {source}; first explained CV fold'
        except Exception as exc:
            plt.close('all')
            self.pdp_status = f'skipped: {exc}'
            print(f'[WARN] RF PDP / ICE skipped: {exc}')

    def _ranking_plot(self, frame, column, title, xlabel, filename, error_column=None):
        top = frame.sort_values(column, ascending=False).head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(top) + 2)))
        error = None if error_column is None else top[error_column].to_numpy()
        ax.barh(top['Feature'], top[column], xerr=error, capsize=3,
                color=plt.cm.Blues(0.65), edgecolor='black')
        ax.set(title=title, xlabel=xlabel)
        fig.tight_layout()
        fig.savefig(self.plots_dir / filename, dpi=300)
        plt.close(fig)

    def finalize(self):
        if not self.enabled:
            return
        if len(self.seen_indices) != len(self.y):
            raise RuntimeError('RF explanation collection did not cover every CV test row once.')
        fold_values = np.asarray(self.fold_importances)
        basic = pd.DataFrame({
            'Feature': self.features, 'RF Importance Mean': fold_values.mean(axis=0),
            'RF Importance Std': fold_values.std(axis=0, ddof=1) if len(fold_values) > 1 else np.zeros(len(self.features)),
        })
        basic['RF Rank'] = basic['RF Importance Mean'].rank(ascending=False, method='min').astype(int)
        basic = basic.sort_values(['RF Rank', 'Feature']).reset_index(drop=True)
        basic.to_csv(self.out_dir / f'{self.slug}_rf_feature_importance.csv', index=False)
        basic.to_excel(self.out_dir / f'{self.slug}_rf_feature_importance.xlsx', index=False)
        fold_table = pd.DataFrame(fold_values, columns=self.features)
        fold_table.insert(0, 'Fold', np.arange(1, len(fold_values) + 1))
        fold_table.to_csv(self.out_dir / f'{self.slug}_rf_feature_importance_by_fold.csv', index=False)
        self._ranking_plot(
            basic, 'RF Importance Mean', f'{self.slug.title()} RF — Basic Feature Importance',
            'Mean decrease in impurity (mean ± SD across CV folds)',
            f'rf_feature_importance_{self.slug}.png', error_column='RF Importance Std',
        )
        if not self.global_shap:
            raise RuntimeError('No held-out global RF SHAP rows were collected.')
        order = np.argsort(self.global_row_indices)
        shap_values = np.asarray(self.global_shap)[order]
        model_values = np.asarray(self.global_values)[order]
        row_indices = np.asarray(self.global_row_indices)[order]
        importance = np.abs(shap_values).mean(axis=(0, 2))
        shap_table = pd.DataFrame({'Feature': self.features, 'Mean Absolute SHAP': importance})
        shap_table['SHAP Rank'] = shap_table['Mean Absolute SHAP'].rank(ascending=False, method='min').astype(int)
        shap_table = shap_table.sort_values(['SHAP Rank', 'Feature']).reset_index(drop=True)
        shap_table.to_csv(self.out_dir / f'{self.slug}_shap_global_importance.csv', index=False)
        shap_table.to_excel(self.out_dir / f'{self.slug}_shap_global_importance.xlsx', index=False)
        comparison = basic.merge(shap_table, on='Feature', validate='one_to_one').sort_values('SHAP Rank')
        comparison.to_csv(self.out_dir / f'{self.slug}_rf_vs_shap_feature_ranking.csv', index=False)
        comparison.to_excel(self.out_dir / f'{self.slug}_rf_vs_shap_feature_ranking.xlsx', index=False)
        self._ranking_plot(
            shap_table, 'Mean Absolute SHAP', f'{self.slug.title()} RF — Held-out Global SHAP Importance',
            'Mean |SHAP| across sampled test rows and all classes (probability units)',
            f'shap_global_bar_{self.slug}.png',
        )
        majority_class = int(pd.Series(self.y).value_counts().idxmax())
        plot_values = shap_values[:, :, majority_class]
        report_frame = pd.DataFrame(model_values, columns=self.features)
        self.shap.summary_plot(plot_values, report_frame, max_display=20, show=False)
        plt.title(f'{self.slug.title()} RF — held-out SHAP: {self.class_names[majority_class]}')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'shap_beeswarm_{self.slug}.png', dpi=300, bbox_inches='tight')
        plt.close()
        top_feature = str(shap_table.iloc[0]['Feature'])
        top_index = self.features.index(top_feature)
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(model_values[:, top_index], plot_values[:, top_index],
                            c=self.y[row_indices], cmap='tab20', alpha=0.8, s=20)
        ax.set(title=f'{self.slug.title()} RF — SHAP Dependence: {top_feature}',
               xlabel=f'{top_feature} (encoded value / text-group RMS)',
               ylabel=f'SHAP for {self.class_names[majority_class]} (probability units)')
        fig.colorbar(points, ax=ax, label='True class index')
        fig.tight_layout()
        fig.savefig(self.plots_dir / f'shap_dependence_{self.slug}_{sanitize_filename(top_feature)}.png', dpi=300)
        plt.close(fig)
        for category in ['correct', 'incorrect']:
            if category not in self.local_rows:
                for stale in [
                    self.out_dir / f'{self.slug}_shap_local_{category}.csv',
                    self.out_dir / f'{self.slug}_shap_force_{category}.html',
                    self.plots_dir / f'shap_waterfall_{self.slug}_{category}.png',
                ]:
                    stale.unlink(missing_ok=True)
                self.local_rows[category] = {
                    'Category': category, 'Status': f'unavailable: no {category} OOF predictions',
                    'Source': self.slug, 'Model': 'Random Forest', 'Random Seed': RANDOM_STATE,
                }
                print(f'[RF SHAP:{self.slug}] No {category} held-out example exists; no example fabricated.')
        local_summary = pd.DataFrame([self.local_rows[key] for key in ['correct', 'incorrect']])
        local_summary.to_csv(self.out_dir / f'{self.slug}_shap_local_examples_summary.csv', index=False)
        local_summary.to_excel(self.out_dir / f'{self.slug}_shap_local_examples_summary.xlsx', index=False)
        if self.local_tables:
            pd.concat(self.local_tables, ignore_index=True).to_csv(
                self.out_dir / f'{self.slug}_shap_local_explanation.csv', index=False
            )
        pd.DataFrame(self.oof_rows).sort_values('Dataset Row Index').to_csv(
            self.out_dir / f'{self.slug}_rf_oof_predictions_for_shap.csv', index=False
        )
        pd.DataFrame(self.fold_audit).to_csv(self.out_dir / f'{self.slug}_rf_shap_fold_audit.csv', index=False)
        np.savez_compressed(
            self.out_dir / f'{self.slug}_rf_shap_heldout_values.npz',
            shap_values=shap_values, model_values=model_values, row_indices=row_indices,
            feature_names=np.asarray(self.features), class_names=np.asarray(self.class_names),
            methods=np.asarray(self.global_methods)[order],
            base_values=np.asarray(self.global_bases)[order],
            probabilities=np.asarray(self.global_probabilities)[order],
            fold_ids=np.asarray(self.global_fold_ids)[order],
        )
        correlation = (
            float(stats.spearmanr(comparison['RF Importance Mean'], comparison['Mean Absolute SHAP']).statistic)
            if comparison['RF Importance Mean'].nunique() > 1 and comparison['Mean Absolute SHAP'].nunique() > 1
            else None
        )
        metadata = {
            'Dataset': self.slug, 'XAI Model': 'Random Forest', 'Random Seed': RANDOM_STATE,
            'Validation Folds': len(self.fold_audit), 'OOF Rows': len(self.oof_rows),
            'Global Explained Rows': len(row_indices), 'Reported Features': self.features,
            'Basic Importance': 'RF mean decrease in impurity, averaged across existing CV folds',
            'Global SHAP': 'mean absolute source contribution over sampled test rows and all classes',
            'Feature Selection': 'all evaluated source predictors retained; rankings are descriptive',
            'SHAP Method': (
                'Exact whole-source interventional Shapley in every encoded-field fold'
                if self.slug in {'memory', 'disk'} else
                'Interventional Tree SHAP with exact same-game source fallback'
            ),
            'SHAP Methods Used': sorted(set(self.global_methods)),
            'Tree Backend Disabled After Failure': self.tree_backend_failed,
            'Background Rows Requested': self.options.xai_background_size,
            'SHAP Version': getattr(self.shap, '__version__', 'unknown'),
            'scikit-learn Version': __import__('sklearn').__version__,
            'SHAP Units': 'class probability; additivity checked against the evaluated RF predictions',
            'Text Feature Values': 'encoded RMS for plots; original source text in local CSV tables',
            'Encoded SHAP Grouping': (
                'Whole encoded source-field blocks masked jointly in an exact source-field game'
                if self.slug in {'memory', 'disk'} else
                'One numeric source predictor per game feature; Tree and exact fallback use the same game'
            ),
            'Rank Spearman Correlation': correlation,
            'Local Selection': 'first eligible test fold; median confidence within each outcome',
            'Grouped Validation': self.groups is not None,
            'PDP / ICE': self.pdp_status,
            'Notes': 'Uses component-grouped primary folds and training-only preprocessing; RF is prespecified, not selected as the best classifier. Text RMS colors are not original evidence values; local CSVs preserve original fields.',
        }
        with (self.out_dir / f'{self.slug}_xai_metadata.json').open('w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
        pd.DataFrame([{
            **metadata, 'Reported Features': ' | '.join(self.features),
        }]).to_csv(self.out_dir / f'{self.slug}_xai_metadata.csv', index=False)
        # Retire earlier single-example files so an old training example cannot
        # be mistaken for either of the new held-out outcome examples.
        for old_file in [
            self.plots_dir / f'shap_waterfall_{self.slug}.png',
            self.out_dir / f'{self.slug}_shap_force_plot.html',
            self.out_dir / 'memory_command_line_local_shap.csv' if self.slug in {'memory', 'disk'} else None,
        ]:
            if old_file is not None:
                old_file.unlink(missing_ok=True)
        print(f'[RF SHAP:{self.slug}] Feature rankings and held-out explanations saved to {self.out_dir}')



# -----------------------------------------------------------------------------
# Auditable cohort preparation. No full-data learned transformation occurs here.
# -----------------------------------------------------------------------------
def write_json(path, value):
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, (Path, datetime, pd.Timestamp)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        return obj
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(clean(value), indent=2, default=convert, allow_nan=False), encoding='utf-8')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t')
    return pd.read_csv(path)


def canonical_source_columns(frame):
    """Normalize header spelling only, never infer missing evidence identities."""
    aliases = {'sha_256': 'sha256', 'parent_sha_256': 'parent_sha256'}
    names = []
    for column in frame.columns:
        value = re.sub(r'[\s-]+', '_', str(column).strip().lower())
        names.append(aliases.get(value, value))
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source column names after header normalization')
    result = frame.copy()
    result.columns = names
    return result


def normalize_source_nominals(frame, source):
    """Return documented categorical normalization, not fabricated raw evidence.

    Legacy numeric category codes remain nominal tokens. OneHotEncoder is still
    fitted on training data only. A token is NOT a recovered import hash/type.
    Encoded path/command/registry text cannot substitute for original text.
    """
    notes = []
    missing_strings = {'', 'nan', 'none', 'null', 'unknown', 'na', 'n/a'}
    for column in set(PREDICTORS[source]) - set(NUMERIC[source]):
        if column not in frame:
            continue
        values = frame[column]
        observed = values.dropna()
        if column in CATEGORICAL[source]:
            opaque_count = 0
            missing_code_count = 0
            def nominal(value):
                nonlocal opaque_count, missing_code_count
                if pd.isna(value):
                    return ''
                text = str(value).strip()
                lower = text.lower()
                if lower in missing_strings:
                    return ''
                # Preserve a genuine 32-hex-character imphash STRING before
                # interpreting numeric-looking legacy codes. Lost Excel digits
                # or missing original hashes are never reconstructed.
                if column == 'parent_imphash' and isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{32}', text):
                    return lower
                if isinstance(value, (int, float, np.integer, np.floating)) or re.fullmatch(r'[+-]?[0-9]+(?:\.[0-9]+)?', text):
                    number = Decimal(str(value))
                    if not number.is_finite():
                        return ''
                    if column == 'parent_imphash' and number in {Decimal(0), Decimal(-1)}:
                        missing_code_count += 1
                        return ''
                    opaque_count += 1
                    token = (str(int(number)) if number == number.to_integral_value()
                             else format(number, 'f').rstrip('0').rstrip('.'))
                    return 'opaque_numeric_' + column + ':' + token
                return text
            frame[column] = values.map(nominal)
            if opaque_count or missing_code_count:
                notes.append({'field': column, 'opaque_numeric_rows': opaque_count,
                    'missing_sentinel_rows': missing_code_count,
                    'interpretation': 'opaque nominal codes; original raw hash/type not reconstructed',
                    'preprocessing': 'training-fold one-hot encoding; no numeric distance/order assigned'})
        elif len(observed) and pd.api.types.is_numeric_dtype(observed):
            # Missing source text is often stored as an all-zero placeholder.
            # Treat that as absent evidence, not as a learned vocabulary.
            numeric = pd.to_numeric(observed, errors='coerce')
            if numeric.isin([0, -1]).all():
                frame[column] = ''
                notes.append({'field': column, 'interpretation': 'numeric missing-only text field; kept empty'})
            else:
                raise ValueError(f'{source}.{column} contains numeric encoded text. Use the source-schema workbook with original strings; numeric codes cannot reconstruct this field.')
    return frame, notes


_SOURCE_PROBE_CACHE = {}


def probe_source_candidate(path, source):
    """Inspect schemas without fitting features or using targets for ranking."""
    path = Path(path).resolve()
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns, source)
    if cache_key in _SOURCE_PROBE_CACHE:
        return dict(_SOURCE_PROBE_CACHE[cache_key])
    entry = {'path': str(path), 'compatible': False}
    try:
        if path.suffix.lower() in {'.xlsx', '.xls'}:
            sample = pd.read_excel(path, nrows=64)
        elif path.suffix.lower() == '.parquet':
            sample = pd.read_parquet(path).head(64)
        else:
            sample = pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',', nrows=64)
        sample = canonical_source_columns(sample)
        entry['columns'] = list(sample.columns)
        raw_network = source == 'network' and bool({'domains', 'hosts'} & set(sample.columns))
        required = {'sha256_sample', 'target'} if raw_network else set(SCHEMAS[source])
        missing = sorted(required - set(sample.columns))
        entry['missing_required_fields'] = missing
        if missing:
            entry['reason'] = 'missing original source fields or originating-subject identity'
        else:
            _, notes = normalize_source_nominals(sample, source)
            entry.update(compatible=True, source_normalization_notes=notes,
                         opaque_category_fields=sum(n.get('opaque_numeric_rows', 0) > 0 for n in notes),
                         raw_network_lists=raw_network)
    except Exception as error:
        entry['reason'] = f'{type(error).__name__}: {error}'
    _SOURCE_PROBE_CACHE[cache_key] = dict(entry)
    return entry


def source_file(source, directories, explicit=None, selection_report=None):
    """Select a source-schema table, skipping reduced/encoded-text candidates.

    Explicit paths are never silently replaced. Automatic selection inspects
    matching filenames, including downloaded '(n)' copies, in directory order.
    Within one directory original nominal fields precede opaque coded fields;
    non-preprocessed tables then precede preprocessed copies. No score/label
    purity or target distribution is consulted for selection.
    """
    entries = []
    if explicit:
        path = Path(explicit)
        if not path.is_absolute() and not path.exists():
            path = Path(directories[0]) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = probe_source_candidate(path, source)
        entries.append(entry)
        selected = path.resolve() if entry['compatible'] else None
    else:
        selected = None
        seen = set()
        suffixes = {'.xlsx', '.xls', '.csv', '.tsv', '.parquet'}
        for directory in directories:
            directory = Path(directory)
            if not directory.is_dir():
                continue
            paths = sorted((p for p in directory.iterdir() if p.is_file() and
                p.suffix.lower() in suffixes and
                p.name.lower().startswith(source + '_dataset')), key=lambda p: p.name.lower())
            candidates = []
            for path in paths:
                if str(path.resolve()) in seen:
                    continue
                seen.add(str(path.resolve()))
                entry = probe_source_candidate(path, source)
                entries.append(entry)
                if entry['compatible']:
                    candidates.append((entry, path))
            if candidates:
                def rank(pair):
                    entry, path = pair
                    name = path.stem.lower()
                    return (entry.get('opaque_category_fields', 0), 'preprocessed' in name,
                            name != source + '_dataset', name)
                selected = min(candidates, key=rank)[1].resolve()
                break
    report = {'source': source, 'explicit_input': explicit, 'selected_path': str(selected) if selected else None,
              'selection_uses_labels_or_model_scores': False, 'candidates': entries}
    if selection_report:
        write_json(selection_report, report)
    if selected is None:
        detail = '; '.join(Path(e['path']).name + ': ' +
            (str(e.get('missing_required_fields')) if e.get('missing_required_fields') else e.get('reason', 'incompatible'))
            for e in entries)
        raise ValueError(f'{source}: no compatible original source-schema table found. '
            f'Use --{source}-file PATH to a workbook containing {SCHEMAS[source]}. '
            'Reduced engineered tables cannot recreate missing text or parent_sha256. '
            'For the exact uploaded inputs, extract the complete WinMal2026_v20_Ready_To_Run.zip '
            'and run its run_winmal_v20.sh launcher. '
            f'Candidates: {detail or "none found"}')
    if selection_report:
        print(f'[INPUT] {source}: {selected}', flush=True)
        for entry in entries:
            if not entry['compatible']:
                print(f'[INPUT-SKIP] {Path(entry["path"]).name}: ' +
                      str(entry.get('missing_required_fields') or entry.get('reason')), flush=True)
    return selected


def normalize_sha(value):
    if pd.isna(value):
        return ''
    value = str(value).strip().lower()
    return value if re.fullmatch(r'[0-9a-f]{64}', value) else ''


def network_raw_features(frame):
    """Versioned deterministic summary, retaining silent/all-zero rows.

    Domains: lowercase hostname, trailing dot removed, URL hostname extracted;
    unique valid non-IP names are used. TLD means final DNS label, not eTLD+1.
    Service matches use exact-domain or dot-boundary suffix matching.
    IP counts include valid IPv4/IPv6 only; public means ipaddress.is_global.
    Missing/nonpositive connection counts produce a zero domains/connection ratio.
    This is a new feature-definition version, not a reproduction of old scores.
    """
    def tokens(value):
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        if pd.isna(value):
            return []
        return re.findall(r'[^\s,;|\[\]{}()\'\"]+', str(value))
    def domains(value):
        result = set()
        for token in tokens(value):
            try:
                host = (urlsplit(token).hostname if '://' in token else token).lower().rstrip('.')
                if ':' in host:
                    host = host.split(':', 1)[0]
                try:
                    ip_address(host)
                    continue
                except ValueError:
                    pass
                if '.' in host and re.fullmatch(r'[a-z0-9_.-]+', host):
                    result.add(host)
            except (ValueError, AttributeError):
                continue
        return sorted(result)
    def ips(value):
        result = set()
        for token in tokens(value):
            try:
                result.add(ip_address(token.strip('[]')))
            except ValueError:
                continue
        return result
    result = frame.copy()
    rows = []
    for _, row in frame.iterrows():
        ds = domains(row.get('domains', ''))
        addresses = ips(row.get('hosts', ''))
        connections = pd.to_numeric(row.get('total_network_connections', 0), errors='coerce')
        def entropy(value):
            counts = np.asarray(list(__import__('collections').Counter(value).values()), dtype=float)
            p = counts / counts.sum()
            return float(-(p * np.log2(p)).sum())
        rows.append({
            'avg_domain_len': float(np.mean([len(v) for v in ds])) if ds else 0.,
            'mean_domain_entropy': float(np.mean([entropy(v) for v in ds])) if ds else 0.,
            'n_public_service_domains': sum(any(d == s or d.endswith('.' + s) for s in PUBLIC_SERVICE_DOMAINS) for d in ds),
            'n_unique_tlds': len({v.rsplit('.', 1)[-1] for v in ds}),
            'n_public_ips': sum(v.is_global for v in addresses),
            'n_unique_ips': len(addresses),
            'domains_per_conn': len(ds) / float(connections) if pd.notna(connections) and connections > 0 else 0.,
        })
    for column in PREDICTORS['network']:
        result[column] = [r[column] for r in rows]
    return result


def component_groups(frame, source, include_vectors=False):
    """Union file/process and subject hashes in a common executable namespace.

    Transitive links survive later cohort filtering because groups are built on
    all identity-valid source rows before duplicate/support/label filtering.
    Optional identical-vector edges use predictors only and never target labels.
    """
    roots = list(range(len(frame)))
    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    def union(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            roots[max(ra, rb)] = min(ra, rb)
    owner = {}
    for pos, (_, row) in enumerate(frame.iterrows()):
        for column in dict.fromkeys([SUBJECT_COLUMN[source], OBJECT_COLUMN[source]]):
            value = normalize_sha(row[column])
            if value:
                if value in owner:
                    union(pos, owner[value])
                else:
                    owner[value] = pos
    if include_vectors:
        # Preserve all existing identity components, including links through rows
        # not present in this cohort, before adding vector edges.
        if '_component' in frame:
            for indices in frame.groupby('_component', sort=False).indices.values():
                for pos in indices[1:]:
                    union(int(indices[0]), int(pos))
        for indices in frame.groupby('_vector', sort=False).indices.values():
            for pos in indices[1:]:
                union(int(indices[0]), int(pos))
    members = {}
    for pos in range(len(frame)):
        members.setdefault(root(pos), []).append(pos)
    labels = {}
    for positions in members.values():
        identities = sorted({normalize_sha(frame.iloc[i][c]) for i in positions
                             for c in [SUBJECT_COLUMN[source], OBJECT_COLUMN[source]]})
        key = hashlib.sha256('|'.join(identities).encode()).hexdigest()[:24]
        for pos in positions:
            labels[pos] = 'component_' + key
    return np.asarray([labels[i] for i in range(len(frame))])


def feature_fingerprints(frame, source):
    values = frame[PREDICTORS[source]].copy()
    for column in values:
        if column in NUMERIC[source]:
            values[column] = pd.to_numeric(values[column], errors='coerce')
        else:
            values[column] = values[column].fillna('').astype(str)
    return np.asarray([hashlib.sha256(json.dumps(
        [None if pd.isna(v) else v for v in row], default=str,
        separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        for row in values.itertuples(index=False, name=None)])


def clean_source(path, source, audit_dir):
    """Return identity-valid, subject-label-consistent rows and complete audits."""
    audit_dir = ensure_dir(audit_dir)
    frame = canonical_source_columns(read_table(path))
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError('Duplicate column names are not supported')
    raw_count = len(frame)
    frame['_row_id'] = np.arange(len(frame), dtype=int)  # zero-based source row position
    feature_definition = 'provided source-schema fields; upstream derivation must be documented'
    if source == 'network' and {'domains', 'hosts'}.intersection(frame.columns):
        frame = network_raw_features(frame)
        feature_definition = 'network_raw_summary_v17 (definitions and service suffixes in source)'
    missing = sorted(set(SCHEMAS[source]) - set(frame.columns))
    if missing:
        raise ValueError(f'{source}: missing required source fields {missing}; identifiers cannot be reconstructed')
    frame, normalization_notes = normalize_source_nominals(frame, source)
    write_json(audit_dir / 'source_value_normalization.json', normalization_notes)
    if normalization_notes:
        print(f'[INPUT-NORMALIZATION] {source}: {normalization_notes}', flush=True)
    frame['target'] = canonical_target(frame['target'])
    missing_target = frame['target'].isna() | frame['target'].isin(['', 'unknown', 'none', 'nan', 'na', 'null'])
    identities_ok = np.ones(len(frame), dtype=bool)
    identity_quality = {}
    for column in dict.fromkeys([SUBJECT_COLUMN[source], OBJECT_COLUMN[source]]):
        supplied_values = frame[column].copy()
        frame[column] = frame[column].map(normalize_sha)
        invalid = frame[column].eq('')
        identity_quality[column] = {
            'invalid_or_missing_rows': int(invalid.sum()),
            'invalid_numeric_values': int((invalid & pd.to_numeric(supplied_values, errors='coerce').notna()).sum()),
        }
        identities_ok &= frame[column].ne('').to_numpy()
    if not identities_ok.all():
        print(f'[IDENTITY-AUDIT] {source}: excluding {int((~identities_ok).sum())} rows '
              f'without required valid SHA-256 identities; details={identity_quality}', flush=True)
    reasons = np.where(~identities_ok, 'invalid_or_missing_sha256',
                       np.where(missing_target, 'missing_target', ''))
    excluded = frame.loc[reasons != '', ['_row_id', 'target']].copy()
    excluded['reason'] = reasons[reasons != '']
    # Identity components are computed BEFORE missing-target/conflict/support
    # exclusions, retaining links through excluded observations when hashes exist.
    valid_identity = frame.loc[identities_ok].reset_index(drop=True).copy()
    valid_identity['_subject'] = valid_identity[SUBJECT_COLUMN[source]]
    valid_identity['_component'] = component_groups(valid_identity, source)
    frame = valid_identity.loc[~missing_target.loc[valid_identity['_row_id']].to_numpy()].copy()
    subject_labels = frame.groupby('_subject')['target'].nunique()
    conflicts = set(subject_labels[subject_labels > 1].index)
    conflict_rows = frame[frame['_subject'].isin(conflicts)]
    if len(conflict_rows):
        add = conflict_rows[['_row_id', 'target']].copy()
        add['reason'] = 'conflicting_originating_subject_family'
        excluded = pd.concat([excluded, add], ignore_index=True)
    frame = frame.loc[~frame['_subject'].isin(conflicts)].copy()
    duplicate_mask = frame.duplicated(subset=SCHEMAS[source], keep='first')
    if duplicate_mask.any():
        add = frame.loc[duplicate_mask, ['_row_id', 'target']].copy()
        add['reason'] = 'exact_duplicate_source_record'
        excluded = pd.concat([excluded, add], ignore_index=True)
    frame = frame.loc[~duplicate_mask].reset_index(drop=True)
    for column in NUMERIC[source]:
        if source == 'memory' and column == 'total_processes':
            frame[column] = frame[column].map(lambda v: np.nan if pd.isna(v) else parse_process_count(v))
        frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for column in set(PREDICTORS[source]) - set(NUMERIC[source]):
        frame[column] = frame[column].fillna('').astype(str)
    frame['_vector'] = feature_fingerprints(frame, source)
    frame['_object'] = frame[OBJECT_COLUMN[source]]
    excluded.to_csv(audit_dir / 'excluded_records.csv', index=False)
    frame.groupby('_object').agg(rows=('target', 'size'), contexts=('target', 'nunique'),
                                subjects=('_subject', 'nunique')).to_csv(audit_dir / 'object_context_ambiguity.csv')
    vector_audit = frame.groupby('_vector').agg(rows=('target', 'size'),
                                               labels=('target', 'nunique'), subjects=('_subject', 'nunique'))
    vector_audit.to_csv(audit_dir / 'feature_collision_audit.csv')
    write_json(audit_dir / 'preparation.json', {
        'source': source, 'input_path': str(path), 'input_sha256': sha256_file(path),
        'input_records': raw_count, 'clean_records': len(frame),
        'exclusions': excluded['reason'].value_counts().to_dict(),
        'subjects': int(frame['_subject'].nunique()), 'components': int(frame['_component'].nunique()),
        'feature_definition': feature_definition, 'all_zero_network_rows_removed': False,
        'conflicting_vector_rows_removed': False, 'purity_or_accuracy_filter_used': False,
        'label_meaning': 'originating analysed-subject family context, inherited from input target',
        'artifact_family_independently_validated': False,
        'identity_quality_by_column': identity_quality,
        'source_value_normalization': normalization_notes,
        'upstream_preparation_warning': 'Earlier filtering or imputation cannot be undone. Use original schema-preserving rows; collection date is not malware first-seen date.',
        'largest_component_rows': int(frame['_component'].value_counts().max()) if len(frame) else 0,
        'conflicting_feature_vectors': int(vector_audit['labels'].gt(1).sum()),
    })
    return frame


def eligible_cohort(frame, min_samples, groups='_component'):
    stats_frame = frame.groupby('target').agg(rows=('target', 'size'),
                      subjects=('_subject', 'nunique'), independent_groups=(groups, 'nunique'))
    stats_frame['included'] = (stats_frame['rows'] >= min_samples) & (stats_frame['independent_groups'] >= 2)
    stats_frame['reason'] = np.select([stats_frame['rows'] < min_samples,
                                      stats_frame['independent_groups'] < 2],
                                     ['below_fixed_minimum_support', 'fewer_than_two_independent_groups'], default='included')
    kept = frame.loc[frame['target'].isin(stats_frame.index[stats_frame['included']])].reset_index(drop=True)
    return kept, stats_frame.reset_index()


class StableTextVectorizer(BaseEstimator, TransformerMixin):
    """Fold-fitted TF-IDF with a single zero column for an empty training field.

    Empty vocabulary handling never inspects held-out text or fits a fallback
    vocabulary on test data. Numeric/categorical source blocks remain separate.
    """
    def __init__(self, max_features=512, analyzer='char_wb', ngram_range=(3, 5)):
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
    def fit(self, X, y=None):
        self.vectorizer_ = CompatibleTfidfVectorizer(max_features=self.max_features,
            analyzer=self.analyzer, ngram_range=self.ngram_range, sublinear_tf=True,
            dtype=np.float32)
        try:
            self.vectorizer_.fit(pd.Series(X).fillna('').astype(str))
            self.empty_ = False
        except ValueError as error:
            if 'empty vocabulary' not in str(error).lower():
                raise
            self.empty_ = True
        return self
    def transform(self, X):
        if self.empty_:
            return csr_matrix((len(X), 1), dtype=np.float32)
        return self.vectorizer_.transform(pd.Series(X).fillna('').astype(str))
    def get_feature_names_out(self, input_features=None):
        return np.asarray(['empty_training_field']) if self.empty_ else self.vectorizer_.get_feature_names_out()


class DenseWithinBudget(BaseEstimator, TransformerMixin):
    def __init__(self, max_mb=512, dtype='float32'):
        self.max_mb = max_mb
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        output_dtype = np.dtype(self.dtype)
        if output_dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Dense estimator dtype must be float32 or float64')
        amount = int(X.shape[0]) * int(X.shape[1]) * output_dtype.itemsize
        if amount > self.max_mb * 1024**2:
            raise MemoryError(f'Dense estimator input needs {amount / 1024**2:.1f} MiB; budget={self.max_mb} MiB. Increase --dense-budget-mb or use sparse-compatible models.')
        return np.asarray(X.toarray() if hasattr(X, 'toarray') else X, dtype=output_dtype)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def source_preprocessor(source, fields, model_name, args):
    transformers = []
    for field in fields:
        if field in NUMERIC[source]:
            numeric = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
            if source == 'memory' and model_name == 'Naive Bayes':
                # MultinomialNB needs nonnegative numeric counts as well as
                # nonnegative text weights; clipping uses TRAIN-fitted limits.
                numeric.append(('scale', MinMaxScaler(clip=True)))
            elif model_name in {'Logistic Regression', 'SVM (RBF)', 'KNN', 'Naive Bayes'}:
                numeric.append(('scale', StandardScaler()))
            transformers.append(('field_' + field, Pipeline(numeric), [field]))
        elif field in CATEGORICAL[source]:
            transformers.append(('field_' + field, OneHotEncoder(
                handle_unknown='ignore', max_categories=512, dtype=np.float32), [field]))
        else:
            transformers.append(('field_' + field, StableTextVectorizer(max_features=args.text_max_features), field))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)


def available_models(source, args, seed):
    global RANDOM_STATE
    RANDOM_STATE = seed
    factories = {'disk': build_disk_models, 'memory': build_memory_models, 'network': build_network_models}
    models = factories[source](args.gpu_enabled)
    models['Dummy Majority'] = DummyClassifier(strategy='most_frequent')
    if args.models != ['all']:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            raise ValueError(f'{source}: unavailable models {unknown}; install optional packages or select available models')
        models = {k: v for k, v in models.items() if k in args.models or k == 'Dummy Majority'}
    return models


def make_pipeline(source, fields, name, model, args, seed):
    if set(fields).intersection(IDENTIFIER_COLUMNS):
        raise ValueError('Identity/target metadata cannot enter a predictor matrix')
    estimator = safe_clone(model)
    params = estimator.get_params(deep=False)
    updates = {key: seed for key in ['random_state', 'random_seed'] if key in params}
    if updates:
        estimator.set_params(**updates)
    steps = [('features', source_preprocessor(source, fields, name, args))]
    if name in {'Gradient Boosting', 'Naive Bayes', 'CatBoost'}:
        # GaussianNB's log likelihood/probability normalization loses precision
        # with high-dimensional float32 inputs. Fit AND predict in float64.
        dense_dtype = 'float64' if isinstance(estimator, GaussianNB) else 'float32'
        steps.append(('dense', DenseWithinBudget(args.dense_budget_mb, dtype=dense_dtype)))
    # No early stopping/eval_set on outer test folds; all settings are fixed.
    steps.append(('classifier', estimator))
    return Pipeline(steps)


def normalize_class_predictions(values, n_samples, class_axis, model_name='classifier'):
    """Accept standard (n,) and CatBoost (n,1) class outputs; reject invalid
    multi-output shapes or nonintegral/out-of-axis labels before OOF assignment.
    """
    values = np.asarray(values)
    if values.ndim == 0 and n_samples == 1:
        values = values.reshape(1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != n_samples:
        raise ValueError(f'{model_name}.predict returned {values.shape}; expected ({n_samples},) or ({n_samples}, 1) class labels')
    numeric = values.astype(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f'{model_name}.predict returned invalid/nonintegral encoded class labels')
    labels = numeric.astype(np.int64)
    if not np.isin(labels, np.asarray(class_axis)).all():
        raise ValueError(f'{model_name}.predict returned labels outside the fitted class axis')
    return labels


def checked_model_probabilities(raw_values, model_classes, all_classes, n_samples,
                                context, output_dir):
    """Align class probabilities and audit them without repairing invalid values.

    Thresholds match the previous probability-mass check. No rows are dropped,
    no NaNs are replaced, and no uniform distributions or renormalization are
    substituted. A failure preserves the model/fold and raw numeric evidence.
    """
    raw = np.asarray(raw_values)
    model_classes = np.asarray(model_classes)
    all_classes = np.asarray(all_classes)
    diagnostic = dict(context)
    diagnostic.update(raw_probability_dtype=str(raw.dtype), raw_shape=list(raw.shape),
                      expected_rows=int(n_samples), expected_columns=len(all_classes),
                      fitted_class_axis=model_classes.tolist(), expected_class_axis=all_classes.tolist(),
                      atol=1e-5, rtol=1e-5, repaired=False)
    failure = None
    aligned = None
    try:
        if raw.ndim != 2 or raw.shape != (n_samples, len(model_classes)):
            raise ValueError('probability dimensions do not match rows/fitted classes')
        if model_classes.ndim != 1 or len(np.unique(model_classes)) != len(model_classes):
            raise ValueError('fitted probability class axis is not one-dimensional and unique')
        if not np.isin(model_classes, all_classes).all():
            raise ValueError('fitted probability class axis contains an unexpected class')
        if raw.dtype.kind not in 'biuf':
            raise ValueError('probabilities are not a real numeric matrix')
        numeric = np.asarray(raw, dtype=np.float64)
        aligned = align_proba_to_all_classes(numeric, model_classes, all_classes)
        mass = aligned.sum(axis=1)
        finite = np.isfinite(aligned)
        finite_values = aligned[finite]
        finite_mass = mass[np.isfinite(mass)]
        diagnostic.update(
            nonfinite_entries=int((~finite).sum()),
            negative_entries=int((aligned < 0).sum()),
            above_one_entries=int((aligned > 1).sum()),
            probability_min=float(finite_values.min()) if finite_values.size else None,
            probability_max=float(finite_values.max()) if finite_values.size else None,
            row_sum_min=float(finite_mass.min()) if finite_mass.size else None,
            row_sum_max=float(finite_mass.max()) if finite_mass.size else None,
            maximum_absolute_row_sum_error=float(np.abs(finite_mass - 1.).max()) if finite_mass.size else None)
        if not finite.all():
            raise ValueError('probabilities contain NaN or infinity')
        if (aligned < 0).any() or (aligned > 1).any():
            raise ValueError('probabilities are outside [0, 1]')
        if not np.allclose(mass, 1., atol=1e-5, rtol=1e-5):
            raise ValueError('probability rows do not sum to one within the existing tolerance')
    except (ValueError, TypeError, OverflowError) as error:
        failure = str(error)
    if failure is not None:
        diagnostic.update(status='failed', reason=failure)
        output_dir = ensure_dir(output_dir)
        path = output_dir / 'invalid_probabilities.json'
        write_json(path, diagnostic)
        if raw.dtype.kind in 'biuf':
            np.savez_compressed(output_dir / 'invalid_probabilities.npz', probabilities=raw,
                                model_classes=model_classes, all_classes=all_classes)
        where = ', '.join(f'{key}={value}' for key, value in context.items())
        raise RuntimeError(f'Invalid predicted probabilities ({where}): {failure}; '
            f'raw_dtype={raw.dtype}, row_sum_min={diagnostic.get("row_sum_min")}, '
            f'row_sum_max={diagnostic.get("row_sum_max")}. Details: {path}')
    diagnostic['status'] = 'passed'
    return aligned, diagnostic


def metric_values(y_true, y_pred, prob=None, classes=None, weights=None):
    labels = np.asarray(classes if classes is not None else sorted(set(y_true) | set(y_pred)))
    result = {
        'accuracy': float(accuracy_score(y_true, y_pred, sample_weight=weights)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0, sample_weight=weights)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0, sample_weight=weights)),
        'macro_precision': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0, sample_weight=weights)),
        'macro_recall_fixed_axis': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0, sample_weight=weights)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred, sample_weight=weights)),
        'auc_ovr_macro': None,
    }
    if prob is not None and weights is None:
        try:
            if len(labels) == 2:
                result['auc_ovr_macro'] = float(roc_auc_score(y_true, prob[:, 1], labels=labels))
            elif len(labels) > 2:
                result['auc_ovr_macro'] = float(roc_auc_score(y_true, prob, labels=labels, multi_class='ovr', average='macro'))
        except ValueError:
            pass  # absent test classes make that AUC undefined, not zero
    return result


def group_bootstrap(y_true, y_pred, groups, classes, draws, seed):
    """Conditional percentile intervals for FIXED OOF predictions, resampling
    identity components. These are not confidence intervals for retraining or
    independent-fold/independent-seed significance tests.
    """
    if draws <= 0:
        return {}
    labels = {v: i for i, v in enumerate(classes)}
    ids, inverse = np.unique(groups, return_inverse=True)
    k = len(classes)
    matrices = np.zeros((len(ids), k, k), dtype=np.int64)
    np.add.at(matrices, (inverse, [labels[v] for v in y_true], [labels[v] for v in y_pred]), 1)
    rng = np.random.default_rng(seed)
    acc, macro = [], []
    for _ in range(draws):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        cm = np.tensordot(counts, matrices, axes=(0, 0))
        diagonal = np.diag(cm)
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        acc.append(diagonal.sum() / cm.sum())
        macro.append(np.divide(2 * diagonal, denominator, out=np.zeros(k, dtype=float), where=denominator > 0).mean())
    return {'conditional_component_bootstrap': {
        'draws': draws, 'components': len(ids), 'fixed_family_axis': list(classes),
        'accuracy_percentile_95': np.quantile(acc, [.025, .975]).tolist(),
        'macro_f1_percentile_95': np.quantile(macro, [.025, .975]).tolist(),
        'scope': 'uncertainty conditional on fixed OOF predictions; excludes model-retraining uncertainty and is not a model-comparison significance test'}}


def identity_overlap(frame, train, test):
    tr, te = frame.iloc[train], frame.iloc[test]
    return {
        'subject_overlap': len(set(tr['_subject']) & set(te['_subject'])),
        'object_overlap': len(set(tr['_object']) & set(te['_object'])),
        'any_executable_overlap': len((set(tr['_subject']) | set(tr['_object'])) & (set(te['_subject']) | set(te['_object']))),
        'component_overlap': len(set(tr['_component']) & set(te['_component'])),
        'exact_feature_vector_overlap': len(set(tr['_vector']) & set(te['_vector'])),
    }


def make_splits(frame, protocol, requested, seed):
    y = LabelEncoder().fit_transform(frame['target'])
    if len(set(y)) < 2:
        raise ValueError('At least two eligible families are required')
    group_column = {'component': '_component', 'subject': '_subject', 'object': '_object', 'vector_component': '_vector_component'}.get(protocol)
    if group_column:
        limit = int(frame.groupby('target')[group_column].nunique().min())
        groups = frame[group_column].to_numpy()
    else:
        limit = int(frame['target'].value_counts().min())
        groups = None
    all_classes = set(y)
    # Reduction is based only on fold feasibility, never model scores.
    # SGKF can fail to place mixed-label components well; try deterministic
    # shuffles and reduce folds, recording the exact accepted partition.
    for folds in range(min(requested, limit), 1, -1):
        for attempt in range(8):
            split_seed = seed + attempt * 1009
            splitter = (StratifiedKFold(folds, shuffle=True, random_state=split_seed) if groups is None
                        else StratifiedGroupKFold(folds, shuffle=True, random_state=split_seed))
            splits = list(splitter.split(np.zeros(len(y)), y, groups))
            if all(set(y[train]) == all_classes for train, test in splits):
                for train, test in splits:
                    if group_column and set(frame.iloc[train][group_column]) & set(frame.iloc[test][group_column]):
                        raise AssertionError('Group split overlap')
                    overlap = identity_overlap(frame, train, test)
                    if protocol in {'component', 'vector_component'} and overlap['any_executable_overlap']:
                        raise AssertionError('Executable identity crossed primary folds')
                return splits, {'requested_folds': requested, 'effective_folds': folds,
                                'split_seed': split_seed, 'feasibility_attempt': attempt,
                                'protocol': protocol, 'group_column': group_column,
                                'all_training_folds_contain_all_families': True}
    raise ValueError('No feasible >=2-fold partition with every family in each training fold; inspect components, collect more independent subjects, or revise the prespecified population')


def ablation_fields(source):
    if source == 'disk':
        return {'no_parent_fields': ['size', 'type', 'file_name', 'file_path'],
                'no_names_paths': ['size', 'type', 'parent_size', 'parent_type', 'parent_imphash'],
                'no_parent_or_names_paths': ['size', 'type'],
                'no_parent_imphash': [v for v in PREDICTORS[source] if v != 'parent_imphash']}
    if source == 'memory':
        return {'no_paths_commands': ['created_files', 'registry', 'total_processes', 'modules_count'],
                'no_file_registry_text': ['normalized_path', 'command_line', 'total_processes', 'modules_count'],
                'counts_only': ['total_processes', 'modules_count']}
    return {'no_domain_shape': [v for v in PREDICTORS[source] if v not in {'avg_domain_len', 'mean_domain_entropy'}],
            'no_ip_counts': [v for v in PREDICTORS[source] if v not in {'n_public_ips', 'n_unique_ips'}]}


def plot_confusion_numeric(cm, names, path, title):
    fig, ax = plt.subplots(figsize=(max(7, len(names) * .35), max(6, len(names) * .32)))
    image = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names,
           yticklabels=names, xlabel='Predicted family context', ylabel='Observed family context', title=title)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def export_predictions(frame, classes, predictions, probabilities, fold_ids, path):
    result = frame[['_row_id', '_subject', '_object', '_component', 'target']].copy()
    result.columns = ['source_row_id', 'subject_sha256', 'object_sha256', 'component', 'true_family']
    result['fold'] = fold_ids
    result['predicted_family'] = np.asarray(classes)[predictions]
    result['correct'] = result['true_family'].eq(result['predicted_family'])
    result['confidence'] = probabilities.max(axis=1)
    for index, label in enumerate(classes):
        result['probability__' + str(label)] = probabilities[:, index]
    result.to_csv(path, index=False)
    return result


def write_per_family(y, pred, classes, frame, out_path):
    report = classification_report(y, pred, labels=np.arange(len(classes)), target_names=classes,
                                   zero_division=0, output_dict=True)
    rows = []
    for i, name in enumerate(classes):
        mask = frame['target'].eq(name)
        rows.append({'family': name, **report[name],
                     'unique_subjects': int(frame.loc[mask, '_subject'].nunique()),
                     'independent_components': int(frame.loc[mask, '_component'].nunique())})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def summarize_split_audit(frame, splits):
    return [{'fold': i, 'train_rows': len(train), 'test_rows': len(test),
             'train_subjects': int(frame.iloc[train]['_subject'].nunique()),
             'test_subjects': int(frame.iloc[test]['_subject'].nunique()),
             **identity_overlap(frame, train, test)}
            for i, (train, test) in enumerate(splits, 1)]


def run_cv(frame, source, variant, protocol, fields, seed, args, model_names=None,
           fixed_splits=None, fixed_meta=None):
    global RANDOM_STATE
    RANDOM_STATE = seed
    out = ensure_dir(Path(args.output_dir) / source / variant / f'seed_{seed}')
    plots = ensure_dir(out / 'plots')
    if fixed_splits is None:
        splits, split_meta = make_splits(frame, protocol, args.folds, seed)
    else:
        splits, split_meta = fixed_splits, dict(fixed_meta)
    classes = sorted(frame['target'].unique().tolist())
    y = pd.Categorical(frame['target'], categories=classes).codes
    X = frame.loc[:, fields].copy()
    checks = summarize_split_audit(frame, splits)
    pd.DataFrame(checks).to_csv(out / 'split_overlap_audit.csv', index=False)
    fold_ids = np.full(len(frame), -1, dtype=int)
    train_test = []
    for fold, (train, test) in enumerate(splits, 1):
        fold_ids[test] = fold
        train_test.append({'fold': fold, 'train_source_row_ids': frame.iloc[train]['_row_id'].tolist(),
                           'test_source_row_ids': frame.iloc[test]['_row_id'].tolist()})
    assert np.all(fold_ids > 0)
    split_fingerprint = hashlib.sha256(json.dumps(train_test, sort_keys=True).encode()).hexdigest()
    write_json(out / 'folds.json', {'split': split_meta, 'fingerprint': split_fingerprint, 'partitions': train_test})
    frame[['_row_id', '_subject', '_object', '_component', '_vector', 'target']].assign(fold=fold_ids).to_csv(out / 'cohort_and_test_folds.csv', index=False)
    models = available_models(source, args, seed)
    if model_names is not None:
        models = {name: model for name, model in models.items() if name in model_names or name == 'Dummy Majority'}
    if not models:
        raise ValueError('No estimators selected')
    outputs = []
    for name, model in models.items():
        slug = sanitize_filename(name)
        model_out = ensure_dir(out / slug)
        signature = hashlib.sha256(json.dumps({
            'code': args.code_sha256, 'source_input': args.input_checksums[source], 'split': split_fingerprint,
            'fields': fields, 'seed': seed, 'name': name, 'params': model.get_params(deep=False),
            'text_max_features': args.text_max_features, 'dense_budget_mb': args.dense_budget_mb,
            'bootstrap': args.bootstrap, 'xai_enabled': not args.skip_xai,
            'xai_seed': args.xai_seed, 'background': args.xai_background_size,
            'explain': args.xai_explain_size, 'plots': not args.skip_plots},
            sort_keys=True, default=str).encode()).hexdigest()
        cached = model_out / 'completed.json'
        if args.resume and cached.exists():
            payload = json.loads(cached.read_text())
            if payload['signature'] != signature:
                raise ValueError(f'Resume signature changed: {model_out}; use a new output directory')
            print(f'[RESUME] {source}/{variant}/{seed}/{name}', flush=True)
            xai_status = model_out / 'xai_status.json'
            if xai_status.exists() and json.loads(xai_status.read_text()).get('status') == 'failed':
                args.nonfatal_failures.append({'source': source, 'stage': 'cached_xai',
                    'error': 'Previous XAI failure remains unresolved; use a new output directory after fixing it.'})
            outputs.append(payload['summary'])
            continue
        print(f'[FIT] {source} {variant} seed={seed} model={name} rows={len(frame)} families={len(classes)} folds={len(splits)}', flush=True)
        pipeline = make_pipeline(source, fields, name, model, args, seed)
        write_json(model_out / 'model_specification.json', {
            'model': name, 'parameters': pipeline.named_steps['classifier'].get_params(deep=False),
            'source_fields': fields, 'preprocessing': repr(pipeline.named_steps['features']),
            'dense_budget_mb': args.dense_budget_mb,
            'dense_input_dtype': pipeline.named_steps['dense'].dtype if 'dense' in pipeline.named_steps else None,
            'selection': 'fixed settings, no outer-test tuning',
            'class_axis': classes, 'seed': seed, 'signature': signature})
        prob = np.full((len(frame), len(classes)), np.nan)
        predicted = np.full(len(frame), -1, dtype=int)
        fold_metrics, warning_rows, probability_audits = [], [], []
        xai = None
        if variant == 'primary' and name == 'Random Forest' and seed == args.xai_seed and not args.skip_xai:
            try:
                xai = RandomForestSourceXAI(X, y, classes, source, ensure_dir(model_out / 'xai'),
                    AnalysisOptions(run_xai=True, run_tsne=False,
                       xai_background_size=args.xai_background_size, xai_explain_size=args.xai_explain_size),
                    sample_ids=frame['_object'], display_values=X, groups=frame['_component'])
            except Exception as error:
                args.nonfatal_failures.append({'source': source, 'stage': 'xai_initialization', 'error': str(error)})
                write_json(model_out / 'xai_status.json', {'status': 'failed', 'error': str(error)})
        started = time.perf_counter()
        for fold, (train, test) in enumerate(splits, 1):
            fitted = clone(pipeline)
            if name == 'KNN':
                fitted.set_params(classifier__n_neighbors=min(fitted.named_steps['classifier'].n_neighbors, len(train)))
            tick = time.perf_counter()
            write_json(Path(args.output_dir) / 'runtime_progress.json', {
                'source': source, 'variant': variant, 'model': name,
                'seed': seed, 'fold': fold, 'stage': 'fitting'})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fitted.fit(X.iloc[train], y[train])
            for warning in caught:
                warning_rows.append({'fold': fold, 'category': warning.category.__name__, 'message': str(warning.message)})
            fit_seconds = time.perf_counter() - tick
            pd.DataFrame(warning_rows, columns=['fold', 'category', 'message']).to_csv(model_out / 'fit_warnings.csv', index=False)
            tick = time.perf_counter()
            write_json(Path(args.output_dir) / 'runtime_progress.json', {
                'source': source, 'variant': variant, 'model': name,
                'seed': seed, 'fold': fold, 'stage': 'predicting_and_validating_probabilities'})
            raw_prob = fitted.predict_proba(X.iloc[test])
            aligned, probability_audit = checked_model_probabilities(
                raw_prob, fitted.classes_, np.arange(len(classes)), len(test),
                {'source': source, 'variant': variant, 'seed': seed, 'model': name, 'fold': fold}, model_out)
            probability_audits.append(probability_audit)
            pd.DataFrame(probability_audits).to_csv(model_out / 'probability_validation.csv', index=False)
            # Use estimator.predict, preserving its declared tie rule.
            fold_pred = normalize_class_predictions(fitted.predict(X.iloc[test]),
                len(test), np.arange(len(classes)), name)
            prediction_seconds = time.perf_counter() - tick
            prob[test], predicted[test] = aligned, fold_pred
            row = {'fold': fold, 'test_rows': len(test), 'fit_seconds': fit_seconds,
                   'prediction_seconds': prediction_seconds, **metric_values(y[test], fold_pred, aligned, np.arange(len(classes)))}
            fold_metrics.append(row)
            pd.DataFrame(fold_metrics).to_csv(model_out / 'fold_metrics.csv', index=False)
            if xai is not None:
                try:
                    xai.record_fold(fitted, X.iloc[test], train, test, fold_pred, aligned, training_values=X.iloc[train])
                except Exception as error:
                    args.nonfatal_failures.append({'source': source, 'stage': 'xai', 'error': str(error)})
                    write_json(model_out / 'xai_status.json', {'status': 'failed', 'error': str(error), 'fold': fold})
                    xai = None
        if not np.isfinite(prob).all() or np.any(predicted < 0):
            raise AssertionError('Incomplete out-of-fold predictions')
        if xai is not None:
            try:
                xai.finalize()
                write_json(model_out / 'xai_status.json', {'status': 'completed', 'model_scope': 'prespecified RF, not necessarily leading classifier'})
            except Exception as error:
                args.nonfatal_failures.append({'source': source, 'stage': 'xai_finalization', 'error': str(error)})
                write_json(model_out / 'xai_status.json', {'status': 'failed', 'error': str(error)})
        pd.DataFrame(warning_rows, columns=['fold', 'category', 'message']).to_csv(model_out / 'fit_warnings.csv', index=False)
        oof = export_predictions(frame, classes, predicted, prob, fold_ids, model_out / 'oof_predictions.csv')
        np.savez_compressed(model_out / 'oof_probabilities.npz', probabilities=prob, predictions=predicted,
                            y=y, source_row_ids=frame['_row_id'].to_numpy(), classes=np.asarray(classes), fold=fold_ids)
        write_per_family(y, predicted, classes, frame, model_out / 'per_family_metrics.csv')
        cm = confusion_matrix(y, predicted, labels=np.arange(len(classes)))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(model_out / 'confusion_counts.csv')
        errors = oof.loc[~oof['correct']].groupby(['true_family', 'predicted_family']).size().reset_index(name='errors').sort_values('errors', ascending=False)
        errors.to_csv(model_out / 'directed_errors.csv', index=False)
        fold_df = pd.DataFrame(fold_metrics)
        subject_weight = (1 / frame['_subject'].map(frame['_subject'].value_counts())).to_numpy()
        summary = {'source': source, 'variant': variant, 'protocol': protocol, 'seed': seed, 'model': name,
                   'rows': len(frame), 'families': len(classes), 'subjects': int(frame['_subject'].nunique()),
                   'components': int(frame['_component'].nunique()), 'folds': len(splits),
                   'source_fields': '|'.join(fields), 'split_fingerprint': split_fingerprint,
                   'validation_scope': 'primary' if protocol in {'component', 'vector_component'} else 'overlap_diagnostic_only',
                   'elapsed_seconds': time.perf_counter() - started,
                   'convergence_warning_count': sum(r['category'] == 'ConvergenceWarning' for r in warning_rows),
                   **metric_values(y, predicted, prob, np.arange(len(classes)))}
        summary['cohort_fingerprint'] = hashlib.sha256(json.dumps({
            'input': args.input_checksums[source],
            'rows': frame['_row_id'].tolist()}, sort_keys=True).encode()).hexdigest()
        for key in ['subject_overlap', 'object_overlap', 'any_executable_overlap',
                    'component_overlap', 'exact_feature_vector_overlap']:
            summary['folds_with_' + key] = sum(row[key] > 0 for row in checks)
            summary['maximum_' + key] = max(row[key] for row in checks)
        subject_metrics = metric_values(y, predicted, classes=np.arange(len(classes)), weights=subject_weight)
        summary.update({'equal_subject_weight_' + key: val for key, val in subject_metrics.items() if val is not None})
        for key in ['accuracy', 'macro_f1', 'weighted_f1', 'balanced_accuracy']:
            summary[key + '_fold_mean'] = float(fold_df[key].mean())
            summary[key + '_fold_sd_descriptive'] = float(fold_df[key].std(ddof=1))
        uncertainty = group_bootstrap(frame['target'].to_numpy(), oof['predicted_family'].to_numpy(),
                                     frame['_component'].to_numpy(), classes,
                                     args.bootstrap if variant == 'primary' else 0, seed)
        write_json(model_out / 'uncertainty.json', uncertainty)
        write_json(model_out / 'pooled_metrics.json', summary)
        if not args.skip_plots:
            plot_confusion_numeric(cm, classes, plots / f'{slug}_confusion.png', f'{source.title()} {name}: {variant}')
            plot_binary_or_multiclass_roc(y, prob, name, classes, plots / f'{slug}_roc.png')
            plot_binary_or_multiclass_pr(y, prob, name, classes, plots / f'{slug}_precision_recall.png')
        write_json(cached, {'signature': signature, 'summary': summary})
        outputs.append(summary)
        pd.DataFrame(outputs).to_csv(out / 'model_summary.csv', index=False)
    return outputs, splits, split_meta


def make_review_template(frame, source, out_dir, args):
    candidates = frame.sample(frac=1, random_state=42).drop_duplicates(['target', '_subject'])
    sample = candidates.groupby('target', group_keys=False).head(args.review_per_family).sort_values(['target', '_row_id'])
    key = sample[['_row_id', '_subject', '_object', 'target']].copy()
    key.columns = ['source_row_id', 'subject_sha256', 'object_sha256', 'input_family_context']
    key.insert(0, 'source', source)
    key.to_csv(out_dir / 'review_private_label_key.csv', index=False)
    blinded = key.drop(columns='input_family_context')
    for column in ['review_scope', 'reviewed_family', 'evidence_reference', 'reviewer_notes']:
        blinded[column] = ''
    blinded = blinded.merge(sample[['_row_id'] + PREDICTORS[source]], left_on='source_row_id', right_on='_row_id').drop(columns='_row_id')
    blinded.to_csv(out_dir / 'review_blinded_template.csv', index=False)
    write_json(out_dir / 'review_status.json', {'status': 'pending independent review', 'sample_rows': len(sample),
        'sampling': 'seed-42 stratified sample, at most one row per subject per family',
        'review_scopes': ['subject_context', 'artefact_family'],
        'warning': 'The input label key is separate. Reviewers should not use the input family as independent evidence. Raw source text may itself reveal names; document blinding limits.'})


def validate_review_file(frame, source, path, out_dir):
    reviews = read_table(path)
    required = {'source', 'source_row_id', 'review_scope', 'reviewed_family', 'evidence_reference'}
    if not required.issubset(reviews):
        raise ValueError(f'Label review CSV requires {sorted(required)}')
    reviews = reviews.loc[reviews['source'].eq(source)].copy()
    reviews['reviewed_family'] = canonical_target(reviews['reviewed_family'])
    reviews = reviews.loc[reviews['reviewed_family'].notna() & reviews['reviewed_family'].ne('') &
                          reviews['evidence_reference'].notna() & reviews['evidence_reference'].astype(str).str.strip().ne('')]
    if reviews.duplicated(['source_row_id', 'review_scope']).any():
        raise ValueError('Duplicate completed label review for a source row/scope')
    if not reviews['review_scope'].isin(['subject_context', 'artefact_family']).all():
        raise ValueError('review_scope must be subject_context or artefact_family')
    checked = reviews.merge(frame[['_row_id', 'target', '_subject', '_object']], left_on='source_row_id', right_on='_row_id', validate='many_to_one', how='left', indicator=True)
    if checked['_merge'].ne('both').any():
        raise ValueError('Review references excluded/nonexistent source rows; inspect excluded_records.csv')
    checked['agrees_with_input_context'] = checked['reviewed_family'].eq(checked['target'])
    checked.to_csv(out_dir / 'completed_independent_reviews.csv', index=False)
    for scope, group in checked.groupby('review_scope'):
        group.groupby('target').agg(reviewed_rows=('target', 'size'),
            agreeing_rows=('agrees_with_input_context', 'sum')).assign(
            scope=scope).to_csv(out_dir / f'review_agreement_{scope}.csv')
    write_json(out_dir / 'independent_review_status.json', {
        'completed_rows_by_scope': checked['review_scope'].value_counts().to_dict(),
        'subject_context_disagreements': int((checked['review_scope'].eq('subject_context') & ~checked['agrees_with_input_context']).sum()),
        'artifact_context_differences_are_not_automatically_label_errors': True,
        'sample_estimate_scope': 'descriptive reviewed subset; no claim of independent verification of the entire dataset',
        'rows_relabelled_for_benchmark': False})


def temporal_partition(frame, source, metadata_path, cutoff):
    metadata = read_table(metadata_path)
    required = {'source', 'subject_sha256', 'timestamp', 'timestamp_kind'}
    if not required.issubset(metadata):
        raise ValueError(f'Temporal metadata requires {sorted(required)}')
    metadata = metadata.loc[metadata['source'].eq(source)].copy()
    metadata['subject_sha256'] = metadata['subject_sha256'].map(normalize_sha)
    if metadata['subject_sha256'].eq('').any() or metadata.duplicated('subject_sha256').any():
        raise ValueError('Temporal metadata needs exactly one valid row per source/subject; resolve multiple reports explicitly')
    if len(metadata) == 0:
        raise ValueError(f'No temporal metadata for {source}')
    kinds = set(metadata['timestamp_kind'].dropna().astype(str))
    if len(kinds) != 1 or not kinds.issubset({'first_seen', 'report_execution'}):
        raise ValueError('Use one timestamp_kind per source: first_seen or report_execution. Retrieval/collection dates cannot stand in for malware age.')
    if metadata['timestamp_kind'].isna().any():
        raise ValueError('Every timestamp requires an explicit timestamp_kind')
    metadata['_timestamp'] = pd.to_datetime(metadata['timestamp'], utc=True, errors='coerce')
    if metadata['_timestamp'].isna().any():
        raise ValueError('Invalid temporal timestamp values')
    cut = pd.to_datetime(cutoff, utc=True)
    timed = frame.merge(metadata[['subject_sha256', '_timestamp']], left_on='_subject', right_on='subject_sha256', how='left', validate='many_to_one')
    grouped = timed.groupby('_component')['_timestamp'].agg(['min', 'max', 'count', 'size'])
    train_groups = grouped.index[(grouped['count'] == grouped['size']) & (grouped['max'] < cut)]
    test_groups = grouped.index[(grouped['count'] == grouped['size']) & (grouped['min'] >= cut)]
    train = timed.loc[timed['_component'].isin(train_groups)].reset_index(drop=True)
    test = timed.loc[timed['_component'].isin(test_groups)].reset_index(drop=True)
    audit = {'cutoff_utc': str(cut), 'timestamp_kind': next(iter(kinds)),
             'scope': 'forward first-seen generalization' if kinds == {'first_seen'} else 'transfer between report-execution periods; does not establish malware first-seen chronology',
             'rule': 'all component timestamps before cutoff for training; all on/after for test; straddling or incomplete components excluded',
             'train_rows_before_train_support_rule': len(train), 'test_rows': len(test),
             'excluded_spanning_or_incomplete_rows': len(timed) - len(train) - len(test),
             'missing_timestamp_rows': int(timed['_timestamp'].isna().sum()),
             'cutoff_selection': 'provided before outcome inspection; no automated accuracy-based cutoff search'}
    return train, test, audit


def filter_external_identity(train, external):
    identities = set(train['_subject']) | set(train['_object'])
    overlap = external['_subject'].isin(identities) | external['_object'].isin(identities)
    # Remove entire externally connected components when any member intersects
    # training. This handles transitive connections within the external table.
    blocked = set(external.loc[overlap, '_component'])
    keep = ~external['_component'].isin(blocked)
    return external.loc[keep].reset_index(drop=True), external.loc[~keep].copy()


def run_holdout(train, test, source, variant, seed, args, evidence):
    counts = train['target'].value_counts()
    train = train.loc[train['target'].isin(counts.index[counts >= args.min_samples])].reset_index(drop=True)
    if train['target'].nunique() < 2 or len(test) == 0:
        raise ValueError(f'{variant}: requires >=2 training families meeting training-only support and a nonempty test set')
    if (set(train['_subject']) | set(train['_object'])) & (set(test['_subject']) | set(test['_object'])):
        raise AssertionError('Training/test executable identity overlap in holdout')
    out = ensure_dir(Path(args.output_dir) / source / variant / f'seed_{seed}')
    classes = sorted(train['target'].unique().tolist())
    y_train = pd.Categorical(train['target'], categories=classes).codes
    known = test['target'].isin(classes).to_numpy()
    test[['_row_id', '_subject', '_object', '_component', 'target']].assign(known_training_family=known).to_csv(out / 'test_manifest.csv', index=False)
    train[['_row_id', '_subject', '_object', '_component', 'target']].to_csv(out / 'train_manifest.csv', index=False)
    test.loc[~known].groupby('target').size().rename('test_records').to_csv(out / 'unseen_training_families.csv')
    write_json(out / 'holdout_design.json', {**evidence,
        'training_only_minimum_rows': args.min_samples, 'training_rows': len(train), 'test_rows': len(test),
        'training_families': classes, 'known_family_test_coverage': float(known.mean()),
        'test_families_not_used_to_select_training_classes': True,
        'unknown_family_rejection_implemented': False,
        'warning': 'Known-family metrics apply to the stated subset. All-test accuracy counts unseen-family cases as errors; this is not an open-set detector.'})
    models = available_models(source, args, seed)
    models = {name: model for name, model in models.items() if name in args.diagnostic_models or name == 'Dummy Majority'}
    results = []
    for name, model in models.items():
        model_out = ensure_dir(out / sanitize_filename(name))
        signature = hashlib.sha256(json.dumps({'code': args.code_sha256, 'input': args.input_checksums[source],
            'evidence': evidence, 'seed': seed, 'name': name, 'parameters': model.get_params(deep=False),
            'text_max_features': args.text_max_features,
            'train': train['_row_id'].tolist(), 'test_subjects': test['_subject'].tolist()}, sort_keys=True, default=str).encode()).hexdigest()
        completed = model_out / 'completed.json'
        if args.resume and completed.exists():
            payload = json.loads(completed.read_text())
            if payload['signature'] != signature:
                raise ValueError('Holdout resume inputs/settings changed; use a new output directory')
            results.append(payload['summary'])
            continue
        model_pipe = make_pipeline(source, PREDICTORS[source], name, model, args, seed)
        if name == 'KNN':
            model_pipe.set_params(classifier__n_neighbors=min(model_pipe.named_steps['classifier'].n_neighbors, len(train)))
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model_pipe.fit(train[PREDICTORS[source]], y_train)
        pred = normalize_class_predictions(model_pipe.predict(test[PREDICTORS[source]]),
            len(test), np.arange(len(classes)), name)
        raw_probabilities = model_pipe.predict_proba(test[PREDICTORS[source]])
        probabilities, probability_audit = checked_model_probabilities(
            raw_probabilities, model_pipe.classes_, np.arange(len(classes)), len(test),
            {'source': source, 'variant': variant, 'seed': seed, 'model': name, 'fold': 'holdout'}, model_out)
        pd.DataFrame([probability_audit]).to_csv(model_out / 'probability_validation.csv', index=False)
        pred_names = np.asarray(classes)[pred]
        table = export_predictions(test, classes, pred, probabilities, np.zeros(len(test), dtype=int), model_out / 'holdout_predictions.csv')
        table['known_training_family'] = known
        table.to_csv(model_out / 'holdout_predictions.csv', index=False)
        known_metrics = {}
        if known.any():
            y_known = pd.Categorical(test.loc[known, 'target'], categories=classes).codes
            known_metrics = metric_values(y_known, pred[known], probabilities[known], np.arange(len(classes)))
            write_per_family(y_known, pred[known], classes, test.loc[known].reset_index(drop=True), model_out / 'known_family_metrics.csv')
        summary = {'source': source, 'variant': variant, 'protocol': 'independent_holdout', 'seed': seed,
                   'model': name, 'rows': len(test), 'training_rows': len(train), 'families': len(classes),
                   'all_test_accuracy': float(np.mean(pred_names == test['target'].to_numpy())),
                   'known_family_test_coverage': float(known.mean()),
                   'elapsed_seconds': time.perf_counter() - tick,
                   **{'known_' + key: value for key, value in known_metrics.items()}}
        pd.DataFrame([{'category': w.category.__name__, 'message': str(w.message)} for w in caught]).to_csv(model_out / 'fit_warnings.csv', index=False)
        write_json(model_out / 'model_specification.json', {'source_fields': PREDICTORS[source],
                   'parameters': model_pipe.named_steps['classifier'].get_params(deep=False),
                   'preprocessing': repr(model_pipe.named_steps['features']),
                   'dense_input_dtype': model_pipe.named_steps['dense'].dtype if 'dense' in model_pipe.named_steps else None})
        write_json(completed, {'signature': signature, 'summary': summary})
        results.append(summary)
    pd.DataFrame(results).to_csv(out / 'model_summary.csv', index=False)
    return results


def export_paired_ablations(results, root):
    records = pd.DataFrame(results)
    if records.empty or 'macro_f1' not in records:
        return
    pairs = []
    for source in records['source'].unique():
        subset = records.loc[records['source'].eq(source)]
        base = subset.loc[subset['variant'].eq('primary')]
        for _, row in subset.loc[subset['variant'].str.startswith('ablation_')].iterrows():
            matching = base.loc[base['seed'].eq(row['seed']) & base['model'].eq(row['model'])]
            if len(matching) != 1:
                continue
            full = matching.iloc[0]
            if full['split_fingerprint'] != row['split_fingerprint'] or full['rows'] != row['rows']:
                raise AssertionError('Ablation comparison must have identical rows and folds')
            pairs.append({'source': source, 'seed': row['seed'], 'model': row['model'], 'ablation': row['variant'],
                          'rows': row['rows'], 'same_folds': True,
                          'full_macro_f1': full['macro_f1'], 'ablated_macro_f1': row['macro_f1'],
                          'macro_f1_full_minus_ablated': full['macro_f1'] - row['macro_f1'],
                          'accuracy_full_minus_ablated': full['accuracy'] - row['accuracy'],
                          'interpretation': 'descriptive paired OOF difference, not an independent-fold significance test'})
    pd.DataFrame(pairs).to_csv(Path(root) / 'paired_ablation_differences.csv', index=False)


def export_protocol_comparisons(results, root):
    """Describe differences on an identical cohort; do not infer causality or
    use overlap diagnostics as evidence of performance on independent subjects.
    """
    records = pd.DataFrame(results)
    if records.empty:
        return
    pairs = []
    for _, row in records.iterrows():
        variant = row['variant']
        if variant.startswith('split_diagnostic_'):
            reference_variant = 'primary'
        elif variant == 'collision_common_vector_component':
            reference_variant = 'collision_common_component'
        else:
            continue
        reference = records.loc[records['variant'].eq(reference_variant) &
            records['source'].eq(row['source']) & records['seed'].eq(row['seed']) &
            records['model'].eq(row['model'])]
        if len(reference) != 1:
            continue
        base = reference.iloc[0]
        if base.get('cohort_fingerprint') != row.get('cohort_fingerprint'):
            raise AssertionError('Protocol comparisons require identical source rows')
        pairs.append({'source': row['source'], 'seed': row['seed'], 'model': row['model'],
            'variant': variant, 'reference_variant': reference_variant, 'rows': row['rows'],
            'families': row['families'], 'same_cohort': True,
            'macro_f1_variant_minus_reference': row['macro_f1'] - base['macro_f1'],
            'accuracy_variant_minus_reference': row['accuracy'] - base['accuracy'],
            'variant_folds_with_executable_overlap': row['folds_with_any_executable_overlap'],
            'variant_folds_with_exact_vector_overlap': row['folds_with_exact_feature_vector_overlap'],
            'scope': 'descriptive protocol sensitivity; preprocessing stays fold-fitted in both protocols'})
    pd.DataFrame(pairs).to_csv(Path(root) / 'protocol_sensitivity_comparisons.csv', index=False)


def describe_environment(args):
    packages = {}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'openpyxl', 'shap', 'xgboost', 'catboost']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'script_version': SCRIPT_VERSION, 'script_sha256': args.code_sha256,
            'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'cpu_threads': CPU_THREADS, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'gpu_enabled': args.gpu_enabled, 'arguments': vars(args),
            'primary_metric': 'pooled out-of-fold macro-F1 on a fixed family axis',
            'claim_limit': 'No promise of conference acceptance. No child-family ground truth, temporal evidence, or empirical results are invented.'}


def self_test():
    """Meaningful invariant checks; no real malware dataset is required."""
    import tempfile
    from types import SimpleNamespace
    def h(number):
        return hashlib.sha256(str(number).encode()).hexdigest()
    rows = []
    for family in ['alpha', 'beta']:
        for report in range(6):
            subject = h(f'{family}_subject_{report}')
            for child in range(2):
                rows.append({'sha256': h(f'{family}_object_{report}_{child}'), 'size': 10 + report,
                    'type': 'pe', 'parent_sha256': subject, 'parent_size': 100 + report,
                    'parent_type': 'exe', 'parent_imphash': 'import_' + family,
                    'file_name': f'{family}_{child}.exe', 'file_path': f'c:/{family}/{report}', 'target': family})
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / 'disk.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        frame = clean_source(path, 'disk', root / 'audit')
        cohort, _ = eligible_cohort(frame, 10)
        splits, _ = make_splits(cohort, 'component', 5, 42)
        for train, test in splits:
            assert identity_overlap(cohort, train, test)['any_executable_overlap'] == 0
        # Shared image across subjects must transitively link all corresponding
        # report rows, including cross-role object==subject equality.
        linked = frame.copy()
        linked.loc[2, 'sha256'] = linked.loc[0, 'sha256']
        linked.loc[4, 'sha256'] = linked.loc[2, 'parent_sha256']
        groups = component_groups(linked, 'disk')
        assert groups[0] == groups[1] == groups[2] == groups[3] == groups[4] == groups[5]
        args = SimpleNamespace(text_max_features=20, dense_budget_mb=100)
        X = cohort[PREDICTORS['disk']]
        train, test = splits[0]
        pipe = make_pipeline('disk', PREDICTORS['disk'], 'Random Forest', RandomForestClassifier(n_estimators=5, random_state=42), args, 42)
        pipe.fit(X.iloc[train], pd.Categorical(cohort['target']).codes[train])
        vocab = pipe.named_steps['features'].named_transformers_['field_file_path'].vectorizer_.vocabulary_.copy()
        changed = X.iloc[test].copy()
        changed['file_path'] = 'uniquetesttokenzzzz'
        pipe.predict(changed)
        assert vocab == pipe.named_steps['features'].named_transformers_['field_file_path'].vectorizer_.vocabulary_
        assert not any('zzzz' in token for token in vocab)
        empty = StableTextVectorizer().fit(['', ''])
        assert empty.transform(['unseen']).shape == (1, 1)
        # A low-purity identical-vector family is retained by the fixed rule.
        network = pd.DataFrame([{**{c: 0 for c in PREDICTORS['network']},
            'sha256_sample': h(f'{f}_{i}'), 'target': f} for f in ['a', 'b'] for i in range(10)])
        npath = root / 'network.csv'
        network.to_csv(npath, index=False)
        cleaned = clean_source(npath, 'network', root / 'network_audit')
        kept, audit = eligible_cohort(cleaned, 10)
        assert len(kept) == 20 and audit['included'].all()
        assert kept[PREDICTORS['network']].eq(0).all().all()
        # Future-only tokens do not influence the representation; invalid hashes
        # fail identity eligibility rather than receiving a shared 'Unknown' key.
        bad = pd.DataFrame(rows)
        bad.loc[0, 'parent_sha256'] = 'Unknown'
        bad.to_csv(path, index=False)
        assert len(clean_source(path, 'disk', root / 'bad_audit')) == len(rows) - 1
        boot = group_bootstrap(np.array(['a', 'a', 'b', 'b']), np.array(['a', 'b', 'b', 'b']),
                               np.array(['g1', 'g1', 'g2', 'g2']), ['a', 'b'], 20, 42)
        assert len(boot['conditional_component_bootstrap']['macro_f1_percentile_95']) == 2
        # Temporal components crossing the cutoff are excluded in their entirety.
        md = pd.DataFrame({'source': 'disk', 'subject_sha256': frame['_subject'].unique(),
            'timestamp': ['2026-01-01'] * 6 + ['2026-04-01'] * 6, 'timestamp_kind': 'first_seen'})
        md.to_csv(root / 'metadata.csv', index=False)
        tr, te, _ = temporal_partition(frame, 'disk', root / 'metadata.csv', '2026-03-01')
        assert len(tr) == 12 and len(te) == 12
        # External overlap excludes every row belonging to an overlapping component.
        filtered, blocked = filter_external_identity(frame.iloc[:2], frame)
        assert len(blocked) == 2 and len(filtered) == len(frame) - 2
        # Regression: CatBoost may return a column of labels, including the
        # exact 311-row shape reported on the cluster.
        labels = np.arange(311) % 3
        assert np.array_equal(normalize_class_predictions(labels[:, None], 311, [0, 1, 2], 'CatBoost'), labels)
        assert normalize_class_predictions(np.asarray([[1]]), 1, [0, 1]).shape == (1,)
        dense64 = DenseWithinBudget(max_mb=1, dtype='float64')
        assert dense64.fit_transform(csr_matrix([[1., 0.]], dtype=np.float32)).dtype == np.float64
        try:
            DenseWithinBudget(max_mb=.05, dtype='float64').fit_transform(np.zeros((100, 100)))
        except MemoryError:
            pass
        else:
            raise AssertionError('Float64 dense-memory estimate used float32 byte counts')
        audit_context = {'source': 'disk', 'model': 'Naive Bayes', 'fold': 1}
        aligned, status = checked_model_probabilities(np.array([[.2, .8], [.6, .4]]),
            np.array([0, 1]), np.array([0, 1]), 2, audit_context, root / 'probability_checks')
        assert status['status'] == 'passed' and status['repaired'] is False
        assert np.array_equal(aligned, [[.2, .8], [.6, .4]])
        for bad in [np.array([[np.nan, .5]]), np.array([[-.1, 1.1]]),
                    np.array([[.1, .1]]), np.array([[0., 0.]])]:
            try:
                checked_model_probabilities(bad, [0, 1], [0, 1], 1,
                    audit_context, root / 'probability_checks')
            except RuntimeError as error:
                assert 'Naive Bayes' in str(error) and 'fold=1' in str(error)
                detail = json.loads((root / 'probability_checks/invalid_probabilities.json').read_text())
                assert detail['status'] == 'failed' and detail['repaired'] is False
            else:
                raise AssertionError('Invalid probability values were accepted or silently repaired')
        for invalid in [np.zeros((3, 2)), np.asarray([0, .5, 1]), np.asarray([0, 1, 9])]:
            try:
                normalize_class_predictions(invalid, 3, [0, 1, 2])
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid prediction shape/class labels were accepted')
        coded, notes = normalize_source_nominals(pd.DataFrame({
            'parent_imphash': [0, 1, 2**60 + 1, 2**60 + 2]}), 'disk')
        assert coded.loc[0, 'parent_imphash'] == ''
        assert coded.loc[1, 'parent_imphash'].startswith('opaque_numeric_')
        assert coded.loc[2, 'parent_imphash'] != coded.loc[3, 'parent_imphash']
        assert notes[0]['opaque_numeric_rows'] == 3
        # Automatic discovery must skip reduced Memory tables, including when
        # the usable source-schema copy has a download suffix in its filename.
        candidates = ensure_dir(root / 'source_discovery')
        pd.DataFrame({'sha256': [h('process')], 'total_processes': [2],
                      'target': ['alpha']}).to_csv(candidates / 'memory_dataset_preprocessed.csv', index=False)
        good = pd.DataFrame([{'sha256': h('process'), 'normalized_path': 'c:/sample.exe',
            'command_line': 'sample --run', 'parent_sha256': h('subject'),
            'total_processes': 2, 'created_files': '', 'registry': '',
            'modules_count': 1, 'target': 'alpha'}])
        good.to_csv(candidates / 'memory_dataset_preprocessed_high_accuracy(1).csv', index=False)
        chosen = source_file('memory', [candidates])
        assert chosen.name == 'memory_dataset_preprocessed_high_accuracy(1).csv'
        try:
            source_file('memory', [candidates], explicit=str(candidates / 'memory_dataset_preprocessed.csv'))
        except ValueError as error:
            assert 'parent_sha256' in str(error)
        else:
            raise AssertionError('Explicit reduced input was silently replaced')
    print('SELF-TEST PASSED: grouping, train-only vocabulary, min10/no-purity filtering, hash/holdout isolation, class shapes, float64 memory budget, strict probability diagnostics, nominal codes and schema discovery.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bundled_data = Path(__file__).resolve().parent / 'Dataset'
    default_data = bundled_data if bundled_data.is_dir() else DEFAULT_SLURM_DATA_DIR
    parser.add_argument('--data-dir', default=str(default_data))
    default_output = (Path(__file__).resolve().parent / 'results' / 'winmal_v20'
                      if bundled_data.is_dir() else DEFAULT_SLURM_OUTPUT_DIR)
    parser.add_argument('--output-dir', default=str(default_output))
    parser.add_argument('--datasets', nargs='+', choices=list(SCHEMAS), default=list(SCHEMAS))
    for source in SCHEMAS:
        parser.add_argument('--' + source + '-file', help='Explicit original schema-preserving input path')
    parser.add_argument('--experiments', nargs='+', choices=['primary', 'split-audit', 'ablations', 'collision', 'temporal', 'external', 'label-audit'],
                        default=['primary', 'split-audit', 'ablations', 'collision', 'temporal', 'external', 'label-audit'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 350])
    parser.add_argument('--folds', type=int, default=10, help='Requested folds; reduced only for independent-group feasibility, recorded in folds.json')
    parser.add_argument('--min-samples', type=int, default=10, help='Uniform fixed row support for all sources; also requires >=2 independent components for CV')
    parser.add_argument('--models', nargs='+', default=['all'], help='Quoted model names, e.g. "Logistic Regression" "Random Forest"; Dummy Majority always added')
    parser.add_argument('--diagnostic-models', nargs='+', default=['Logistic Regression', 'Random Forest'], help='Prespecified models for split/ablation/collision/holdout experiments')
    parser.add_argument('--collision-sources', nargs='+', choices=list(SCHEMAS), default=['network'])
    parser.add_argument('--bootstrap', type=int, default=500, help='Component-bootstrap draws on fixed primary OOF predictions; 0 disables')
    parser.add_argument('--text-max-features', type=int, default=512, help='Fixed per-source-field TF-IDF cap, fitted separately inside each training fold')
    parser.add_argument('--dense-budget-mb', type=int, default=512)
    parser.add_argument('--metadata-csv', help='source,subject_sha256,timestamp,timestamp_kind; original timestamps only')
    parser.add_argument('--temporal-cutoff', help='Prespecified UTC boundary; required with temporal metadata')
    parser.add_argument('--external-data-dir', help='Separate independently collected source tables with matching schema')
    parser.add_argument('--label-audit-csv', help='Completed independent reviews; template is exported automatically')
    parser.add_argument('--review-per-family', type=int, default=5)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--skip-xai', action='store_true')
    parser.add_argument('--xai-seed', type=int, default=42)
    parser.add_argument('--xai-background-size', type=int, default=32)
    parser.add_argument('--xai-explain-size', type=int, default=60)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--enable-tsne', action='store_true', help='Optional descriptive visualization only; never used as evaluation evidence')
    parser.add_argument('--skip-tsne', action='store_true', help='Compatibility flag; t-SNE is already off unless --enable-tsne')
    parser.add_argument('--skip-3d-tsne', action='store_true')
    parser.add_argument('--tsne-max-points', type=int, default=2000)
    parser.add_argument('--tsne-iterations', type=int, default=1000)
    parser.add_argument('--tsne-perplexity', type=float, default=30.)
    parser.add_argument('--resume', action='store_true', help='Reuse only completed model runs with matching data/code/split/config signatures')
    parser.add_argument('--audit-only', action='store_true', help='Prepare cohort/review/overlap audits without model fitting')
    parser.add_argument('--self-test', action='store_true', help='Run synthetic invariant checks and exit')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in ['folds', 'min_samples', 'text_max_features', 'dense_budget_mb', 'review_per_family', 'xai_background_size', 'xai_explain_size', 'tsne_max_points']:
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.folds < 2 or args.min_samples < 2 or args.bootstrap < 0 or not args.seeds or min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Need >=2 folds/support, nonnegative bootstrap/seeds, and unique seeds')
    if (args.metadata_csv is None) != (args.temporal_cutoff is None):
        raise ValueError('--metadata-csv and --temporal-cutoff must be provided together')
    if args.models != ['all'] and 'all' in args.models:
        raise ValueError('Use --models all by itself, or specify quoted model names')
    if args.enable_tsne and (args.tsne_iterations < 250 or args.tsne_perplexity <= 0):
        raise ValueError('t-SNE requires >=250 iterations and positive perplexity')
    if not args.skip_xai and args.xai_seed not in args.seeds:
        raise ValueError('--xai-seed must be in --seeds, or use --skip-xai')
    output = Path(args.output_dir).resolve()
    previous = None
    if (output / 'run_metadata.json').exists():
        if not args.resume:
            raise FileExistsError('Output already contains a run. Use a new --output-dir or --resume to prevent stale/mixed results.')
        previous = json.loads((output / 'run_metadata.json').read_text())
    ensure_dir(output)
    args.output_dir = str(output)
    args.code_sha256 = sha256_file(__file__)
    args.input_checksums = {}
    if previous:
        if previous.get('script_sha256') != args.code_sha256:
            raise ValueError('Resume requires the identical script; use a new output directory after code changes')
        ignored = {'resume', 'code_sha256', 'input_checksums', 'nonfatal_failures', 'gpu_enabled'}
        current_options = {k: v for k, v in vars(args).items() if k not in ignored}
        prior_options = {k: v for k, v in previous['arguments'].items() if k not in ignored}
        if current_options != prior_options:
            raise ValueError('Resume requires unchanged experimental options; use a new output directory')
    for name in ['metadata_csv', 'label_audit_csv']:
        value = getattr(args, name)
        if value:
            checksum = sha256_file(value)
            if previous and previous.get('input_checksums', {}).get(name, checksum) != checksum:
                raise ValueError(f'Resume supplemental input changed: {name}; use a new output directory')
            args.input_checksums[name] = checksum
    args.nonfatal_failures = []
    args.gpu_enabled = resolve_gpu_mode(args.use_gpu, args.require_gpu)
    print_header('WINMAL2026 RUNTIME')
    print(f'Script version: {SCRIPT_VERSION}', flush=True)
    print(f'CPU threads: {CPU_THREADS}; GPU enabled: {args.gpu_enabled}', flush=True)
    write_json(output / 'text_vectorizer_runtime_check.json', validate_text_vectorizers())
    metadata = describe_environment(args)
    metadata['started_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'run_metadata.json', metadata)
    results, statuses, failures = [], [], []
    directories = build_search_dirs(args.data_dir)
    started = time.perf_counter()
    for source in args.datasets:
        print_header(f'{source.upper()}: EVIDENCE-AUDIT SUITE')
        write_json(output / 'runtime_progress.json', {
            'source': source, 'stage': 'input_discovery_and_preparation'})
        try:
            path = source_file(source, directories, getattr(args, source + '_file'),
                               selection_report=output / source / 'input_selection.json')
            checksum = sha256_file(path)
            if previous and previous.get('input_checksums', {}).get(source, checksum) != checksum:
                raise ValueError('Resume source input changed; use a new output directory')
            args.input_checksums[source] = checksum
            if args.external_data_dir:
                external_candidate = source_file(source, [Path(args.external_data_dir)])
                external_checksum = sha256_file(external_candidate)
                if previous and previous.get('input_checksums', {}).get(source + '_external', external_checksum) != external_checksum:
                    raise ValueError('Resume external input changed; use a new output directory')
                args.input_checksums[source + '_external'] = external_checksum
            source_out = ensure_dir(output / source)
            audit_out = ensure_dir(source_out / 'audits')
            clean = clean_source(path, source, audit_out)
            clean[SCHEMAS[source] + ['_row_id', '_subject', '_object', '_component', '_vector']].to_csv(audit_out / 'clean_source_records.csv', index=False)
            cohort, eligibility = eligible_cohort(clean, args.min_samples)
            eligibility.to_csv(audit_out / 'primary_cohort_eligibility.csv', index=False)
            make_review_template(clean, source, audit_out, args)
            if args.label_audit_csv:
                validate_review_file(clean, source, args.label_audit_csv, audit_out)
            statuses.append({'source': source, 'experiment': 'label-audit', 'status': 'review_file_processed' if args.label_audit_csv else 'pending_independent_review'})
            if args.audit_only:
                statuses.append({'source': source, 'experiment': 'model_fitting', 'status': 'not_requested_audit_only'})
                continue
            if len(cohort) == 0 or cohort['target'].nunique() < 2:
                raise ValueError('Insufficient independent groups/families after explicit eligibility rules; inspect audits, do not revert to row splits')
            if not args.skip_plots:
                plot_class_counts(cohort['target'], audit_out / 'primary_family_support.png', f'{source.title()}: fixed eligible cohort')
            for seed in args.seeds:
                # Always build one primary partition and reuse it unchanged for
                # every feature ablation. Fit primary models only when requested.
                splits, split_meta = make_splits(cohort, 'component', args.folds, seed)
                if 'primary' in args.experiments:
                    rows, _, _ = run_cv(cohort, source, 'primary', 'component', PREDICTORS[source], seed, args,
                                        fixed_splits=splits, fixed_meta=split_meta)
                    results.extend(rows)
                if 'split-audit' in args.experiments:
                    for protocol in ['row', 'subject', 'object'] if source != 'network' else ['row']:
                        rows, _, _ = run_cv(cohort, source, 'split_diagnostic_' + protocol, protocol,
                                            PREDICTORS[source], seed, args, args.diagnostic_models)
                        results.extend(rows)
                if 'ablations' in args.experiments:
                    for name, fields in ablation_fields(source).items():
                        rows, _, _ = run_cv(cohort, source, 'ablation_' + name, 'component', fields, seed, args,
                                            args.diagnostic_models, fixed_splits=splits, fixed_meta=split_meta)
                        results.extend(rows)
                if 'collision' in args.experiments and source in args.collision_sources:
                    collisions = clean.copy()
                    collisions['_vector_component'] = component_groups(collisions, source, include_vectors=True)
                    common, table = eligible_cohort(collisions, args.min_samples, '_vector_component')
                    table.to_csv(audit_out / 'collision_common_cohort_eligibility.csv', index=False)
                    if common['target'].nunique() < 2:
                        statuses.append({'source': source, 'experiment': 'collision', 'seed': seed,
                                         'status': 'infeasible', 'reason': 'Exact-vector components leave fewer than two independently evaluable families'})
                    else:
                        for protocol in ['component', 'vector_component']:
                            rows, _, _ = run_cv(common, source, 'collision_common_' + protocol, protocol,
                                                PREDICTORS[source], seed, args, args.diagnostic_models)
                            results.extend(rows)
                pd.DataFrame(results).to_csv(output / 'all_experiment_results.csv', index=False)
            if 'temporal' in args.experiments:
                if args.metadata_csv:
                    train, test, evidence = temporal_partition(clean, source, args.metadata_csv, args.temporal_cutoff)
                    evidence['metadata_sha256'] = sha256_file(args.metadata_csv)
                    for seed in args.seeds:
                        results.extend(run_holdout(train, test, source, 'temporal', seed, args, evidence))
                    statuses.append({'source': source, 'experiment': 'temporal', 'status': 'completed'})
                else:
                    statuses.append({'source': source, 'experiment': 'temporal', 'status': 'blocked_missing_real_metadata_and_prespecified_cutoff'})
            if 'external' in args.experiments:
                if args.external_data_dir:
                    external_path = source_file(source, [Path(args.external_data_dir)])
                    if external_path == path or sha256_file(external_path) == sha256_file(path):
                        raise ValueError('External holdout cannot reuse the training input')
                    external = clean_source(external_path, source, ensure_dir(source_out / 'external_audits'))
                    external, blocked = filter_external_identity(cohort, external)
                    blocked.to_csv(source_out / 'external_audits' / 'training_identity_overlap_exclusions.csv', index=False)
                    for seed in args.seeds:
                        results.extend(run_holdout(cohort, external, source, 'external', seed, args,
                            {'external_sha256': sha256_file(external_path), 'overlap_excluded_rows': len(blocked),
                             'collection_independence': 'must be documented by authors; identity disjointness alone does not prove independent collection'}))
                    statuses.append({'source': source, 'experiment': 'external', 'status': 'completed'})
                else:
                    statuses.append({'source': source, 'experiment': 'external', 'status': 'blocked_missing_independent_collection'})
            if args.enable_tsne and not args.skip_tsne:
                # Explicitly descriptive full-cohort fit, separate from predictive
                # evaluation. This matrix is never reused by any classifier.
                transformer = source_preprocessor(source, PREDICTORS[source], 'Random Forest', args)
                sample = cohort.sample(n=min(len(cohort), args.tsne_max_points), random_state=42)
                matrix = transformer.fit_transform(sample[PREDICTORS[source]])
                dense = DenseWithinBudget(args.dense_budget_mb).fit_transform(matrix)
                out = ensure_dir(source_out / 'descriptive_tsne')
                ensure_dir(out / 'plots')
                labels = LabelEncoder().fit(sample['target'])
                run_tsne_analysis(pd.DataFrame(dense), labels.transform(sample['target']), list(labels.classes_), source, out,
                    AnalysisOptions(run_xai=False, run_tsne=True, run_tsne_3d=not args.skip_3d_tsne,
                        tsne_max_points=args.tsne_max_points, tsne_iterations=args.tsne_iterations,
                        tsne_perplexity=args.tsne_perplexity))
                write_json(out / 'scope.json', {'scope': 'descriptive visualization; full-cohort representation fit; not held-out performance or separability evidence'})
            statuses.append({'source': source, 'experiment': 'requested_model_runs', 'status': 'completed'})
        except Exception as error:
            failure = {'source': source, 'stage': 'source_suite', 'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
            progress = output / 'runtime_progress.json'
            if progress.exists():
                failure['last_stage'] = json.loads(progress.read_text())
            failures.append(failure)
            write_json(output / source / 'failure.json', failure)
            print(f'[FAILED] {source}: {error}', file=sys.stderr, flush=True)
        finally:
            pd.DataFrame(results).to_csv(output / 'all_experiment_results.csv', index=False)
            write_json(output / 'experiment_status.json', statuses)
            write_json(output / 'failures.json', failures + args.nonfatal_failures)
    export_paired_ablations(results, output)
    export_protocol_comparisons(results, output)
    if results:
        table = pd.DataFrame(results)
        main_rows = table.loc[table['variant'].eq('primary')]
        if len(main_rows):
            main_rows.groupby(['source', 'model'])[['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy']].agg(['mean', 'std', 'min', 'max']).to_csv(output / 'primary_seed_sensitivity_descriptive.csv')
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['elapsed_seconds'] = time.perf_counter() - started
    metadata['input_bytes'] = sum(source_file(source, directories, getattr(args, source + '_file')).stat().st_size
                                  for source in args.datasets if source in args.input_checksums)
    metadata['result_bytes_before_final_metadata_write'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    metadata['input_checksums'] = args.input_checksums
    metadata['failures'] = failures + args.nonfatal_failures
    metadata['status'] = 'completed_with_failures' if metadata['failures'] else 'completed_requested_available_experiments'
    try:
        import resource
        metadata['process_peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata['memory_scope'] = 'process-wide high-water RSS; not model-specific memory or remote sandbox cost'
    except ImportError:
        pass
    write_json(output / 'run_metadata.json', metadata)
    print(f'Results: {output}; status={metadata["status"]}', flush=True)
    return 1 if metadata['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
