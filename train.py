import argparse
from pathlib import Path
from typing import Any

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io.wavfile as wav
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "Trio-data"
LABELS_CSV = "trio_data_labels_only_outcomes.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "outputs"
OVERFITTING_FIGURE = OUTPUT_DIR / "overfitting_gap.png"
LABEL_MAP = {"MTD": "MTDpos", "MTD_no_lession": "MTDneg", "AdLD": "AdLD"}
VALID_LABELS = tuple(LABEL_MAP.values())
VALID_METRICS = ("balanced_accuracy", "recall")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a logistic-regression voice classification model."
    )
    parser.add_argument(
        "--metric",
        choices=VALID_METRICS,
        default="balanced_accuracy",
        help="Metric to optimize during grid search.",
    )
    parser.add_argument(
        "--poslabel",
        choices=VALID_LABELS,
        help=(
            "Positive label for binary training. If omitted, the script trains "
            "a multinomial model. Required when --metric recall is used."
        ),
    )
    args = parser.parse_args()
    if args.metric == "recall" and args.poslabel is None:
        parser.error("--metric recall requires --poslabel for binary recall scoring.")
    return args


def get_path(file_name: str) -> Path:
    """Resolve and validate an audio file path.

    Parameters
    ----------
    file_name : str
        Audio filename relative to `DATA_DIR`.

    Returns
    -------
    Path
        Absolute path to the requested audio file.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    """
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    return file_path


def load_recording(id: str) -> tuple[int, npt.NDArray[np.float64]]:
    """Load and normalize a WAV recording.

    Parameters
    ----------
    id : str
        Recording filename.

    Returns
    -------
    tuple[int, npt.NDArray[np.float64]]
        Sample rate and max-normalized waveform.
    """
    sample_rate, audio_data = wav.read(get_path(id))

    return sample_rate, audio_data / np.max(audio_data)


def get_duration_in_sec(id: str) -> float:
    """Compute recording duration in seconds.

    Parameters
    ----------
    id : str
        Recording filename.

    Returns
    -------
    float
        Duration in seconds.
    """
    sr, y = load_recording(id)
    return len(y) / sr


def normalize_diagnosis_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CSV diagnosis names to the public training labels."""
    normalized = df["diagnosis"].map(LABEL_MAP)
    if normalized.isna().any():
        unknown = sorted(df.loc[normalized.isna(), "diagnosis"].dropna().unique())
        raise ValueError(
            f"Unknown diagnosis label(s): {unknown}. Expected one of {sorted(LABEL_MAP)}."
        )
    return df.assign(diagnosis=normalized)


def collect_labels(file_path: Path) -> pd.DataFrame:
    """Load labels CSV and augment with audio duration.

    Parameters
    ----------
    file_path : Path
        Path to labels CSV.

    Returns
    -------
    pd.DataFrame
        Labels with normalized diagnoses and an added `duration_sec` column.

    Raises
    ------
    FileNotFoundError
        If the labels CSV does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    df = normalize_diagnosis_labels(df)
    return df.assign(duration_sec=[get_duration_in_sec(id) for id in df.id.values])


def trim_silence_and_fft(
    audio_path: Path,
    target_length: int | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: int = 20,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any], int]:
    """Trim silence from an audio file and compute FFT magnitude features.

    Parameters
    ----------
    audio_path : Path
        Path to the WAV file.
    target_length : int | None, optional
        Desired waveform length before FFT. If `None`, uses trimmed length.
    frame_length : int, optional
        Frame length for silence trimming.
    hop_length : int, optional
        Hop length for silence trimming.
    top_db : int, optional
        Threshold (in dB) for silence trimming aggressiveness.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[Any], int]
        FFT magnitude (positive frequencies), trimmed waveform, and sample rate.
    """

    audio, sr = librosa.load(audio_path, sr=None)
    trimmed_audio, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    if target_length is not None:
        if len(trimmed_audio) < target_length:
            trimmed_audio = np.pad(
                trimmed_audio, (0, target_length - len(trimmed_audio)), mode="constant"
            )
        elif len(trimmed_audio) > target_length:
            trimmed_audio = trimmed_audio[:target_length]

    fft_result = np.fft.fft(trimmed_audio)
    fft_magnitude = np.abs(fft_result)
    fft_magnitude = fft_magnitude[: len(fft_magnitude) // 2]

    return fft_magnitude, trimmed_audio, sr


def perform_fft(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute FFT features for label rows and align outputs by `id`.

    Parameters
    ----------
    labels : pd.DataFrame
        Label table with an `id` column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Labels reindexed to FFT rows and FFT feature table.

    Raises
    ------
    ValueError
        If FFT rows are missing after reindexing, if null rows are present,
        or if label/feature alignment checks fail.
    """
    features = []
    fft_ids = []
    for i in labels["id"].values:
        yf, _, _ = trim_silence_and_fft(get_path(i), target_length=2048)
        fft_ids.append(i)
        features.append(yf)

    fft_df = pd.DataFrame(features, index=fft_ids)
    fft_df.index.name = "id"

    fft_df = fft_df.reindex(labels["id"].values)
    if fft_df.isnull().any().any():
        missing_ids = fft_df.index[fft_df.isnull().any(axis=1)].tolist()
        raise ValueError(f"Missing FFT rows for ids: {missing_ids}")

    labels_aligned = labels.set_index("id").loc[fft_df.index]
    if len(labels_aligned) != len(fft_df):
        raise ValueError("Label/FFT length mismatch after alignment.")
    if not (labels_aligned.index == fft_df.index).all():
        raise ValueError("Label/FFT index mismatch after alignment.")
    if fft_df.isnull().any(axis=1).any():
        raise ValueError("Null FFT rows detected after alignment.")

    return labels_aligned, fft_df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svd", TruncatedSVD(random_state=42)),
            (
                "clf",
                LogisticRegression(solver="saga", max_iter=10000, random_state=42),
            ),
        ]
    )


def build_param_grid() -> list[dict[str, Any]]:
    # l1_ratio=1 -> L1, l1_ratio=0 -> L2, 0<l1_ratio<1 -> ElasticNet.
    return [
        {
            "svd__n_components": list(range(2, 30)),
            "clf__l1_ratio": np.linspace(0.0, 1.0, 11),
            "clf__C": np.logspace(-4, 2, 6),
        },
    ]


def build_target(labels: pd.DataFrame, poslabel: str | None) -> pd.Series:
    if poslabel is None:
        return labels["diagnosis"]
    return labels["diagnosis"].where(
        labels["diagnosis"] == poslabel, other=f"not_{poslabel}"
    )


def build_scorer(metric_name: str, poslabel: str | None) -> str | Any:
    if metric_name == "balanced_accuracy":
        return "balanced_accuracy"
    if poslabel is None:
        raise ValueError("Recall scoring requires a positive label.")
    return make_scorer(metrics.recall_score, pos_label=poslabel, zero_division=0)


def score_predictions(
    metric_name: str,
    y_true: pd.Series,
    y_pred: npt.ArrayLike,
    poslabel: str | None,
) -> float:
    if metric_name == "balanced_accuracy":
        return metrics.balanced_accuracy_score(y_true, y_pred)
    if poslabel is None:
        raise ValueError("Recall scoring requires a positive label.")
    return metrics.recall_score(y_true, y_pred, pos_label=poslabel, zero_division=0)


def plot_overfitting_gap(
    train_score: float,
    cv_score: float,
    metric_name: str,
    n_samples: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Same-data performance", train_score, "#0F766E"),
        ("3-fold cross-validation", cv_score, "#B45309"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    fig.suptitle("Overfitting Check: Same-Data vs Cross-Validation", fontsize=14)
    fig.text(0.5, 0.91, f"Metric: {metric_name} | n={n_samples}", ha="center")

    for ax, (title, score, color) in zip(axes, panels, strict=True):
        ax.bar([title], [score], color=color, width=0.55)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.text(0, min(score + 0.04, 0.96), f"{score:.3f}", ha="center")
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.25)

    gap = train_score - cv_score
    fig.text(0.5, 0.02, f"Generalization gap: {gap:.3f}", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labelsdf = collect_labels(DATA_DIR / LABELS_CSV)
    labels, fftdf = perform_fft(labelsdf)
    y = build_target(labels, args.poslabel)
    x = fftdf.values

    pipeline = build_pipeline()
    param_grid = build_param_grid()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = build_scorer(args.metric, args.poslabel)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    mode = "binary" if args.poslabel else "multinomial"
    print(f"Training mode: {mode}")
    print(f"Metric: {args.metric}")
    if args.poslabel:
        print(f"Positive label: {args.poslabel}")
    print("Starting hyperparameter tuning...")
    grid_search.fit(x, y)

    print("\n=== Tuning Complete ===")
    print(f"Best CV {args.metric}: {grid_search.best_score_:.4f}")
    print("Best Hyperparameters found:")
    for param, val in grid_search.best_params_.items():
        print(f"  {param}: {val}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x)
    train_score = score_predictions(args.metric, y, y_pred, args.poslabel)
    plot_overfitting_gap(
        train_score=train_score,
        cv_score=grid_search.best_score_,
        metric_name=args.metric,
        n_samples=len(y),
        output_path=OVERFITTING_FIGURE,
    )

    print("\n=== Final Performance Profile ===")
    print(metrics.classification_report(y, y_pred, zero_division=0))
    print(f"Same-data {args.metric}: {train_score:.4f}")
    print(f"Cross-validation {args.metric}: {grid_search.best_score_:.4f}")
    print(f"Overfitting figure saved to: {OVERFITTING_FIGURE}")


if __name__ == "__main__":
    main()
