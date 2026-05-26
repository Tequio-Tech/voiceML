from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io.wavfile as wav
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "Trio-data"
LABELS_CSV = "trio_data_labels_only_outcomes.csv"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")


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


def collect_labels(file_path: Path) -> pd.DataFrame:
    """Load labels CSV and augment with audio duration.

    Parameters
    ----------
    file_path : Path
        Path to labels CSV.

    Returns
    -------
    pd.DataFrame
        Labels with an added `duration_sec` column.

    Raises
    ------
    FileNotFoundError
        If the labels CSV does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {file_path}")
    df = pd.read_csv(file_path)
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

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)
    # print(f"Original audio length: {len(audio)} samples ({len(audio)/sr:.2f} seconds)")

    # Method 1: Using librosa's trim function (recommended)
    # This removes silence based on energy threshold
    trimmed_audio, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    # Pad or truncate to target length if specified
    if target_length is not None:
        if len(trimmed_audio) < target_length:
            # Pad with zeros
            trimmed_audio = np.pad(
                trimmed_audio, (0, target_length - len(trimmed_audio)), mode="constant"
            )
        elif len(trimmed_audio) > target_length:
            # Truncate
            trimmed_audio = trimmed_audio[:target_length]

    # Apply FFT
    fft_result = np.fft.fft(trimmed_audio)
    fft_magnitude = np.abs(fft_result)

    # Only keep positive frequencies (first half)
    fft_magnitude = fft_magnitude[: len(fft_magnitude) // 2]

    return fft_magnitude, trimmed_audio, sr


labelsdf = collect_labels(DATA_DIR / LABELS_CSV)


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
    L = []
    fft_ids = []
    for i in labels["id"].values:
        yf, _, _ = trim_silence_and_fft(get_path(i), target_length=2048)
        fft_ids.append(i)
        L.append(yf)

    # Artifact with FFT features aligned to label order by `id`
    fft_df = pd.DataFrame(L, index=fft_ids)
    fft_df.index.name = "id"

    # Reindex in the exact labels order and fail fast if any ID is missing
    fft_df = fft_df.reindex(labels["id"].values)
    if fft_df.isnull().any().any():
        missing_ids = fft_df.index[fft_df.isnull().any(axis=1)].tolist()
        raise ValueError(f"Missing FFT rows for ids: {missing_ids}")

    # Labels aligned 1:1 with FFT rows
    labels_aligned = labels.set_index("id").loc[fft_df.index]
    if len(labels_aligned) != len(fft_df):
        raise ValueError("Label/FFT length mismatch after alignment.")
    if not (labels_aligned.index == fft_df.index).all():
        raise ValueError("Label/FFT index mismatch after alignment.")
    if fft_df.isnull().any(axis=1).any():
        raise ValueError("Null FFT rows detected after alignment.")

    return labels_aligned, fft_df


labels, fftdf = perform_fft(labelsdf)

# # 1. Prepare your data elements
Y = labels["diagnosis"]
X = fftdf.values
# 2. Build the Pipeline to prevent data leakage during CV
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svd", TruncatedSVD(random_state=42)),
        (
            "clf",
            LogisticRegression(solver="saga", max_iter=10000, random_state=42),
        ),
    ]
)

# 3. Create a clean Parameter Grid
# Parameterization avoids deprecated `penalty` usage in sklearn>=1.8:
# l1_ratio=1 -> L1, l1_ratio=0 -> L2, 0<l1_ratio<1 -> ElasticNet.
param_grid = [
    {
        "svd__n_components": list(range(2, 30)),
        "clf__l1_ratio": np.linspace(0.0, 1.0, 11),
        "clf__C": np.logspace(-4, 2, 6),
    },
]

# 4. Define Stratified Cross-Validation
# With ~29 total samples and a minimum class size of 9, a 3-fold split ensures
# each fold gets exactly 3 samples from the smallest class to keep evaluation stable.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 5. Set up the Grid Search targeting balanced accuracy
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,  # Uses all available CPU cores
    verbose=1,
)

# 6. Run the tuning execution
print("Starting hyperparameter tuning...")
grid_search.fit(X, Y)

# --- Output Results ---
print("\n=== Tuning Complete ===")
print(f"Best CV Balanced Accuracy: {grid_search.best_score_:.4f}")
print("Best Hyperparameters found:")
for param, val in grid_search.best_params_.items():
    print(f"  {param}: {val}")

# Evaluate performance on the entire set using the optimal setup
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)

print("\n=== Final Performance Profile ===")
print(metrics.classification_report(Y, y_pred))
print(f"Overall Balanced Accuracy: {metrics.balanced_accuracy_score(Y, y_pred):.4f}")
