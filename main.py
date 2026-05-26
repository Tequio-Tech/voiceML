"""
- referral to the ENT / Voice based on recording, different classes.
- diff overlapping syndromes, or cluster
"""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "Trio-data"
LABELS_CSV = "trio_data_labels_only_outcomes.csv"
PLOTS_ENABLED = True

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")


def get_path(file_name: str) -> Path:
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    return file_path


def load_recording(id: str):
    sample_rate, audio_data = wav.read(get_path(id))

    return sample_rate, audio_data / np.max(audio_data)


def get_duration_in_sec(id):
    sr, y = load_recording(id)
    return len(y) / sr


case_desc = dict()
case_desc["MTDpos"] = "Muscle Tension Dysphonia with lesions (MTD)"
case_desc["MTDneg"] = "Muscle Tension Dysphonia without lesions (MTDneg)"
case_desc["AdLD"] = "Adductor Laryngeal  Dystonia (AdLD)"
pd.DataFrame(case_desc, index=["Description"]).T.reset_index().rename(
    columns={"index": "diagnosis"}
)


labels_path = DATA_DIR / LABELS_CSV
if not labels_path.exists():
    raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

df = pd.read_csv(labels_path)
df = df.assign(duration_sec=[get_duration_in_sec(id) for id in df.id.values])


#
label_map = {"MTD": "MTDpos", "MTD_no_lession": "MTDneg", "AdLD": "AdLD"}


df["diagnosis"] = df["diagnosis"].map(lambda x: label_map[x])

df["diagnosis"].value_counts(normalize=False)

df["diagnosis"].value_counts(normalize=True)

# we have a ~33 split but we only have 29 patients total

if PLOTS_ENABLED:
    sns.histplot(df, x="duration_sec")
    plt.title("Distribution of Voice Recording Durations (29 patients)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")

"""# EDA RECORDINGS"""

# prompt: how do I play audio data in .way format ?
id = "G4-0010 RP2.wav"
_, s = load_recording(id)
if PLOTS_ENABLED:
    plt.plot(s, ".")

# Audio playback intentionally disabled for patient-data privacy (HIPAA).
if PLOTS_ENABLED:
    plt.title(
        f"Raw Data for patient 10 \n num data points ={len(s.flatten()):,} , diagnosis = AdLD "
    )
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

df[df["id"].apply(lambda x: "G4-0010" in x)]

# collect all audio
audio = pd.DataFrame(
    [load_recording(id)[1] for id in df.id.values], index=df.id
).fillna(0)  # .iloc[:,0:duration]
sel_ids = df[df.duration_sec <= 10]["id"].values
raw_audio = audio.loc[sel_ids]

pts, observations = raw_audio.shape


## svd is done on a
## perform SV on th audio
U, s, V = np.linalg.svd(raw_audio.values, full_matrices=False, compute_uv=True)
svdr = pd.DataFrame(
    {
        "num comp": np.arange(1, len(s) + 1),
        "% of variance explained": np.cumsum(s**2 / np.sum(s**2)),
    }
)


pts, observations = raw_audio.shape

if PLOTS_ENABLED:
    svdr.round(2).plot.scatter(x="num comp", y="% of variance explained")
    plt.title(
        f"We transformed our data from {observations} features per patient to 20 features per patient\n. Yet retained 80% of the explanatory power\n"
    )


X = StandardScaler().fit_transform(U[:, 0:15])
Y = df.set_index("id").loc[sel_ids]["diagnosis"]
clf = LogisticRegression(penalty="l2").fit(X, Y)

print(metrics.classification_report(Y, clf.predict(X)))

if PLOTS_ENABLED:
    metrics.ConfusionMatrixDisplay.from_predictions(
        Y == "AdLD", clf.predict(X) == "AdLD"
    )


X = U[:, 0:19]
Y = df.set_index("id").loc[sel_ids]["diagnosis"]
clf = LogisticRegression(penalty="l2").fit(X, Y)

print(metrics.classification_report(Y, clf.predict(X)))


X = U[:, 0:20]
Y = df.set_index("id").loc[sel_ids]["diagnosis"]
clf = LogisticRegression(penalty="l2").fit(X, Y)

print(metrics.classification_report(Y, clf.predict(X)))

data_dict = {}
srs = []
audio_length = []

for i in df["id"].values:
    sr, ad = load_recording(i)
    srs.append(sr)
    audio_length.append(len(ad))

df2 = pd.DataFrame(
    {"srs": srs, "audio_length": audio_length, "id": df["id"].values}
).merge(df, on="id")

df2.sort_values("audio_length")

if PLOTS_ENABLED:
    sns.stripplot(data=df2, x="diagnosis", y="audio_length")
    df2["audio_length"].hist()

"""# Machine Learning"""


def trim_silence_and_fft(
    audio_path,
    target_length=None,
    silence_threshold=0.01,
    frame_length=2048,
    hop_length=512,
    top_db=20,
):
    """
    Process audio file by trimming silence and applying FFT

    Parameters:
    - audio_path: path to the .wav file
    - target_length: desired length for FFT output (if None, uses trimmed length)
    - silence_threshold: energy threshold for silence detection
    - frame_length: frame length for silence detection
    - hop_length: hop length for silence detection
    - top_db: threshold for trimming (higher = more aggressive trimming)

    Returns:
    - fft_magnitude: FFT magnitude spectrum
    - trimmed_audio: trimmed audio signal
    - sample_rate: sampling rate
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


df2

get_path(i)

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

n_components = 10

L = []

for i in df["id"].values:
    yf, trim, _ = trim_silence_and_fft((get_path(i)), target_length=2048)

    L.append(yf)

X = pd.DataFrame(L, index=df2["id"])
# X = X.T.rolling(window=10).mean().dropna().T

Xs = StandardScaler().fit_transform(X)

svd_ = TruncatedSVD(n_components=n_components).fit(Xs)

U = svd_.transform(Xs)
# U = U.cumsum(1)

# age_scaled = (df2.age - df2.age.mean())/ df2.age.std()

# X=np.concat([U,age_scaled.values.reshape(-1,1)],axis=1)#.shape

Y = df2["diagnosis"]


# # # Create a dummy classifier (e.g., predicting the most frequent class)
# yhat_prior = np.random.choice(list(df2['diagnosis'].unique()),df2.shape[0])
# print("Dummy Classifier Report:")
# print(metrics.classification_report(Y, yhat_prior))


# Log Reg
method = "Log Reg"
if method == "Log Reg":
    clf = LogisticRegression(
        fit_intercept=True, max_iter=10000, penalty="l1", solver="saga2"
    ).fit(U, Y)
else:
    method = "Hist"
    clf = HistGradientBoostingClassifier(max_depth=20, max_iter=100).fit(U, Y)

print(metrics.classification_report(Y, clf.predict(U)))


if PLOTS_ENABLED:
    metrics.ConfusionMatrixDisplay.from_predictions(Y, clf.predict(U))
    plt.title(method)


# 1. Prepare your data elements
X = pd.DataFrame(L, index=df2["id"])
Y = df2["diagnosis"]

# 2. Build the Pipeline to prevent data leakage during CV
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svd", TruncatedSVD(random_state=42)),
        (
            "clf",
            LogisticRegression(
                max_iter=10000, multi_class="multinomial", random_state=42
            ),
        ),
    ]
)

# 3. Create a clean Parameter Grid
# We split it into two dictionaries to avoid passing invalid combinations (like l1_ratio to L1/L2)
param_grid = [
    {
        "svd__n_components": [2, 3, 5, 9, 10, 20],
        "clf__solver": ["saga"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": np.logspace(-4, 2, 6),  # Exploration from 0.001 to 100
    },
    {
        "svd__n_components": [2, 3, 5, 9, 10, 20],
        "clf__solver": ["saga"],
        "clf__penalty": ["elasticnet"],
        "clf__C": np.logspace(-4, 2, 6),
        "clf__l1_ratio": [0.2, 0.5, 0.8],  # Explores the mix of L1 and L2
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
