import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def setup_1():
    import marimo as mo
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

    def collect_all_recordings() -> pd.DataFrame:
        # collect all audio
        audio = pd.DataFrame(
        [load_recording(id)[1] for id in df.id.values], index=df.id
        ).fillna(0)  # .iloc[:,0:duration]
        sel_ids = df[df.duration_sec <= 10]["id"].values
        raw_audio = audio.loc[sel_ids]

        pts, observations = raw_audio.shape

        return raw_audio

    labels_path = DATA_DIR / LABELS_CSV
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    df = pd.read_csv(labels_path)
    df = df.assign(duration_sec=[get_duration_in_sec(id) for id in df.id.values])

    label_map = {"MTD": "MTDpos", "MTD_no_lession": "MTDneg", "AdLD": "AdLD"}
    df["diagnosis"] = df["diagnosis"].map(lambda x: label_map[x])

    df["diagnosis"].value_counts(normalize=False)

    case_desc = dict()
    case_desc["MTDpos"] = "Muscle Tension Dysphonia with lesions (MTD)"
    case_desc["MTDneg"] = "Muscle Tension Dysphonia without lesions (MTDneg)"
    case_desc["AdLD"] = "Adductor Laryngeal  Dystonia (AdLD)"

    #show case description
    print(pd.DataFrame(case_desc, index=["Description"]).T.reset_index().rename(
        columns={"index": "diagnosis"}
    )
         )

    print(case_desc)
    return (
        GridSearchCV,
        PLOTS_ENABLED,
        Pipeline,
        StandardScaler,
        StratifiedKFold,
        TruncatedSVD,
        df,
        get_path,
        load_recording,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(df, plt, sns):
    sns.histplot(df, x="duration_sec")
    plt.title("Distribution of Voice Recording Durations (29 patients)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")

    return


@app.cell
def _(load_recording, plt):

    id = "G4-0010 RP2.wav"
    _, s = load_recording(id)
    plt.plot(s, ".")
    plt.title(
            f"Raw Data for patient 10 \n num data points ={len(s.flatten()):,} , diagnosis = AdLD "
        )
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    return


@app.cell
def _():
    return


@app.cell
def _():





    return


@app.cell
def _(
    GridSearchCV,
    PLOTS_ENABLED,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    TruncatedSVD,
    df,
    df2,
    get_path,
    np,
    pd,
    plt,
    trim_silence_and_fft,
):

    """# Machine Learning"""





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
    return (i,)


if __name__ == "__main__":
    app.run()
