import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav
    from pathlib import Path

    return Path, mo, np, pd, plt, sns, wav


@app.cell
def _(Path):
    data_dir = Path(__file__).resolve().parent / "Trio-data"
    labels_path = data_dir / "trio_data_labels_only_outcomes.csv"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    return data_dir, labels_path


@app.cell
def _(labels_path, pd):
    df = pd.read_csv(labels_path)
    df
    return (df,)


@app.cell
def _(data_dir, np, wav):
    def load_recording(file_name: str):
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        sample_rate, audio_data = wav.read(file_path)
        return sample_rate, audio_data / np.max(audio_data)

    def duration_sec(file_name: str):
        sr, y = load_recording(file_name)
        return len(y) / sr

    df = df.assign(duration_sec=[duration_sec(i) for i in df["id"].values])
    df
    return df, load_recording


@app.cell
def _(df, plt, sns):
    sns.histplot(df, x="duration_sec")
    plt.title("Distribution of Voice Recording Durations")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.gcf()
    return


@app.cell
def _(df, load_recording, plt):
    sample_id = df["id"].iloc[0]
    _, signal = load_recording(sample_id)
    plt.plot(signal, ".")
    plt.title(f"Waveform (No Audio Playback): {sample_id}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md("""
    **Privacy Note**

    This notebook intentionally does not render or play patient audio. It only visualizes derived signals.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
