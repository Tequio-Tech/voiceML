# voiceML

Preliminary Evaluation of a Machine Learning Model for Differentiating Adductor Laryngeal Dystonia and Muscle Tension Dysphonia.

## Running the Training Script

Use `uv` to run the project environment.

Train the default multinomial model with the three normalized outcomes `MTDpos`, `MTDneg`, and `AdLD`:

```bash
uv run train.py --metric balanced_accuracy
```

Train a binary model for one positive outcome versus all other outcomes:

```bash
uv run train.py --metric sen --poslabel AdLD
uv run train.py --metric ppv --poslabel AdLD
```

Valid positive labels are:

- `MTDpos`
- `MTDneg`
- `AdLD`

The script writes an overfitting diagnostic figure to `data/outputs` with a
configuration-specific filename.

For the default multinomial model:

```bash
uv run train.py --metric balanced_accuracy
```

writes:

```text
data/outputs/overfitting_gap__model-multinomial__metric-balanced_accuracy.png
```

For a binary sensitivity model:

```bash
uv run train.py --metric sen --poslabel AdLD
```

writes:

```text
data/outputs/overfitting_gap__model-binary__metric-sen__poslabel-AdLD.png
```

For a binary PPV (positive predictive value) model:

```bash
uv run train.py --metric ppv --poslabel AdLD
```

writes:

```text
data/outputs/overfitting_gap__model-binary__metric-ppv__poslabel-AdLD.png
```

The figure compares same-data performance against 3-fold cross-validation performance. A large gap is expected with this small sample size (`n=29`) and should be interpreted as evidence of overfitting risk.
