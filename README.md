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
uv run train.py --metric recall --poslabel AdLD
```

Valid positive labels are:

- `MTDpos`
- `MTDneg`
- `AdLD`

The script writes an overfitting diagnostic figure to:

```text
data/outputs/overfitting_gap.png
```

The figure compares same-data performance against 3-fold cross-validation performance. A large gap is expected with this small sample size (`n=29`) and should be interpreted as evidence of overfitting risk.
