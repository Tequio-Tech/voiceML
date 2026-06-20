# voiceML

Preliminary Evaluation of a Machine Learning Model for Differentiating Adductor Laryngeal Dystonia and Muscle Tension Dysphonia.
## Background
Introduction
Voice disorders are common and diagnostically heterogeneous, with meaningful functional and occupational consequences. In a cross-sectional survey of 1,326 adults, the lifetime prevalence of self-reported voice disorder was 29.9%, and 6.6% reported a current voice disorder. Respondents also reported reduced job performance and voice-related work absence.1 Given that dysphonia can arise from structural, inflammatory, behavioral, and neurologic mechanisms, accurate subtyping is essential for selecting appropriate evaluation and treatment pathways.2
Two diagnoses that frequently overlap in perceptual presentation are muscle tension dysphonia (MTD) and adductor laryngeal dystonia (AdLD; often termed adductor spasmodic dysphonia). Both can present with strained, effortful voice qualities, and MTD can mimic the perceptual features of AdLD, creating diagnostic confusion when evaluation relies predominantly on expert auditory-perceptual judgment.3 AdLD is a task-specific focal dystonia characterized by involuntary, task-dependent spasms of the laryngeal adductor muscles during speech and is typically managed with botulinum toxin chemodenervation.4,5 In contrast, MTD is a voice disorder characterized by dysregulated, hyperfunctional laryngeal muscle activity in the absence of an underlying neurologic etiology and is typically managed with behavioral voice therapy.3 
To reduce subjectivity, objective acoustic analysis and machine learning (ML) have been increasingly applied to voice disorder detection and classification. Many ML systems report strong performance, although these frequently differentiate “pathological vs healthy,” and capture only the more common causes of dysphonia.6,7 Additionally, systematic reviews note that much of the ML voice-disorder literature relies on a limited number of shared databases, which often incorporate constrained speech tasks and inconsistent diagnostic labels, potentially limiting the ability to capture etiologic distinctions, task-dependent features, and real-world clinical variability. Additionally, these studies often rely on extensive preprocessing and manual feature engineering and require substantial computational resources.7 
These limitations highlight the need for simpler approaches using minimally processed data with reduced reliance on feature engineering. In this pilot study, we evaluate the feasibility of applying singular value decomposition (SVD) and logistic regression to raw voice recordings to differentiate AdLD from MTD within a small, expert-labeled cohort.


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
uv run train.py --metric f1 --poslabel AdLD
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

For a binary F1 model:

```bash
uv run train.py --metric f1 --poslabel AdLD
```

writes:

```text
data/outputs/overfitting_gap__model-binary__metric-f1__poslabel-AdLD.png
data/outputs/cv_metric_profile__model-binary__metric-f1__poslabel-AdLD.png
```

The figure compares same-data performance against 3-fold cross-validation performance. A large gap is expected with this small sample size (`n=29`) and should be interpreted as evidence of overfitting risk.

Binary runs also print a 3-fold cross-validated metric profile for PPV,
sensitivity, specificity, and F1, and save the matching profile figure to
`data/outputs`.
