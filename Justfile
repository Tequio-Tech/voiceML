train-adld:
    uv run train.py --metric f1 --poslabel AdLD
    uv run train.py --metric sen --poslabel AdLD
    uv run train.py --metric ppv --poslabel AdLD

train-multi:
    uv run train.py --metric balanced_accuracy
