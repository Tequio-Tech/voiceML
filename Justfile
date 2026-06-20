train-adld:
    uv run train.py --metric spc --poslabel AdLD
    uv run train.py --metric sen --poslabel AdLD
    uv run train.py --metric ppv --poslabel AdLD

train-multi:
    uv run train.py --metric bal_acc
