train-multi:
    uv run train.py
train-adld:
    uv run train.py --metric spc --poslabel AdLD
    uv run train.py --metric sen --poslabel AdLD
    uv run train.py --metric ppv --poslabel AdLD
train-mtd_pos:
    uv run train.py --metric spc --poslabel MTDpos
    uv run train.py --metric sen --poslabel MTDpos
    uv run train.py --metric ppv --poslabel MTDpos
train-mtd_neg:
    uv run train.py --metric spc --poslabel MTDneg
    uv run train.py --metric sen --poslabel MTDneg
    uv run train.py --metric ppv --poslabel MTDneg
