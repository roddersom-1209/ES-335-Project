"""The train/test split, and the cross-validation folds used for tuning.

WHY A GROUPED SPLIT
Many phones appear several times in the data, once per storage variant
("iPhone 16 128GB", "iPhone 16 256GB", "iPhone 16 512GB"). These rows share
every spec except storage, and their prices differ by a fixed step.

With a plain random split of rows, one variant would usually land in the
training set and its sibling in the test set. The model would then be "tested"
on a phone it has almost seen, and the test score would be too optimistic. It
would measure memory of known phones rather than the ability to price a NEW
phone, which is the question we care about.

So we split by `model_group` (the model name with storage removed): all
variants of one phone go to the same side. The same rule applies inside the
training set when tuning: GroupKFold keeps a phone's variants in the same fold.

Run this file once (python src/split.py) to create data/processed/split.csv.
Every notebook then reads that file, so all models use exactly the same split.

The 5 cross-validation folds are saved in the same file (column "fold", 1-5 for
training rows, 0 for test rows). sklearn's GroupKFold breaks ties between
equally sized phone groups differently in different versions, so computing the
folds on the fly gave slightly different CV scores on different machines.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "phones_clean.csv"
SPLIT_PATH = ROOT / "data" / "processed" / "split.csv"


def make_split(df):
    """Return a Series of "train"/"test" labels, one per row, grouped by model_group.

    Note: test_size is a fraction of GROUPS (phone models), not of rows, so the
    test set has roughly, not exactly, 20 % of the rows.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["model_group"]))
    labels = pd.Series("train", index=df.index)
    labels.iloc[test_idx] = "test"
    return labels


def make_folds(train):
    """Return the CV fold (1 to 5) of each training row, grouped by model_group.

    Same rule as sklearn's GroupKFold: take the phone groups from largest to
    smallest and put each into the fold that has the fewest rows so far. Ties
    between equally sized groups are broken in a fixed order (by group name),
    so the folds are the same on every machine.
    """
    names, group_of_row = np.unique(train["model_group"], return_inverse=True)
    group_sizes = np.bincount(group_of_row)
    rows_in_fold = np.zeros(N_FOLDS)
    fold_of_group = np.zeros(len(names), dtype=int)
    for g in np.argsort(group_sizes, kind="stable")[::-1]:
        fold = np.argmin(rows_in_fold)
        rows_in_fold[fold] += group_sizes[g]
        fold_of_group[g] = fold + 1
    return pd.Series(fold_of_group[group_of_row], index=train.index)


def load_train_test(include_web_lookup=True):
    """Load the clean data and return (train, test) DataFrames.

    The training rows get a "fold" column (1-5) with their saved CV fold.

    include_web_lookup=False drops the 7 phones whose storage/price came from the
    manual web lookup, for the robustness check. Those phones keep the side of the
    split (and the fold) they were given, so both versions use the same split.
    """
    df = pd.read_csv(CLEAN_PATH)
    saved = pd.read_csv(SPLIT_PATH)
    assert len(saved) == len(df), "split.csv is out of date: re-run python src/split.py"
    df["fold"] = saved["fold"]
    split = saved["split"]
    if not include_web_lookup:
        keep = df["storage_source"] == "name"
        df, split = df[keep], split[keep]
    return df[split == "train"].copy(), df[split == "test"].drop(columns="fold")


def cv_folds(train):
    """The saved CV folds as a list of 5 (fit positions, validation positions) pairs.
    Use it in a for-loop, or pass it as cv=... to GridSearchCV."""
    fold = train["fold"].to_numpy()
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(1, N_FOLDS + 1)]


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    split = make_split(df)
    fold = pd.Series(0, index=df.index)
    fold[split == "train"] = make_folds(df[split == "train"])
    pd.DataFrame({"row": df.index, "model_group": df["model_group"], "split": split, "fold": fold}).to_csv(SPLIT_PATH, index=False)

    train, test = df[split == "train"], df[split == "test"]
    shared = set(train["model_group"]) & set(test["model_group"])
    print(f"Saved {SPLIT_PATH.relative_to(ROOT)}")
    print(f"train: {len(train)} rows, {train['model_group'].nunique()} model groups")
    print(f"test:  {len(test)} rows, {test['model_group'].nunique()} model groups ({len(test) / len(df):.1%} of rows)")
    print(f"model groups on both sides: {len(shared)} (must be 0)")
