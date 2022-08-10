from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import funcy as fy
import typing

import src.data.datasets as datasets

class Split(typing.TypedDict):
    train: pd.DataFrame
    test: pd.DataFrame
class SplitDataset(datasets.Dataset):
    train_splits: list[Split]

def split_dataset(dataset: datasets.Dataset, folds=10) -> SplitDataset:
    train_df: pd.DataFrame = dataset["train"]
    test_df: pd.DataFrame = dataset["test"]
    scaler_factory = dataset["scaler_factory"]

    max_train_id = train_df["id"].values.max()
    folds = KFold(folds, shuffle=True, random_state=42).split(
        np.arange(max_train_id + 1) + 1)

    train_splits: list[Split] = []

    for train_idxs, val_idxs in folds:
        train = train_df[train_df["id"].isin(train_idxs)]
        val = train_df[train_df["id"].isin(val_idxs)]
        train_splits.append(dict(train=train, test=val))

    return {
        **dataset,
        "train_splits": train_splits,
        "train": train_df,
        "test": test_df
    }

def map_split_dataset(
    f: typing.Callable, dataset: SplitDataset,
    g: typing.Callable = lambda _: dict()) -> SplitDataset:
    return {
        **dataset,
        **f(dataset, **g(dataset)),
        "train_splits": fy.lmap(
            lambda s: f(s, **g(dataset)),
            dataset["train_splits"])
    }

def preprocess_split(
    split: Split, scaler_factory: typing.Callable,
    ignore_columns: list[str] = [],
    **kwargs) -> Split:
    split_scaler = scaler_factory()
    train = apply_scaling_fn(split_scaler.fit_transform, split["train"])
    test = apply_scaling_fn(split_scaler.transform, split["test"])
    removable_cols = list(train.columns[train.std(ddof=1) < 0.1e-10])
    removable_cols += ignore_columns
    train = train.drop(removable_cols, axis=1)
    test = test.drop(removable_cols, axis=1)
    train = dataframe_to_supervised(train)
    test = dataframe_to_supervised(test)

    return {
        **split,
        "train": train,
        "test": test,
        "in_dim": train[0][0].shape[-1]
    }


def apply_scaling_fn(f: typing.Callable, df: pd.DataFrame) -> pd.DataFrame:
    pre_cols = ["id"]
    post_cols = ["rul"]
    cols = pre_cols + post_cols
    sdf = df.drop(cols, axis=1)
    sdf = pd.DataFrame(data=f(sdf), index=df.index, columns=sdf.columns)
    return pd.concat([df[pre_cols], sdf, df[post_cols]], axis=1)


def dataframe_to_supervised(
    df: pd.DataFrame, n_in=29, n_out=1, dropnan=True) -> tuple[list[np.array], list[np.array]]:
    X_list, y_list = [], []
    for id in df.id.unique():
        id_df = df[df.id == id]
        # X  = series_to_supervised(id_df.drop(["id", "rul"], axis=1), n_in, n_out, dropnan).astype(np.float32).values
        # X_reshaped = np.reshape(X, (X.shape[0], n_in+1, X.shape[1]/(n_in+1)))
        # print(X_reshaped.shape)
        # X_list.append(X_reshaped)
        X = series_to_supervised(id_df.drop(["id", "rul"], axis=1), n_in, n_out, dropnan).astype(np.float32).values
        X_list.append(X.reshape(X.shape[0], n_in+1, X.shape[1]//(n_in+1), 1))
        rul = id_df["rul"].astype(np.float32).values.reshape(-1,1)
        #piecewise RUL definition
        rul[rul>130] = 130
        y_list.append(rul[n_in:])

    return X_list, y_list

def series_to_supervised(df: pd.DataFrame, n_in: int, n_out: int, dropnan: bool):
    n_vars = df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


preprocess_split_dataset = fy.partial(map_split_dataset, preprocess_split, g=lambda d: d)
