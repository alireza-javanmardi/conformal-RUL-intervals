from pandas.core.window.rolling import Window
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
import pandas as pd

def split_dataset(dataset, calval_size=0.2, random_state=None):
    """split data into train and calibration+validation

    Args:
        dataset: a dictionary containing train and test dataframes and their scaler. 
        calval_size (float, optional):  the proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        a dictionary containing train-calval split df, calval id's, train df, and test df. 
    """
    train_df = dataset["train"]
    test_df = dataset["test"]

    train_idx, calval_idx = split_by_group(X=train_df, groups=train_df["id"], n_splits=1, test_size=calval_size, random_state=random_state)
    train , calval = train_df.loc[train_idx], train_df.loc[calval_idx]
    train_split = dict(train=train, test=calval)

    return {
        **dataset,
        "train_split": train_split,
        "train": train_df,
        "test": test_df
    }

def split_by_group(X, groups, n_splits=1, test_size=0.2, random_state=None):
    """split data in a way that points with the same group be in the same split

    Args:
        X: Training data
        groups: Group labels for the samples used while splitting the dataset into train/test set
        n_splits (int, optional): Number of re-shuffling & splitting iterations. Defaults to 1.
        test_size (float, optional): proportion of groups to include in the test split. Defaults to 0.2.
        random_state (int, optional): Defaults to None.

    Returns:
        training and test indices.
    """
    gp_split = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gp_split.split(X, groups=groups))
    return train_idx, test_idx


def preprocess_split(
    split, scaler_factory, window_size,
    removable_cols: list[str] = [], 
    ignore_columns: list[str] = [],
    **kwargs):
    """preprocess a train-calval split: first scale them, then convert them to supervise data

    Args:
        split : a dictionary containing train-calval split
        scaler_factory (_type_): scaler to be used for scaling
        removable_cols (list[str], optional): list of sensors to be removed. Defaults to [].
        ignore_columns (list[str], optional): list of data columns such as "time" and "id" to be ignored. Defaults to [].

    Returns:
        a dictionary containing scaled train-calval split
    """
    split_scaler = scaler_factory
    train = apply_scaling_fn(split_scaler.fit_transform, split["train"])
    test = apply_scaling_fn(split_scaler.transform, split["test"])
    # removable_cols = list(train.columns[train.std(ddof=1) < 0.1e-10])
    removable_cols += ignore_columns
    train = train.drop(removable_cols, axis=1)
    test = test.drop(removable_cols, axis=1)
    train = dataframe_to_supervised(train, n_in=window_size-1, n_out=1)
    test = dataframe_to_supervised(test, n_in=window_size-1, n_out=1)

    return {
        **split,
        "train": train,
        "test": test
    }


def apply_scaling_fn(f, df: pd.DataFrame) -> pd.DataFrame:
    """apply scaling on a df and return it without scaling id's and rul labels

    Args:
        f (a callable function): scaler function
        df (pd.DataFrame): df to be scaled

    Returns:
        pd.DataFrame: scaled df
    """
    pre_cols = ["id"]
    post_cols = ["rul"]
    cols = pre_cols + post_cols
    sdf = df.drop(cols, axis=1)
    sdf = pd.DataFrame(data=f(sdf), index=df.index, columns=sdf.columns)
    return pd.concat([df[pre_cols], sdf, df[post_cols]], axis=1)


def dataframe_to_supervised(
    df, n_in=29, n_out=1, dropnan=True):
    """convert a dataframe of multiple time series into supervised dataframe using windowing technique

    Args:
        df (pd.DataFrame): input dataframe
        n_in (int): number of past measurements in time to be considered, (t-n_in, ... t-1)
        n_out (int): number of current and future measurements in time to be considered, (t, t+1, ... t+n_out)
        dropnan (bool): wether to drop rows with NaN values

    Returns:
        a dictionary containing X's, y's, id's, and indexes
    """
    X_list, y_list, id_list, idx_list = [], [], [], []
    for id in df.id.unique():
        id_df = df[df.id == id]
        id_df_supervised = series_to_supervised(id_df.drop(["id", "rul"], axis=1), n_in, n_out, dropnan)
        X = id_df_supervised.astype(np.float32).values
        X_list.append(X.reshape(X.shape[0], n_in+1, X.shape[1]//(n_in+1), 1)) # shape:(id_df.shape[0]-window_length+1, window_length, features)
        rul = id_df["rul"].astype(np.float32).values.reshape(-1,1)
        #piecewise RUL definition, comment it if you want linear RUL
        rul[rul>125] = 125
        y_list.append(rul[n_in:])
        id_list = id_list + X.shape[0]*[id]
        idx_list.append(id_df_supervised.index.values) 

    return {"X": np.vstack(X_list),
            "y": np.vstack(y_list),
            "id": np.array(id_list),
            "index": np.hstack(idx_list)}

def series_to_supervised(df: pd.DataFrame, n_in: int, n_out: int, dropnan: bool):
    """convert a single time series dataframe into supervised dataframe using windowing technique

    Args:
        df (pd.DataFrame): input dataframe
        n_in (int): number of past measurements in time to be considered, (t-n_in, ... t-1)
        n_out (int): number of current and future measurements in time to be considered, (t, t+1, ... t+n_out)
        dropnan (bool): wether to drop rows with NaN values

    Returns:
        agg (pd.DataFrame): supervised dataframe with (n_in+n_out+1)*df.shape[1] coulmns
    """
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

