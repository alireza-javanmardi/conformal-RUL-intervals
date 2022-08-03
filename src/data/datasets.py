import os
import typing
import pandas as pd
import funcy as fy
from sklearn.preprocessing import StandardScaler

import src.data.utils as du

CMAPSS_DIR = 'data\\CMAPSS'
#a list containing all sensor labels
cmapss_sensor_list = []
for i in range(21):
    cmapss_sensor_list.append(f"sm{i+1:02d}")

#a list containing operating condition labels
cmapss_op_list = ["os1", "os2", "os3"]

# CMAPSS1..4

class Dataset(typing.TypedDict):
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    scaler_factory: typing.Callable

def get_dataset(name: str) -> Dataset:
    if name.startswith("CMAPSS"):
        id = int(name[len("CMAPSS"):])
        train_adr = f"train_FD00{id}.txt"
        test_adr = f"test_FD00{id}.txt"
        rul_adr = f"RUL_FD00{id}.txt"
        train_df = cmapss_data_reader(train_adr)
        test_df = cmapss_data_reader(test_adr, rul_adr)

        # if id in {1, 2}:
        #     unwanted_sensors = {"SM01", "SM05", "SM10", "SM16", "SM18", "SM19"}
        # else:
        #     unwanted_sensors = {"SM01", "SM05", "SM16", "SM18", "SM19"}

        if id in {2, 4}:
            scaler_factory = fy.partial(
                du.KMeansScaler, 6, cmapss_op_list, StandardScaler)
        else:
            scaler_factory = StandardScaler

        return dict(
            name=name,
            train=train_df,
            test=test_df,
            scaler_factory=scaler_factory,
            ignore_columns=cmapss_op_list)
    else:
        raise Exception(f"Unknown dataset {name}.")


def cmapss_data_reader(data_adr: str, rul_adr: str = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_adr : string
        a sting such as "train_FD001.txt".

    Returns
    -------
    data_df : pandas dataframe
        dataframe corresponding to the provided address with meaningful column names.

    """
    #data_adr is train_FD001.txt for the first training data
    data_folder = os.path.join(CMAPSS_DIR, data_adr)
    data_df = pd.read_csv(data_folder, header=None, delim_whitespace = True)
    #preparing the data head for the data
    data_df.columns = ["id", "time", *cmapss_op_list, *cmapss_sensor_list]
    data_df = add_rul_to_df(data_df)

    if rul_adr is not None:
        rul_last = pd.read_csv(os.path.join(CMAPSS_DIR, rul_adr), header=None, delim_whitespace = True).values
        data_df["rul"] = data_df["rul"] + rul_last[data_df["id"].values-1].reshape(-1)

    return data_df

def add_rul_to_df(df: pd.DataFrame) -> pd.DataFrame:
    ruls = df.groupby("id").time.transform('max') - df.time
    df["rul"] = ruls
    return df
