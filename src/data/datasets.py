import os
import pandas as pd

import src.data.utils as du

CMAPSS_DIR = os.path.join("data", "CMAPSS")

#a list containing all sensor measurement labels, sm01 to sm21
cmapss_sensor_list = []
for i in range(21):
    cmapss_sensor_list.append(f"sm{i+1:02d}")

#a list containing operating setting labels
cmapss_op_list = ["os1", "os2", "os3"]

def get_dataset(name, scaler):
    """get a dataset from cmapss datasets

    Args:
        name (str): CMAPSS1 to CMAPSS4.

    Raises:
        Exception: if the name is wrong.

    Returns:
        Dataset: a dictionary containing train and test dataframes and their scaler. 
    """
    if name.startswith("CMAPSS"):
        id = int(name[len("CMAPSS"):])
        train_adr = f"train_FD00{id}.txt"
        test_adr = f"test_FD00{id}.txt"
        rul_adr = f"RUL_FD00{id}.txt"
        train_df = cmapss_data_reader(train_adr)
        test_df = cmapss_data_reader(test_adr, rul_adr)

        if id in {2, 4}:
            scaler_factory = du.KMeansScaler(k=6, kmeans_features=cmapss_op_list, base_scaler=scaler)
            #scaler_factory = scaler
            if id==2:
                window_size = 20
            else:
                window_size = 15

        else:
            scaler_factory = scaler
            window_size = 30

        return dict(
            name=name,
            train=train_df,
            test=test_df,
            scaler_factory=scaler_factory,
            ignore_columns=cmapss_op_list,
            window_size=window_size)
    else:
        raise Exception(f"Unknown dataset {name}.")


def cmapss_data_reader(data_adr, rul_adr = None):
    """read cmapss train or test set into a df

    Args:
        data_adr (str): address of train or test set, e.g., train_FD001.txt for the first train set.
        rul_adr (str, optional): address of rul labels for test set. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe of a train or test set
    """
    data_folder = os.path.join(CMAPSS_DIR, data_adr)
    data_df = pd.read_csv(data_folder, header=None, delim_whitespace = True)
    #preparing the header for the data
    data_df.columns = ["id", "time", *cmapss_op_list, *cmapss_sensor_list]
    data_df = add_rul_to_df(data_df)

    if rul_adr is not None:
        rul_last = pd.read_csv(os.path.join(CMAPSS_DIR, rul_adr), header=None, delim_whitespace = True).values
        data_df["rul"] = data_df["rul"] + rul_last[data_df["id"].values-1].reshape(-1)

    return data_df

def add_rul_to_df(df):
    """add rul labels to df

    Args:
        df (pd.DataFrame): train or test df

    Returns:
        pd.DataFrame: same df with a new "rul" column
    """
    ruls = df.groupby("id").time.transform('max') - df.time
    df["rul"] = ruls
    return df
