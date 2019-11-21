import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from utilities import from_string_to_list


def missing_to_nan(df, info):
    """
    Identify missing or unknown data values and convert them to NaNs

    :param df:
    :param info:
    :return:
    """
    # .

    nan_placeholder = info["missing_or_unknown"].transform(from_string_to_list).apply(pd.Series)
    nan_placeholder.set_index(info["attribute"], inplace=True)

    for index, row in nan_placeholder.iterrows():
        feature_name = index
        values = pd.to_numeric(row, errors="ignore").tolist()
        df[feature_name].replace(to_replace=values, value=np.nan, inplace=True)

    return df


def drop_columns(df, n):
    nan_cl_sum = df.isnull().sum().sort_values(ascending=False)
    df.drop(columns=nan_cl_sum.nlargest(n).index, inplace=True)
    return df


def drop_rows(df):
    threshold = .10
    nan_row_sum = df.isnull().sum(axis="columns")
    rows_mask = nan_row_sum / len(df.columns) > threshold
    return df[~rows_mask]


def dummy(df, column):
    # Create the dummy columns and concat them to the dataframe
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)

    # Now drop the original column in place
    df.drop([column], axis=1, inplace=True)
    return df


def drop_all(df, to_be_dropped):
    for drop in to_be_dropped:
        if drop in df.columns:
            df.drop([drop], axis=1, inplace=True)


def encode_mixed(df, column, rules):
    """
    Encode a mixed variables according to rules mapping
    """

    for key, values in rules.items():
        df[f"{column}_{key}"] = df[column].isin(values).astype(int)

    df.drop([column], axis=1, inplace=True)
    return df


def clean_data(df, info, to_be_dropped, to_dummy_encode):
    # Identify missing or unknown data values and convert them to NaNs.
    df = missing_to_nan(df, info)
    drop_all(df, to_be_dropped)
    df = drop_rows(df)

    for column in to_dummy_encode:
        df = dummy(df, column)

    if "PRAEGENDE_JUGENDJAHRE" in df.columns:
        print("Encoding mixed variable PRAEGENDE_JUGENDJAHRE")
        df = encode_mixed(df, "PRAEGENDE_JUGENDJAHRE", {
            "40s": [1, 2],
            "50s": [3, 4],
            "60s": [6, 7],
            "70s": [8, 9],
            "80s": [10, 11, 12, 13],
            "90s": [14, 15],
            "Mainstream": [1, 3, 5, 8, 10, 12, 14]
        })

    if "CAMEO_INTL_2015" in df.columns:
        print("Encoding mixed variable CAMEO_INTL_2015")

        df = encode_mixed(df, "CAMEO_INTL_2015", {
            "Wealthy": [11, 12, 13, 14, 15],
            "Prosperous": [11, 12, 13, 14, 15],
            "Comfortable": [11, 12, 13, 14, 15],
            "Less Affluent": [11, 12, 13, 14, 15],
            "Poorer": [11, 12, 13, 14, 15],
            "Pre-Family Couples & Singles": [11, 21, 31, 41, 51],
            "Young Couples With Children": [12, 22, 32, 42, 52],
            "Families With School Age Children": [13, 23, 33, 43, 53],
            "Older Families &  Mature Couples": [14, 24, 34, 44, 54],
            "Elders In Retirement": [15, 25, 35, 45, 55]
        })

    if "WOHNLAGE" in df.columns:
        print("Encoding mixed variable WOHNLAGE")

        df = encode_mixed(df, "WOHNLAGE", {
            "Very good neighborhood": [1],
            "Good neighborhood": [2],
            "Average neighborhood": [3],
            "Poor neighborhood": [4],
            "Very poor neighborhood": [5],
            "Rural neighborhood": [7],
            "New in rural neighborhood": [8],
        })

    return df
