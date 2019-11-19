from pandas.api.types import is_numeric_dtype
from .utilities import from_string_to_list


def missing_to_nan(df, info):
    """
    Identify missing or unknown data values and convert them to NaNs

    :param df:
    :param info:
    :return:
    """
    # .

    # The list of special values is given as a string, convert it to a list of strings
    info["missing_or_unknown"] = info["missing_or_unknown"].transform(from_string_to_list)

    for index, row in info.iterrows():
        attribute = row["attribute"]
        missing_or_unknown = row["missing_or_unknown"]

        if missing_or_unknown:      
            if is_numeric_dtype(df[attribute].dtype):
                missing_or_unknown = list(map(int, missing_or_unknown))

            df[attribute].replace(to_replace=missing_or_unknown, value=np.nan, inplace=True)
    
    return df


def drop_columns(df, n):
    nan_histogram = df.isnull().sum()
    to_be_dropped = missing_or_unknown.nlargest(n).index
    
    return df.drop(columns=to_be_dropped, inplace=True)


def drop_rows(df):
    n_columns = len(df.columns)
    rows_selector = df.isnull().sum(axis="columns")/n_columns > 0.10
    return df[~rows_selector]


def dummy(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)

    # now drop the original 'country' column (you don't need it anymore)
    df.drop([column], axis=1, inplace=True)
    return df


def preprocess(df, info):
    # Identify missing or unknown data values and convert them to NaNs.

    df = missing_to_nan(df, info)
    
    df = drop_columns(df, 7):
    
    df = drop_rows(df)
        
    to_be_encoded = ["CJT_GESAMTTYP", "FINANZTYP", "LP_FAMILIE_GROB", 
                 "LP_STATUS_GROB", "NATIONALITAET_KZ", "SHOPPER_TYP",
                 "ZABEOTYP", "GEBAEUDETYP", "CAMEO_DEUG_2015"]
    
    to_be_dropped = ["GFK_URLAUBERTYP", "LP_FAMILIE_FEIN", "LP_STATUS_FEIN", "CAMEO_DEU_2015"]
    
    df["OST_WEST_KZ"] = df["OST_WEST_KZ"].transform(lambda x: 0 if x == 'W' else 1)
    
    # Drop columns

    for drop in to_be_dropped:
        df.drop([drop], axis=1, inplace=True)
        
    # Encode columns

    for encode in to_be_encoded:
        df = dummy(azdias_processed, encode)
        
    
    df["40s"] = df["PRAEGENDE_JUGENDJAHRE"].isin([1, 2]).astype(int)
    df["50s"] = df["PRAEGENDE_JUGENDJAHRE"].isin([3, 4]).astype(int)
    df["60s"] =  df["PRAEGENDE_JUGENDJAHRE"].isin([6, 7]).astype(int)
    df["70s"] =  df["PRAEGENDE_JUGENDJAHRE"].isin([8, 9]).astype(int)
    df["80s"] =  df["PRAEGENDE_JUGENDJAHRE"].isin([10, 11, 12, 13]).astype(int)
    df["90s"] = df["PRAEGENDE_JUGENDJAHRE"].isin([14, 15]).astype(int)

    df["Mainstream"] = df["PRAEGENDE_JUGENDJAHRE"].isin([1, 3, 5, 8, 10, 12, 14]).astype(int)

    df["Mainstream"] = df["PRAEGENDE_JUGENDJAHRE"].isin([2, 4, 6, 7, 9, 11, 13, 15]).astype(int)

    df.drop(["PRAEGENDE_JUGENDJAHRE"], axis=1, inplace=True)
    
    df["Wealthy"] = df["CAMEO_INTL_2015"].isin([11, 12, 13, 14, 15]).astype(int)
    df["Prosperous"] = df["CAMEO_INTL_2015"].isin([11, 12, 13, 14, 15]).astype(int)
    df["Comfortable"] = df["CAMEO_INTL_2015"].isin([11, 12, 13, 14, 15]).astype(int)
    df["Less Affluent"] = df["CAMEO_INTL_2015"].isin([11, 12, 13, 14, 15]).astype(int)
    df["Poorer"] = df["CAMEO_INTL_2015"].isin([11, 12, 13, 14, 15]).astype(int)

    df["Pre-Family Couples & Singles"] = df["CAMEO_INTL_2015"].isin([11, 21, 31, 41, 51]).astype(int)
    df["Young Couples With Children"] = df["CAMEO_INTL_2015"].isin([12, 22, 32, 42, 52]).astype(int)
    df["Families With School Age Children"] = df["CAMEO_INTL_2015"].isin([13, 23, 33, 43, 53]).astype(int)
    df["Older Families &  Mature Couples"] = df["CAMEO_INTL_2015"].isin([14, 24, 34, 44, 54]).astype(int)
    df["Elders In Retirement"] = df["CAMEO_INTL_2015"].isin([15, 25, 35, 45, 55]).astype(int)

    df.drop(["CAMEO_INTL_2015"], axis=1, inplace=True)
    
    df["Very good neighborhood"] = (df["WOHNLAGE"] == 1).astype(int)
    df["Good neighborhood"] = (df["WOHNLAGE"] == 2).astype(int)
    df["Average neighborhood"] = (df["WOHNLAGE"] == 3).astype(int)
    df["Poor neighborhood"] = (df["WOHNLAGE"] == 4).astype(int)
    df["Very poor neighborhood"] = (df["WOHNLAGE"] == 5).astype(int)
    df["Rural neighborhood"] = (df["WOHNLAGE"] == 7).astype(int)
    df["New in rural neighborhood"] = (df["WOHNLAGE"] == 8).astype(int)

    df.drop(["CAMEO_INTL_2015"], axis=1, inplace=True)
    
    df.drop(["KBA05_BAUMAX", "PLZ8_BAUMAX"], axis=1, inplace=True)