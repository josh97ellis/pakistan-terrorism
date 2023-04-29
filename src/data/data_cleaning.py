from pathlib import Path
import pandas as pd
import numpy as np


def read_raw_data() -> pd.DataFrame:
    df = pd.read_csv(f"{Path(__file__).parent.parent.parent}/data/raw/pakistanClean.csv")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[
        'iyear', 'imonth', 'iday', 'provstate', 'city', 'latitude', 'longitude',
        'multiple', 'success', 'suicide', 'attacktype1_txt', 'targtype1_txt',
        'targsubtype1_txt', 'corp1', 'gname', 'weaptype1_txt', 'weapsubtype1_txt',
        'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte', 'property',
        'ishostkid']]
    return df


def replace_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace -9 with NA
    df = df.replace(-9, np.NaN)

    # Replace missing categorical data
    df['city'].fillna('unknown', inplace=True)
    df['targsubtype1_txt'].fillna('unknown', inplace=True)
    df['corp1'].fillna('unknown', inplace=True)
    df['weapsubtype1_txt'].fillna('Uknown Weapon Type', inplace=True)
    
    return df


def write_to_interim(df: pd.DataFrame) -> None:
    df.to_parquet(
        f"{Path(__file__).parent.parent.parent}/data/interim/pakistan_processed.parquet",
        compression='gzip',
        engine='pyarrow',
        index=False)


def main():
    data = read_raw_data()
    data_selected = select_columns(data)
    data_final = replace_missing_data(data_selected)
    write_to_interim(data_final)


if __name__ == '__main__':
    main()
