import pandas as pd

def get_data(path) -> pd.DataFrame:
    return pd.read_csv(path)