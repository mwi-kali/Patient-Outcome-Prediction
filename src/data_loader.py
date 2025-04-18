import pandas as pd

from .config import DATASET_ID, RANDOM_STATE
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_data() -> pd.DataFrame:
    repo = fetch_ucirepo(id=DATASET_ID)
    X = repo.data.features
    y = repo.data.targets
    df = pd.concat([X, y], axis=1)
    return df


def split_data(df, test_size: float = 0.2):
    X = df.drop(columns='death_event')
    y = df['death_event'].astype(int)
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )