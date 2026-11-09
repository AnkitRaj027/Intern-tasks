
import pandas as pd

def load_dataset(path):

    df = pd.read_csv(path)

    print("Dataset loaded successfully!")

    return df
