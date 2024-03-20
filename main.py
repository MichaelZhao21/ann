import os
import urllib.request
import zipfile
import pandas as pd
from src import model

# 10% of the data will be used for testing
TEST_SPLIT = 0.1
DATASET_NAME = "iris"
DATASET_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"


def main():
    """
    Main function runner
    """
    df = download_and_load_dataset()
    config = load_config_file()


def download_and_load_dataset() -> pd.DataFrame:
    """
    Downloads the dataset from the internet or loads it from the local cache

    :return: pandas DataFrame with the dataset
    """
    # Check if dataset exists
    if not os.path.exists("iris"):
        # Download dataset and unzip it
        with urllib.request.urlopen(DATASET_URL) as response:
            with open("iris.zip", "wb") as f:
                f.write(response.read())

        # Unzip the dataset
        with zipfile.ZipFile("iris.zip", "r") as zip_ref:
            zip_ref.extractall("iris")

    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(
        "iris/iris.data",
        names=["sepal-length", "sepal-width", "petal-length", "petal-width", "class"],
    )

    # Print info about dataframe
    print("===== Loaded Dataset Successfully =====")
    print(df)
    print("=======================================\n")

    return df


def load_config_file() -> dict:
    """
    Loads the configuration file

    :return: dictionary with the configuration
    """
    with open("config", "r") as f:
        lines = f.readlines()
        proc_lines = [
            line.strip()
            for line in lines
            if not line.startswith("#") and line.strip() != ""
        ]
        config = [
            {"act": a[0], "rate": float(a[1]), "iters": int(a[2])}
            for a in [line.split() for line in proc_lines]
        ]
        return config


if __name__ == "__main__":
    # main()
    load_config_file()
