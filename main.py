import os
import urllib.request
import zipfile
import pandas as pd

# 10% of the data will be used for testing
TEST_SPLIT = 0.1
DATASET_NAME = "iris"
DATASET_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"


def main():
    df = download_and_load_dataset()


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


if __name__ == "__main__":
    main()
