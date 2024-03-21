import os
import urllib.request
import zipfile
import pandas as pd
from src.model import Model

DATASET_NAME = "iris"
DATASET_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"

# 10% of the data will be used for testing
TEST_SPLIT = 0.1
LAYERS = [4, 8, 6, 3]


def main():
    """
    Main function runner
    """
    df = download_and_load_dataset()
    config = load_config_file()
    output_data = run_tests(df, config)
    save_and_display_output(output_data)


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

        # Get a count of each activation function
        act_counts = { "sigmoid": 0, "tanh": 0, "relu": 0 }
        for c in config:
            act_counts[c["act"]] += 1

        # Print info about configuration
        print("===== Loaded Configuration File =====")
        print('Number of tests:', len(config))
        print('Average number of iterations:', sum([c['iters'] for c in config]) / len(config))
        print('Average learning rate:', sum([c['rate'] for c in config]) / len(config))
        print('Activation function counts:', act_counts)
        print("=====================================\n")

        return config


def run_tests(df: pd.DataFrame, config: dict):
    """
    Runs the tests with the given configuration

    :param df: pandas DataFrame with the dataset
    :param config: dictionary with the configuration
    """
    output_data = []

    print("===== Running Tests... =====\n")

    # Loop through each training example
    for i, c in enumerate(config):
        # Print info
        print(f'===== Test {i}/{len(config)} =====')

        # Create a model with the given configuration
        m = Model(
            activation=c["act"],
            learn_rate=c["rate"],
            iterations=c["iters"],
            layers=LAYERS,
            test_split=TEST_SPLIT,
        )
        m.pre_process(df)

        # Train and test the model
        m.train()
        m.test()

        # Save the results to the output data list
        output_data.append((c['act'], c['rate'], c['iters'], m.train_acc, m.test_acc))

        # Print info
        print("=======================================\n")

    return output_data


def save_and_display_output(output_data):
    """Saves the output to a csv file and displays it in a tabular format."""
    # Save the output to a csv file
    df = pd.DataFrame(
        output_data,
        columns=["activation", "learning_rate", "iterations", "train_acc (%)", "test_acc (%)"],
    )
    df.to_csv("output.csv", index=False)

    # Print info
    print("===== Saved Output to output.csv =====")
    print(df)
    print("=====================================\n")

    # Create a table from the df
    down = df.to_markdown()
    with open("output.md", "w") as f:
        f.write(down)


if __name__ == "__main__":
    main()
