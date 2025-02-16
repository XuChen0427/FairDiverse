import os
import pandas as pd
import ast
from search.utils.process_tabular_data import create_train_split


class Compas:
    """
    A class to handle the COMPAS dataset, process it, and create train-test splits.
    """

    def __init__(self, config):
        """
        Initializes the COMPAS dataset handler.

        :param config: Dictionary containing dataset configuration parameters.
        """
        self.path = config["path"]
        self.data_path = os.path.join(config["path"], "COMPAS.csv")  # Dataset file path
        self.query_col = config["query_col"]  # Query column name
        self.ID = config["id_col"]  # Unique identifier column
        self.sensitive_cols = config["sensitive_cols"]  # Sensitive attribute columns
        self.score_col = config["score_col"]  # Score column
        self.feature_cols = config["feature_cols"]  # Feature columns
        self.k_fold = config["k_fold"]  # Number of folds for cross-validation
        self.ratio_split = config["ratio_split"]  # Train-test split ratio
        self.pos_th = config["pos_th"]  # Positive threshold for classification

        self.processed_file = os.path.join(self.path, "COMPAS_processed.csv")

        # Process the data during initialization
        self.process_data()

    def process_data(self):
        """
        Processes the dataset and creates train-test splits.
        """

        if not os.path.exists(self.processed_file):
            data = pd.read_csv(self.data_path)

            # Assign the same query ID to all rows (since there are no queries)
            data[self.query_col] = 0

            # Handle intersectional sensitive columns if applicable
            if "__" in self.sensitive_cols:
                intersectional_column = self.sensitive_cols[0]
                groups = intersectional_column.split("__")
                data[self.sensitive_cols] = data[groups].agg("__".join, axis=1)

            # Select relevant columns
            selected_columns = [self.query_col, self.ID, *self.feature_cols, self.score_col, *self.sensitive_cols]
            data = data[selected_columns]

            # Save the processed dataset
            data.to_csv(self.processed_file, index=False)

            # Create train-test splits
            split_path = os.path.join(self.path, "splits_data")
            os.makedirs(split_path, exist_ok=True)
            create_train_split(data, self.query_col, self.ID, self.sensitive_cols, self.score_col,
                               self.pos_th, split_path, self.k_fold, 1 - self.ratio_split)

    def read_data(self, i):
        """
        Reads the ith fold for train-test splits.

        :param i: The fold index (for k-fold cross-validation).
        :return: Tuple (data_train:Pandas DataFrame, data_test:Pandas DataFrames).
        """
        data = pd.read_csv(self.processed_file)

        # Paths to train-test split files
        split_path = os.path.join(self.path, "splits_data")
        train_file = os.path.join(split_path, f'train_samples_{i}.txt')
        test_file = os.path.join(split_path, f'test_samples_{i}.txt')

        # Load train-test IDs
        train_ids = self._load_ids(train_file)
        test_ids = self._load_ids(test_file)

        # Split dataset
        data_train = data[data[self.ID].astype(str).isin(train_ids)]
        data_test = data[data[self.ID].astype(str).isin(test_ids)]

        return data_train, data_test

    @staticmethod
    def _load_ids(file_path):
        """
        Helper method to read IDs from a text file.

        :param file_path: Path to the file containing IDs.
        :return: List of IDs.
        """
        with open(file_path, 'r') as f:
            return f.read().splitlines()
