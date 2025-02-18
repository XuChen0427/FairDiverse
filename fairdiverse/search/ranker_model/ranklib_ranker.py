import os
import subprocess
from pathlib import Path
import pandas as pd

from .ranker import Ranker
from ..utils.process_tabular_data import norm_features, check_nan, writeToTXT


class RankLib(Ranker):
    """
    Wrapper class to run the available ranking models in the Ranklib library.
    For more information about available models and params check the official documentation: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
    """
    def __init__(self, configs, dataset):
        """
        Initialize the RankLib model with configuration settings and dataset.

        :param configs : dict
            The configuration dictionary that contains hyperparameters and paths needed for training.
        :param dataset : object
            The dataset object containing the necessary columns like feature columns and target variable.
        """
        super().__init__(configs, dataset)

    def train(self, data_train, data_test, run):
        """
        Trains ranking models using RankLib.

        This method generates RankLib-compatible training data and then runs the RankLib training script.

        :param data_train : pandas.DataFrame
            The training dataset to be used for training the ranking model.
        :param data_test : pandas.DataFrame
            The testing dataset to be used for evaluating the ranking model.
        :param run : str
            The identifier for the current training run.
        """
        self.generate_ranklib_data(data_train, data_test, run)
        for experiment in self._get_experiments(run):
            model_path = self.out_dir / str(run) / experiment / self.configs['ranker']
            if not model_path.exists():
                self._run_ranklib_training(experiment, run)

    def predict(self, data, run, file_name):
        """
        Generates predictions using the trained RankLib model.

        This method reads the predictions from the trained model and saves them as a CSV file.

        :param data : pandas.DataFrame
            The dataset on which predictions need to be made.
        :param run : str
            The identifier for the current run.
        :param file_name : str
            The file name to save the predictions as a CSV.

        :return : pandas.DataFrame
            A DataFrame containing the predictions.
        """
        predictions = self.read_predictions(data, run)
        pred_dir_path = os.path.join(self.out_dir, str(run), "predictions")
        os.makedirs(pred_dir_path, exist_ok=True)
        predictions.to_csv(os.path.join(pred_dir_path, f'{file_name}_pred.csv'))
        return predictions

    def _get_experiments(self, run):
        """
        Fetches all experiment directories that contain '__' in the name.

        :param run : str
            The identifier for the run.

        :return : list
            A list of experiment directory names.
        """
        return [f for f in os.listdir(self.out_dir / str(run)) if "__" in f]

    def _run_ranklib_training(self, experiment, run):
        """
        Executes the RankLib training script.

        This method calls an external script to train the ranking model with the provided configurations.

        :param experiment : str
            The name of the current experiment.
        :param run : str
            The identifier for the current run.
        """
        project_dir = Path.cwd()
        try:
            subprocess.check_call([
                str(project_dir / self.ranker_path / "run-LTR-model.sh"),
                str(project_dir / self.ranker_path),
                str(self.configs['metric']),
                str(self.configs['top_k']),
                str(self.configs['rel_max']),
                str(self.configs['ranker']),
                str(self.configs['ranker_id']),
                str(self.out_dir / str(run) / experiment),
                str(self.configs['lr']),
                str(self.configs['epochs']),
                "none"
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(f"Command output: {e.output}")

    def generate_ranklib_data(self, data_train, data_test, run):
        """
        Generates data formatted for RankLib training and testing.

        This method prepares the training and testing data for RankLib by generating the required feature matrix
        and label information in a format that RankLib can process.

        :param data_train : pandas.DataFrame
            The training dataset.
        :param data_test : pandas.DataFrame
            The testing dataset.
        :param run : str
            The identifier for the current run.
        """
        experiments = list(zip(self.configs['train_data'], self.configs['test_data']))

        for train_name, test_name in experiments:
            data_train_copy = data_train.copy()
            data_test_copy = data_test.copy()

            out_dir = self.out_dir / str(run) / f"{train_name}__{test_name}"
            if not os.path.exists(out_dir):
                out_dir.mkdir(parents=True, exist_ok=True)

                cols_train = self._get_feature_columns(train_name)
                cols_test = self._get_feature_columns(test_name)

                self._validate_data(data_train_copy, cols_train, "train")
                self._validate_data(data_test_copy, cols_test, "test")

                data_train_copy = self._process_data(data_train_copy, cols_train)
                data_test_copy = self._process_data(data_test_copy, cols_test)

                self.create_ranklib_data(cols_train, data_train_copy, out_dir, "train")
                self.create_ranklib_data(cols_test, data_test_copy, out_dir, "test")

    def _get_feature_columns(self, mode):
        """
        Returns feature columns with fairness suffix if applicable.

        :param mode : str
            The mode can be "fair" or another value. If "fair", fairness-related features will be included.

        :return : list
            A list of feature column names.
        """
        suffix = "_fair" if mode == "fair" else ""
        return [col + suffix for col in self.dataset.feature_cols] + [self.dataset.score_col + suffix]

    def _validate_data(self, data, cols, split):
        """
        Checks for NaN values in the dataset.

        This method ensures that there are no missing values in the data before further processing.

        :param data : pandas.DataFrame
            The dataset to be validated.
        :param cols : list
            A list of column names to check for NaN values.
        :param split : str
            The split type, either "train" or "test".

        :raises ValueError : if NaN values are detected.
        """
        if check_nan(data, cols):
            raise ValueError(f"NaN values detected in {split} data!")

    def _process_data(self, data, cols):
        """
        Assigns judgement scores and normalizes features.

        This method applies judgement scoring based on relevance and normalizes the features for RankLib.

        :param data : pandas.DataFrame
            The dataset to be processed.
        :param cols : list
            The list of feature column names.

        :return : pandas.DataFrame
            The processed dataset with normalized features.
        """
        data = data.groupby(self.dataset.query_col).apply(
            lambda x: self.assign_judgement(x, self.dataset.pos_th, cols)).reset_index(drop=True)
        return norm_features(cols, data)

    def create_ranklib_data(self, cols, data, out_dir, split):
        """
        Formats and writes data for RankLib.

        This method prepares the data by formatting it according to RankLib's required format and writes it
        to a text file.

        :param cols : list
            The list of feature columns.
        :param data : pandas.DataFrame
            The data to be written to a text file.
        :param out_dir : Path
            The output directory where the file will be saved.
        :param split : str
            The type of data, either "train" or "test".
        """
        data = self._format_ranklib_data(cols, data)
        output_f = out_dir / f"{split}_ranklib.txt"
        writeToTXT(output_f, data)

    def _format_ranklib_data(self, cols, data):
        """
        Formats features for RankLib.

        This method ensures that the features are formatted in a way RankLib can consume, and shuffles
        the data within query groups.

        :param cols : list
            The list of feature columns.
        :param data : pandas.DataFrame
            The dataset to be formatted.

        :return : pandas.DataFrame
            The formatted dataset ready for RankLib.
        """
        for idx, col in enumerate(cols):
            data[col] = data[col].apply(lambda x: f"{idx + 1}:{round(x, 4)}")

        data["QID"] = data["QID"].apply(lambda x: f"qid:{x}")
        data["UID"] = data[["UID", "judgement", cols[-1]]].astype(str).apply(
            lambda x: f"#docid={x.iloc[0]};rel={x.iloc[1]};{cols[-1]}={x.iloc[2]};",
            axis=1
        )

        # Shuffle within query groups
        data = pd.concat([g.sample(frac=1) for _, g in data.groupby("QID")]).reset_index(drop=True)

        return data[["judgement", "QID"] + cols + ["UID"]]

    def assign_judgement(self, x, th, cols):
        """
        Assigns judgement scores based on relevance ranking.

        This method assigns a judgement score to each document based on its relevance to a query.

        :param x : pandas.DataFrame
            The subset of data belonging to a single query.
        :param th : float
            The threshold for classifying relevance.
        :param cols : list
            The list of feature columns.

        :return : pandas.DataFrame
            The data with the assigned judgement scores.
        """
        mask_pos = x[self.dataset.score_col].apply(lambda x: round(x, 2) > th)
        pos_x, neg_x = x[mask_pos], x[~mask_pos]

        pos_x['judgement'] = pos_x[cols[-1]].rank(ascending=True, method='dense')
        pos_x['judgement'] = self._scale_judgements(pos_x)

        neg_x['judgement'] = 0
        return pd.concat([pos_x, neg_x])

    def _scale_judgements(self, pos_x):
        """
        Scales and rounds judgement values.

        This method scales the judgement values to a defined range and ensures all values are within the valid range.

        :param pos_x : pandas.DataFrame
            The data with the initial judgement scores.

        :return : pandas.Series
            The scaled judgement scores.
        """
        min_rank, max_rank = pos_x['judgement'].min(), pos_x['judgement'].max()
        pos_x['judgement'] = ((pos_x['judgement'] - min_rank) / (max_rank - min_rank + 1)) * self.configs['rel_max']
        pos_x['judgement'] = pos_x['judgement'].round().astype(int)

        # Ensure values are within valid range
        if pos_x['judgement'].max() < self.configs['rel_max']:
            pos_x['judgement'] += self.configs['rel_max'] - pos_x['judgement'].max()
        pos_x['judgement'] = pos_x['judgement'].clip(lower=1)

        return pos_x['judgement']

    def read_predictions(self, data, run):
        """
        Retrieves LTR predictions for the dataset.

        This method loads the predictions from the trained RankLib model.

        :param data : pandas.DataFrame
            The dataset for which predictions need to be made.
        :param run : str
            The identifier for the run.

        :return : pandas.DataFrame
            The dataset with predictions added.
        """
        pred_dir = self.out_dir / str(run)
        predictions = get_LTR_predict(data, pred_dir, self.configs['ranker'], self.dataset.score_col,
                                      self.dataset.query_col, self.dataset.ID)
        return predictions


def get_LTR_predict(data, out_dir, ranker, score_col, query_col, id_col):
    """
    Fetches RankLib prediction scores.

    This method loads prediction scores from the model and merges them with the provided dataset.

    :param data : pandas.DataFrame
        The dataset that needs the predictions.
    :param out_dir : Path
        The directory where the RankLib predictions are stored.
    :param ranker : str
        The name of the ranking model used.
    :param score_col : str
        The column name of the score in the dataset.
    :param query_col : str
        The column representing queries.
    :param id_col : str
        The unique identifier for each data point.

    :return : pandas.DataFrame
        The dataset with added prediction scores.
    """


    experiments = [f for f in os.listdir(out_dir) if "__" in f]

    for experiment in experiments:
        train_set, test_set = experiment.split("__")

        train_set = f"_{train_set}" if train_set == "fair" else ""
        test_set = f"_{test_set}" if test_set == "fair" else ""

        pred_col = f"{score_col}{train_set}__{score_col}{test_set}"

        score_pred = get_prediction_scores(out_dir / experiment / ranker)
        data = data[data[id_col].astype(str).isin(score_pred)]
        data[pred_col] = data[id_col].apply(lambda x: score_pred.get(str(x), 0))
    return data


def get_prediction_scores(pred_path):
    """
    Retrieves prediction scores from the latest RankLib experiment.

    This method reads the predictions generated from the latest experiment and returns them.

    :param pred_path : Path
        The directory containing the prediction files.

    :return : dict
        A dictionary mapping document IDs to predicted scores.
    """


    sub_experiments = [x for x in os.listdir(pred_path) if "experiments_" in x]
    if not sub_experiments:
        raise ValueError(f"No predictions found in {pred_path}!")

    latest_exp = max(sub_experiments, key=lambda x: os.path.getmtime(os.path.join(pred_path, x)))
    pred_file = Path(pred_path) / latest_exp / "predictions" / "prediction.txt"

    if pred_file.exists():
        print(f"**** Reading predictions from {pred_file}")
        with pred_file.open("r") as file:
            lines = file.read().splitlines()
            return {li.split(" ")[2].split(";")[0].replace("docid=", ""): int(li.split(" ")[3]) for li in lines}

    raise ValueError(f"Prediction file not found in {pred_path}!")
