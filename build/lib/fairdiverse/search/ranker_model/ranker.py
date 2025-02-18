import os
from abc import ABC, abstractmethod
from pathlib import Path
from ..fairness_evaluator import evaluate, evaluate_runs


class Ranker(ABC):
    """
    Ranker class for training and evaluating ranking models.

    This abstract class provides methods for training ranking models and evaluating their fairness
    using the provided dataset and configuration settings.

    :param configs : dict
        Configuration dictionary containing paths and settings for the model and evaluation.
    :param dataset : object
        The dataset object containing data and metadata for training and evaluation.
    """

    def __init__(self, configs, dataset):
        """
        Initializes the Ranker class with the given configurations and dataset.

        :param configs : dict
            Configuration dictionary containing paths and settings for the model and evaluation.
        :param dataset : object
            The dataset object containing data and metadata for training and evaluation.
        """
        self.configs = configs
        self.dataset = dataset
        self.out_dir = Path(self.dataset.path) / self.configs["preprocessing_model"] / "ranker"
        self.ranker_path = Path(self.configs["ranker_path"])

    @abstractmethod
    def train(self, data_train, data_test):
        """
        Train the ranking model.

        This method is abstract and needs to be implemented in the subclasses to train a ranking model
        based on the provided training and testing data. The trained model is saved at a specific location.

        :param data_train : pandas.DataFrame
            The data to be used for training the ranking model.
        :param data_test : pandas.DataFrame
            The data to be used for testing the ranking model.

        :return : None
            This method does not return anything. It saves the trained model to a file.
        """
        pass

    @abstractmethod
    def predict(self, data, run, file_name):
        """
        Predict ranking scores for the given data.

        This method is abstract and should be implemented in subclasses to predict ranking scores
        based on the trained model. It appends the predicted scores to the provided data.

        :param data : pandas.DataFrame
            The data to be used for prediction.
        :param run : str
            The identifier for the specific run.
        :param file_name : str
            The name of the file where predictions will be saved.

        :return : pandas.DataFrame
            The data with an added column containing predicted ranking scores.
        """
        pass

    def evaluate_run(self, pred, run, file_name):
        """
        Evaluate the results of a specific run and compute fairness metrics.

        This method evaluates the predictions using fairness metrics for each sensitive attribute
        in the dataset and saves the evaluation results.

        :param pred : pandas.DataFrame
            The predictions to be evaluated.
        :param run : str
            The identifier for the specific run.
        :param file_name : str
            The name of the file containing predictions to evaluate.

        :return : None
            This method does not return anything. It saves the evaluation results to the file.
        """
        eval_path = Path(self.dataset.path) / self.configs["preprocessing_model"] / "evaluate" / str(run) / file_name
        print(eval_path)
        for s_attr in self.dataset.sensitive_cols:
            evaluate(pred, self.dataset.query_col, s_attr, eval_path, self.configs["evaluate"])

    def evaluate(self, file_name):
        """
        Evaluate multiple runs of the ranking model using fairness metrics.

        This method evaluates the model for all runs using the specified fairness metrics from the configuration.

        :param file_name : str
            The name of the file to evaluate.

        :return : None
            This method does not return anything. It performs the evaluation and saves the results.
        """
        runs = self.dataset.k_fold
        eval_path = Path(self.dataset.path) / self.configs["preprocessing_model"] / "evaluate"

        eval_files = [file for file in os.listdir(eval_path / "0" / file_name / self.configs["evaluate"]["metrics"][0])
                      if "QID" in file]
        qids = range(0, len(eval_files))
        evaluate_runs(self.configs["evaluate"], qids, eval_path, runs, file_name)
