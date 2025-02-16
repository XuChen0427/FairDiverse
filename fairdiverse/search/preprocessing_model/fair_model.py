import os
from abc import ABC, abstractmethod
from pathlib import Path

class PreprocessingFairnessIntervention(ABC):
    """
        Abstract base class for preprocessing fairness interventions.

        This class serves as a base for various fairness intervention methods that preprocess datasets
        to reduce biases while maintaining data utility.
    """
    def __init__(self, configs, dataset):
        """
            Initialize the fairness intervention with the given configurations and dataset.

            This method sets up the necessary directories for storing model parameters and
            fair data transformations.

            :param configs : dict
                Configuration dictionary containing model parameters and paths.
            :param dataset : object
                Dataset object containing path information.

            Attributes:
            - self.dataset : object
                Stores the dataset object for reference.
            - self.configs : dict
                Stores the configuration parameters.
            - self.model_path : Path
                Directory path where model parameters will be stored.
            - self.fair_data_path : Path
                Directory path where transformed fair data will be stored.

            Note:
            - The necessary directories are automatically created if they do not exist.
        """
        self.dataset = dataset
        self.configs = configs
        self.model_path = Path(self.dataset.path) / self.configs["name"] / "model_params"
        os.makedirs(self.model_path, exist_ok=True)
        self.fair_data_path = Path(self.dataset.path) / self.configs["name"] / "fair_data"
        os.makedirs(self.fair_data_path, exist_ok=True)
    @abstractmethod
    def fit(self, X_train, run):
        """
        Train the fairness model using the given training dataset.

        :param X_train : pandas.DataFrame
            The training dataset.
        :param run : str
            The identifier for the training run when using k-fold.

        :return : None
        """
        pass

    @abstractmethod
    def transform(self, X, run, file_name=None):
        """
            Apply the transformation to the dataset using the learned model.

            :param X : pandas.DataFrame
                The dataset to which the fairness transformation is applied.
            :param run : str
                The identifier for the transformation run.
            :param file_name : str, optional
                Name of the file to save the transformed dataset.

            :return : pandas.DataFrame
                The dataset with transformed fair columns.
        """
        pass

    def fit_transform(self, X_train, run):
        """
        Learns the model from the training data and returns the data in the new space.
        """
        print('Fitting and transforming...')
        self.fit(X_train, run)
        return self.transform(X_train, run)