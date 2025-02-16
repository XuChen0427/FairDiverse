"""Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.
    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.
    Based on code from https://github.com/zjelveh/learning-fair-representations
    """

import numpy as np
import pandas as pd
import scipy.optimize as optim
import os
from search.preprocessing_model.modules.LFR.loss import LFR_optimisation as LFR_func
from search.preprocessing_model.modules.probabilistic_mapping_helpers import compute_X_hat
from search.preprocessing_model.utils import process_data_input, process_data_output, save_model_data, load_model_data
from search.preprocessing_model.fair_model import PreprocessingFairnessIntervention
class LFR(PreprocessingFairnessIntervention):
    """
        Learning Fair Representations (LFR) fairness intervention.

        This class applies the LFR approach to modify the dataset such that fairness constraints
        are met while preserving as much utility as possible.
        """
    def __init__(self, configs, dataset):
        """
        Initialize the LFR model with the given configurations and dataset.

        :param configs : dict
            Configuration dictionary containing model parameters.
        :param dataset : str
            The dataset to be processed.
        """
        super().__init__(configs, dataset)
    def fit(self, X_train, run):
        """
        Train the LFR fairness model using the given training dataset.

        This method optimizes a fairness objective by learning fair representations
        of the data using constrained optimization.

        :param X_train : pandas.DataFrame or numpy.ndarray
            The training dataset. It is assumed that the last non-sensitive column is the target variable.
        :param run : str
            The identifier for the training run.

        :return : self
            The trained LFR model.
        """
        if not os.path.exists(os.path.join(self.model_path, run)):
            X_train, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
                process_data_input(X_train, self.configs, self.dataset))

            if self.configs["seed"] is not None:
                np.random.seed(self.configs["seed"])

            # assumes that the last non-sensitive column of X_train is Y_train


            Y_train = X_train[:, nonsensitive_column_indices][-1]
            features_dim = X_train.shape[1]

           # Initialize the LFR_module optim objective parameters
            parameters_initialization = np.random.uniform(size=int(self.configs["k"] + features_dim * self.configs["k"]))

            bnd = [(0, 1)] * self.configs["k"] + [(None, None)] * features_dim * self.configs["k"]
            LFR_func.steps = 0


            self.opt_params = optim.fmin_l_bfgs_b(LFR_func, x0=parameters_initialization, epsilon=1e-5,
                                                          args=(X_train, Y_train, sensitive_groups, sensitive_column_indices, self.configs["k"],
                                                                self.configs["A_x"], self.configs["A_y"], self.configs["A_z"],
                                                                group_weights, self.configs["biggest_gap"],
                                                                os.path.join(self.model_path, run)),
                                                          bounds=bnd, approx_grad=True, maxfun=self.configs["maxfun"],
                                                          maxiter=self.configs["maxiter"], disp=False)[0]
            self.w = self.opt_params[:self.configs["k"]]
            self.prototypes = self.opt_params[self.configs["k"]:].reshape((self.configs["k"], features_dim))

            save_model_data(self, os.path.join(self.model_path, run))
        else:
            self.opt_params = load_model_data(os.path.join(self.model_path, run))
        return self

    def transform(self, X, run, file_name=None):
        """
            Apply the fairness transformation to the dataset using the learned model.

            This method ensures fairness by adjusting feature distributions while maintaining data utility.

            :param X : pandas.DataFrame
                The dataset to which the fairness transformation is applied.
            :param run : str
                The identifier for the transformation run.
            :param file_name : str, optional
                Name of the file to save the transformed dataset.

            :return : pandas.DataFrame
                The dataset with transformed fair columns.
        """
        fair_data_path = os.path.join(self.fair_data_path, run)
        os.makedirs(fair_data_path, exist_ok=True)
        fair_data_file = os.path.join(fair_data_path, f'fair_{file_name}_data.csv')
        if not os.path.exists(fair_data_file):
            X_np, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
                process_data_input(X, self.configs, self.dataset))

            X_hat, _ = compute_X_hat(X_np, self.opt_params, self.configs["k"], alpha=False)
            X_fair = process_data_output(X, X_hat, self.dataset, nonsensitive_column_indices, fair_data_path, file_name)
        else:
            X_fair = pd.read_csv(fair_data_file)
        return X_fair


