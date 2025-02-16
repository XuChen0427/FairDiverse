import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from search.preprocessing_model.modules.gFair.loss import gFair_optimisation as gfair_func
from search.preprocessing_model.modules.probabilistic_mapping_helpers import compute_X_hat, compute_euclidean_distances
from search.preprocessing_model.utils import process_data_input, process_data_output, save_model_data, load_model_data
from search.preprocessing_model.fair_model import PreprocessingFairnessIntervention


class gFair(PreprocessingFairnessIntervention):
    """
        gFair is a fairness intervention method based on optimization techniques.

        This class extends PreprocessingFairnessIntervention and applies group fairness constraints to data
        using probabilistic mapping and distance-based optimization.
    """
    def __init__(self, configs, dataset):
        """
            Initialize the gFair model with the given configuration settings and dataset.

            :param configs : dict
                Configuration dictionary containing model parameters.
            :param dataset : str
                The name or path of the dataset to be processed.
        """
        super().__init__(configs, dataset)

    def fit(self, X_train, run):
        """
        Train the gFair fairness model using the given training dataset.

        This method applies optimization to learn group fairness constraints and stores the results
        for later use.

        :param X_train : pandas.DataFrame
            The training dataset. The last column is expected to be the protected attribute.
        :param run : str
            The identifier for the training run.

        :return : None
        """

        if not os.path.exists(os.path.join(self.model_path, run)):
            X_train, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
                process_data_input(X_train, self.configs, self.dataset))

            if self.configs["seed"] is not None:
                np.random.seed(self.configs["seed"])

            print('Fitting gFair_module...')
            indices = list(range(0, X_train.shape[1]))
            nonsensitive_column_indices = [index for index in indices if index not in sensitive_column_indices]
            D_X_F = compute_euclidean_distances(X_train, nonsensitive_column_indices)
            l = len(nonsensitive_column_indices)

            P = X_train.shape[1]
            min_obj = None
            opt_params = None

            for i in range(self.configs["nb_restarts"]):
                x0_init = np.random.uniform(size=int(P + P * self.configs["k"]))
                # setting protected column weights to epsilon
                ## assumes that the column indices from l through P are protected and appear at the end
                for i in range(l, P, 1):
                    x0_init[i] = 0.0001
                bnd = [(0, 1)] * P + [(None, None)] * P * self.configs["k"]

                gfair_func.iters = 0
                opt_result = minimize(gfair_func, x0_init,
                                      args=(X_train, sensitive_groups, sensitive_column_indices, D_X_F,
                                            group_weights, self.configs["biggest_gap"], os.path.join(self.model_path, run),
                                            self.configs["k"], self.configs["A_x"], self.configs["A_z"], self.configs["A_igf"],
                                            ),
                                      method='L-BFGS-B',
                                      jac=False,
                                      bounds=bnd,
                                      options={'maxiter': self.configs["max_iter"],
                                               'maxfun': self.configs["maxfun"],
                                               'eps': 1e-3})

            if (min_obj is None) or (opt_result.fun < min_obj):
                min_obj = opt_result.fun
                opt_params = opt_result.x

            self.opt_params = opt_params
            save_model_data(self, os.path.join(self.model_path, run))

        else:
            self.opt_params = load_model_data(os.path.join(self.model_path, run))


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

            X_hat, _ = compute_X_hat(X_np, self.opt_params, self.configs["k"], alpha=True)
            X_fair = process_data_output(X, X_hat, self.dataset, nonsensitive_column_indices, fair_data_path, file_name)
        else:
            X_fair = pd.read_csv(fair_data_file)
        return X_fair
