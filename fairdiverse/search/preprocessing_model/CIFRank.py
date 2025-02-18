"""
@article{yang2020causal,
  title={Causal intersectionality for fair ranking},
  author={Yang, Ke and Loftus, Joshua R and Stoyanovich, Julia},
  journal={arXiv preprint arXiv:2006.08688},
  year={2020}
}

License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
from modules.CIFRank.run_causal_model import run_causal_model
from modules.CIFRank.generate_counterfactual_data import get_counterfactual_data_real
from fair_model import PreprocessingFairnessIntervention

class CIFRank(PreprocessingFairnessIntervention):
    """
        CIFRank class implements a fairness intervention method using causal intersectionality.

        This class extends the PreprocessingFairnessIntervention class and applies a causal fairness
        ranking model to the given dataset.
    """
    def __init__(self, configs, dataset):
        """
                Initialize the CIFRank model with configuration settings and dataset information.

                :param configs : dict
                    Configuration dictionary containing model parameters.
                :param dataset : str
                    The name or path of the dataset to be processed.
        """
        super().__init__(configs, dataset)

    def fit(self, X_train, run):
        """
        Train the causal fairness ranking model on the given training dataset.

        This method ensures that the necessary model directory exists and then runs the causal model.
        The trained model is saved in `self.model_path`.

        :param X_train : pandas.DataFrame
            The training dataset.
        :param run : str
            The identifier for the training run.

        :return : None
        """
        if not os.path.exists(os.path.join(self.model_path, run)):
            os.makedirs(os.path.join(self.model_path, run))
            run_causal_model(X_train, self.configs, self.dataset, os.path.join(self.model_path, run))

    def transform(self, X, run, file_name=None):
        """
        Apply the fairness transformation to the dataset by generating fair counterfactual columns.

        Fair columns are generated using a naming convention `<column_name>_fair` and saved to a CSV file
        if it does not already exist.

        :param X : pandas.DataFrame
            The dataset to which the fairness method should be applied.
        :param run : str
            The identifier for the transformation run.
        :param file_name : str, optional
            Name of the file to save the transformed dataset.

        :return : pandas.DataFrame
            The dataset with appended fair columns.
        """
        file_path = os.path.join(self.fair_data_path, run, f'fair_{file_name}_data.csv')
        if not os.path.exists(file_path):
            data_fair = self.generate_counterfactual_data(X, run)
            if file_name !=None:
                os.makedirs(os.path.join(self.fair_data_path, run), exist_ok=True)
                data_fair.to_csv(file_path)

        return data_fair


    def generate_counterfactual_data(self, X, run):
        """Generates the fair data by using the causal estimates
        Args:
            data (pandas.Dataframe): data on which to append the fair columns

        Returns:
            data (pandas.Dataframe): data containing the appended fair columns which contain
            the counterfactual values computed based on the causal estimates
        """
        path_causal = os.path.join(self.model_path, run)
        if os.path.exists(path_causal):
            if len(os.listdir(path_causal)):
                fair_df = get_counterfactual_data_real(X, path_causal, self.configs, self.dataset)

        return fair_df