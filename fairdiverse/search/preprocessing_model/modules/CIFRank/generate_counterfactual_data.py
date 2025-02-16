import pandas as pd
import os

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

def get_counterfactual_data_real(data, model_path, configs, dataset):
    """
    Append to data the fair columns containing the counterfactual values
    computed based on the causal estimates saved at model_path

    Args:
        data (pandas.DataFrame): The current DataFrame containing the data to be transformed.
        model_path (str): Path to the output file where the model and predictions will be saved
        configs (dict): Configuration dict of the fairness method
        data_configs (dict): Configuration dict of the dataset

    Returns:
        pandas.DataFrame: The adjusted DataFrame with counterfactual columns added.
    """
    group_list = [x for x in data[configs['group']].unique() if x != configs['control']]

    mediators = pd.read_csv(os.path.join(model_path, "identified_mediators.csv"))
    no_mediators = len(mediators) == 0 or str(mediators['mediators'].values[0]) == 'nan'

    new_cols = []
    for med in dataset.feature_cols:
        if med in mediators['mediators'].values:
            x_res = pd.read_csv(os.path.join(model_path, med + "~" + configs['group'] + "-1.csv"))
            counter_g_base = x_res[x_res["Unnamed: 0"] == configs['group'] + configs['control']]["Estimate"].values[0]

            x_shifts = {configs['control']: 0}
            for gi in group_list:
                if not configs['group'] + gi in x_res["Unnamed: 0"].values:
                    x_shifts[gi] = 0
                else:
                    other_g_base = x_res[x_res["Unnamed: 0"] == configs['group'] + gi]["Estimate"].values[0]
                    x_shifts[gi] = counter_g_base - other_g_base

            feature_shifts = data[configs['group']].apply(lambda x: x_shifts[x])
            data.loc[:, med + "_fair"] = data[med] + feature_shifts
            new_cols.append(med + "_fair")

        else:
            # variables that are not mediators remain unchanged
            data.loc[:, med + "_fair"] = data[med]
            new_cols.append(med + "_fair")

    if no_mediators:
        # direct effect of the IV on the DV --> we keep the observed X as it is
        y_res = pd.read_csv(os.path.join(model_path, dataset.score_col + '~' + configs['group'] + "-1.csv"))
        counter_g_base = y_res[y_res["Unnamed: 0"] == configs['group'] + configs['control']]["Estimate"].values[0]
        y_shifts = {configs['control']: 0}
        for gi in group_list:
            if not configs['group'] + gi in y_res["Unnamed: 0"].values:
                y_shifts[gi] = 0
            else:
                y_shifts[gi] = counter_g_base - y_res[y_res["Unnamed: 0"] == configs['group'] + gi]["Estimate"].values[0]
    else:
        y_shifts = {configs['control']: 0}
        y_shifts_resolve = {configs['control']: 0}
        for gi in group_list:
            if not os.path.exists(os.path.join(model_path, gi + "_med" + ".csv")):
                y_shifts[gi] = 0
                y_shifts_resolve[gi] = 0
            else:
                g_res = pd.read_csv(os.path.join(model_path, gi + "_med" + ".csv"))
                y_shifts[gi] = -g_res[g_res['Metric'] == 'Total Effect']["Estimate"].values[0]
                y_shifts_resolve[gi] = -g_res[g_res['Metric'] == 'Direct Effect']["Estimate"].values[0]

    data["Y_shift"] = data[configs['group']].apply(lambda x: y_shifts[x])
    data[dataset.score_col + "_fair"] = data[dataset.score_col] + data["Y_shift"]
    new_cols.append(dataset.score_col + "_fair")

    return data