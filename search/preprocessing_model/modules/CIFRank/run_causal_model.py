import os
from pathlib import Path

import pandas as pd

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
project_dir = Path.cwd()
def run_causal_model(data, configs, dataset, model_path):
    """
    Runs the causal model estimation on the data and saves the estimates at self.model_path
    Args:
        data (pandas.Dataframe): Data to be used for estimating the causal effects of the sensitive attributes on the data.
        configs (dict): Configuration dict of the fairness method.
    """
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri

    temp = data[data[dataset.score_col] > dataset.pos_th]
    temp = temp[[dataset.query_col, dataset.ID, configs['group'],
                 dataset.score_col] + dataset.feature_cols]

    try:
        pandas2ri.activate()
        r = robjects.r
        r_script = os.path.join(project_dir, "search", "preprocessing_model", "modules", "CIFRank",
                                "R", "estimate_causal_model.R")


        if not os.path.exists(os.path.join(model_path)):
            os.makedirs(os.path.join(model_path))

        r.source(r_script, encoding="utf-8")
        r.estimate_causal_model(temp, configs['group'], dataset.score_col,
                                dataset.feature_cols, configs['control'],
                                os.path.join(model_path))
    except Exception as e:
        with open(os.path.join(model_path, "logs.txt"), 'a') as f:
            f.write("error in causal model: " + str(e) + "\n")

        if len(os.listdir(model_path)) != 0:
            df = pd.DataFrame(columns=["Mediators"])
            df["Mediators"] = 'nan'
            df.to_csv(os.path.join(model_path, 'identified_mediators.csv'))
    groups = temp[configs['group']].unique()
    save_med_results(groups, configs['control'], os.path.join(model_path))


def save_med_results(groups, control, out_path):
    """Save the output of the mediation analysis.
    Args:
        groups (list(str)): List containing the values of the sensitive column of the data.
        control (str): Control value used in the causal estimation.
        out_path (str): Path to the output file where the mediation analysis is saved.
    """
    if os.path.exists(os.path.join(out_path, 'med_output.txt')):
        with open(os.path.join(out_path, 'med_output.txt'), 'r') as f:
            content = f.readlines()
        results_dict = dict()
        next_indirect = False
        for line in content:
            line = line.strip()
            if line.startswith('For the predictor'):
                if len(results_dict.keys()) == 0:
                    pred = line.split(' ')[3]
                    df_med = pd.DataFrame(columns=['Metric', 'Estimate'])
                    results_dict[pred] = ''
                else:
                    results_dict[pred] = df_med
                    pred = line.split(' ')[3]
                    df_med = pd.DataFrame(columns=['Metric', 'Estimate'])

            if line.startswith('The estimated total effect:'):
                total_effect = float(line.split(' ')[4])
                temp_df = pd.DataFrame([['Total Effect', total_effect]], columns=['Metric', 'Estimate'])
                df_med = pd.concat([df_med, temp_df], ignore_index=True)

            if next_indirect:
                splits = line.split(' ')
                if splits[0] == '':
                    indirect_effect = float(line.split(' ')[1])
                else:
                    indirect_effect = float(line.split(' ')[0])
                temp_df = pd.DataFrame([['Indirect Effect', indirect_effect]], columns=['Metric', 'Estimate'])
                df_med = pd.concat([df_med, temp_df], ignore_index=True)
                next_indirect = False

            if line.startswith('y1.all'):
                next_indirect = True

        results_dict[pred] = df_med

        pred_groups = [p.split('pred')[1] for p in results_dict.keys()]
        pred_gr = [g for g in groups if g not in pred_groups and g != control][0]

        index = 0
        for key in results_dict.keys():
            index = index + 1
            df_med = results_dict[key]
            direct_effect = df_med[df_med['Metric'] == 'Total Effect']['Estimate'].values[0] - \
                            df_med[df_med['Metric'] == 'Indirect Effect']['Estimate'].values[0]
            temp_df = pd.DataFrame([['Direct Effect', direct_effect]], columns=['Metric', 'Estimate'])
            df_med = pd.concat([df_med, temp_df], ignore_index=True)

            if key == 'pred' or key == '':
                file_name = pred_gr + '_med.csv'
            elif 'pred.temp1$x' in key:
                file_name = groups[index] + '_med.csv'
            else:
                file_name = key.split('pred')[1] + '_med.csv'

            df_med.to_csv(os.path.join(out_path, file_name))