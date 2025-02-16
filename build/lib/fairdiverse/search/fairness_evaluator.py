import numpy as np
import pandas as pd
import math
import os
from sklearn.metrics import pairwise


def evaluate_runs(args_eval, qids, eval_path, runs, split):
    """
    Evaluate the runs by calculating metrics and saving the results in CSV files.

    :param args_eval : dict
        A dictionary containing evaluation configurations, including metrics and other settings.
    :param qids : list
        A list of unique query identifiers (QIDs) to evaluate.
    :param eval_path : str
        Path where evaluation results will be saved.
    :param runs : int
        Number of runs to evaluate.
    :param split : str
        The data split to evaluate (e.g., "train", "test").

    :return : None
        This function saves the evaluation results as CSV files.
    """
    for qid in qids:
        for eval_measure in args_eval['metrics']:
            output_f = os.path.join(eval_path, "Eval_QID_" + str(qid) + '_' + eval_measure + ".csv")

            if not os.path.exists(output_f):
                res_all = []
                for r in range(runs):
                    file_name = 'Eval_QID_' + str(qid) + '_' + eval_measure + '.csv'
                    path = os.path.join(eval_path, str(r), split, eval_measure, file_name)

                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        res_all.append(df)

                res_all = pd.concat(res_all)
                res_all = res_all.groupby(['rank', 'group', 'k']).mean().reset_index()
                res_all.to_csv(output_f)
                print("--- Save eval file in ", output_f, " --- \n")


def evaluate(data, query_col, s_attribute, eval_path, args_eval):
    """
    Evaluate the given data by calculating metrics for each query and sensitive attribute.

    :param data : pandas.DataFrame
        The data containing the query identifiers and sensitive attribute.
    :param query_col : str
        The column name for the query identifiers in the data.
    :param s_attribute : str
        The sensitive attribute column name.
    :param eval_path : str
        Path where evaluation results will be saved.
    :param args_eval : dict
        A dictionary containing evaluation configurations, including metrics and other settings.

    :return : None
        This function saves the evaluation results as CSV files.
    """
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    qids = data[query_col].unique()

    groups = data[s_attribute].unique()
    for eval_measure in args_eval['metrics']:
        if not os.path.exists(os.path.join(eval_path, eval_measure)):
            os.makedirs(os.path.join(eval_path, eval_measure))
            for qid in qids:
                data_qid = data[data[query_col] == qid]
                res_qid = evaluate_qid(data_qid, eval_measure, s_attribute, groups, args_eval)

                output_f = os.path.join(eval_path, eval_measure, "Eval_QID_" + str(qid) + "_" + eval_measure + ".csv")
                res_qid.to_csv(output_f)
                print("--- Save eval file in ", output_f, " --- \n")


def evaluate_qid(df, eval_measure, s_attribute, sensitive_groups, args_eval):
    """
    Evaluate the data for a single query ID, calculating the specified metrics.

    :param df : pandas.DataFrame
        The data for a single query ID.
    :param eval_measure : str
        The evaluation metric to calculate.
    :param s_attribute : str
        The sensitive attribute column name.
    :param sensitive_groups : list
        A list of sensitive attribute groups.
    :param args_eval : dict
        A dictionary containing evaluation configurations, including rankings and k list.

    :return : pandas.DataFrame
        A DataFrame containing the evaluation results.
    """
    EVAL_RANKINGS = args_eval['rankings']

    seti_quotas = get_quotas_count(df, s_attribute, sensitive_groups=sensitive_groups)

    res_df = pd.DataFrame(columns=["run", "rank", "k", "group", eval_measure])
    k_list = args_eval['k_list']

    for ranking in EVAL_RANKINGS:
        # data sorted by ranking
        ranking_df = get_sort_df(ranking, df, len(df))

        for ki in k_list:
            ki = int(ki)
            res_row = [1, ranking, ki]
            all_row = res_row + ["all"]

            # data sorted by ranking value at top-k
            top_ranking = ranking_df.head(ki)

            if 'individual' in eval_measure:
                if "__" in ranking:
                    yNN = compute_individual_fairness(ranking_df, ranking)
                    all_row.append(yNN)
                else:
                    # not applicable on the output of an LTR model
                    all_row.append(-1)

            if eval_measure == "select_rate":
                # applicable per group
                all_row.append(-1)

            if eval_measure == "diversity":
                # applicable per group
                all_row.append(-1)

            if eval_measure == "exposure":
                # applicable per group
                all_row.append(-1)

            if eval_measure == "igf":
                # applicable per group
                all_row.append(-1)

            res_df.loc[res_df.shape[0]] = all_row

            # group-level evaluation
            cur_quotas = get_quotas_count(top_ranking, s_attribute, sensitive_groups)
            for gi in sensitive_groups:
                gi_row = res_row + [gi]
                if eval_measure == "select_rate":
                    # selection rate to rank inside top-k
                    if gi in cur_quotas and seti_quotas[gi] != 0:
                        if seti_quotas[gi] < 1 / ki:  # at least one candidate in top-k
                            seti_quotas[gi] = 1 / ki
                        gi_row.append(cur_quotas[gi] / seti_quotas[gi])
                    else:
                        gi_row.append(0)

                if eval_measure == "diversity":
                    if gi in cur_quotas:
                        gi_row.append(cur_quotas[gi])
                    else:
                        gi_row.append(0)

                gi_top_ranking = top_ranking[top_ranking[s_attribute] == gi]
                gi_ranking_df = ranking_df[ranking_df[s_attribute] == gi]

                if eval_measure == "exposure":
                    gi_row.append(compute_cumulative_exposer(gi_top_ranking, ki))

                if eval_measure == "igf":
                    if gi_ranking_df is not None:
                        if not gi_top_ranking.shape[0]:
                            gi_row.append(-1)
                        else:
                            gi_row.append(compute_igf_ratio(list(gi_top_ranking["UID"]), gi_ranking_df, ranking))

                if 'individual' in eval_measure:
                    # not applicable to group
                    gi_row.append(-1)

                res_df.loc[res_df.shape[0]] = gi_row

    return res_df


def compute_individual_fairness(data, ranking):
    """
    Compute the individual fairness score for the ranking.

    :param data : pandas.DataFrame
        The data to evaluate fairness.
    :param ranking : str
        The ranking column to use for evaluation.

    :return : float
        The individual fairness score between 0 and 1.
    """
    feature_columns = [col for col in data if 'X' in col and '_' not in col]
    distances_data = pairwise.euclidean_distances(data[feature_columns].to_numpy(),
                                                  data[feature_columns].to_numpy())

    exposers = data[ranking].apply(lambda x: 1 / math.log2(x + 1))
    distances_exposer = pairwise.euclidean_distances(exposers.to_numpy().reshape(-1, 1),
                                                     exposers.to_numpy().reshape(-1, 1))

    yNN = 1 - np.mean(np.abs(distances_data - distances_exposer))
    return yNN


def compute_cumulative_exposer(df_top, ki):
    """
    Compute the cumulative exposure for the top-k items in the ranking.

    :param df_top : pandas.DataFrame
        The top-k items in the ranking.
    :param ki : int
        The number of top-k items.

    :return : float
        The cumulative exposure score.
    """
    if len(df_top) < ki:
        df_top["rank"] = list(range(1, len(df_top) + 1))
    else:
        df_top["rank"] = list(range(1, ki + 1))

    if len(df_top) == 0:
        return -1
    exposer_top_k = sum(df_top['rank'].apply(lambda x: 1 / math.log2(x + 1)))

    return exposer_top_k


def compute_igf_ratio(top_k_IDS, _orig_df, _orig_sort_col):
    """
    Compute the IGF (Item Group Fairness) ratio for the top-k items.

    :param top_k_IDS : list
        A list of IDs representing the top-k items.
    :param _orig_df : pandas.DataFrame
        The original DataFrame containing all items.
    :param _orig_sort_col : str
        The column name to sort the original DataFrame by.

    :return : float
        The IGF ratio.
    """
    accepted_candidates = _orig_df["UID"].isin(top_k_IDS)
    _lowest_accepted_score = min(_orig_df[accepted_candidates][_orig_sort_col])

    rejected_candidates = ~accepted_candidates
    if sum(rejected_candidates) == 0:
        return 1

    _highest_rejected_score = max(_orig_df[rejected_candidates][_orig_sort_col])

    if _highest_rejected_score == 0:
        return 1

    cur_res = min(_orig_df[_orig_df["UID"].isin(top_k_IDS)][_orig_sort_col]) / max(
        _orig_df[~_orig_df["UID"].isin(top_k_IDS)][_orig_sort_col])

    if cur_res > 1:
        return 1
    else:
        return cur_res


def get_quotas_count(_df, s_attribute, sensitive_groups):
    """
    Calculate the quota count for each sensitive group in the data.

    :param _df : pandas.DataFrame
        The data to calculate the quotas for.
    :param s_attribute : str
        The sensitive attribute column name.
    :param sensitive_groups : list
        A list of sensitive attribute groups.

    :return : dict
        A dictionary with the sensitive group names as keys and their corresponding quota counts as values.
    """
    res_dict = {}

    for s in sensitive_groups:
        mask = _df[s_attribute] == s
        res_dict[s] = sum(mask) / len(_df)

    return res_dict


def get_sort_df(_sort_col, _df, _k):
    """
    Sort the DataFrame by the specified column and return the top-k rows.

    :param _sort_col : str
        The column to sort the data by.
    :param _df : pandas.DataFrame
        The DataFrame to be sorted.
    :param _k : int
        The number of top items to return.

    :return : pandas.DataFrame
        The top-k sorted rows from the DataFrame.
    """
    _df[_sort_col] = _df[_sort_col].apply(lambda x: float(x))
    _k = int(_k)
    if "__" in _sort_col:
        sort_df = _df.sort_values(by=_sort_col, ascending=True).head(_k)
    else:
        sort_df = _df.sort_values(by=_sort_col, ascending=False).head(_k)
    return sort_df
