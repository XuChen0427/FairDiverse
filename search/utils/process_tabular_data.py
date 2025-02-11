from sklearn import preprocessing
import os
import pathlib


def create_train_split(df, query_col, id_col, group_col, sort_col, pos_th, data_path, k_fold, ratio_split):
    """
    Creates k-fold train-test splits and saves them as text files.

    :param df: Pandas DataFrame containing the dataset.
    :param query_col: Column representing query IDs (grouping criterion).
    :param id_col: Column containing unique sample IDs.
    :param group_col: Column used for stratified sampling (e.g., demographic group).
    :param sort_col: Column used for positive sample thresholding.
    :param pos_th: Threshold to determine positive samples.
    :param data_path: Directory to save the train-test split files.
    :param k_fold: Number of folds for cross-validation.
    :param ratio_split: Fraction of data used for training in each fold.
    """
    os.makedirs(data_path, exist_ok=True)

    for i in range(k_fold):
        train_ids, test_ids = select_balanced_df(df, query_col, id_col, group_col, sort_col, pos_th, ratio_split)

        # Save train-test splits to files
        with open(os.path.join(data_path, f'test_samples_{i}.txt'), 'w') as f:
            f.write("\n".join(map(str, test_ids)))

        with open(os.path.join(data_path, f'train_samples_{i}.txt'), 'w') as f:
            f.write("\n".join(map(str, train_ids)))


def select_balanced_df(df, query_col, id_col, group_col, sort_col, pos_th, return_ratio):
    """
    Performs stratified sampling to create balanced train-test splits.

    :param df: Pandas DataFrame containing the dataset.
    :param query_col: Column representing query IDs.
    :param id_col: Column containing unique sample IDs.
    :param group_col: Column used for stratification (e.g., demographic group).
    :param sort_col: Column used to separate positive and negative samples.
    :param pos_th: Threshold to classify positive samples.
    :param return_ratio: Fraction of data to use in the training set.
    :return: Tuple (train_samples, test_samples) containing train and test sample IDs.
    """
    train_samples, test_samples, drop_qid = [], [], []

    for qid, df_query in df.groupby(query_col):
        pos_samples = df_query[df_query[sort_col] >= pos_th]  # Positive samples
        neg_samples = df_query[df_query[sort_col] < pos_th]  # Negative samples

        # Identify groups with insufficient samples
        group_sizes = pos_samples[group_col].value_counts()
        insufficient_groups = group_sizes[group_sizes < 2].index.tolist()

        if insufficient_groups:
            drop_qid.append(qid)
            continue  # Skip this query group if it has insufficient data

        # Perform stratified sampling
        stratified_pos_sample = pos_samples.groupby(group_col, group_keys=False).apply(
            lambda x: x.sample(frac=return_ratio, random_state=42))
        stratified_neg_sample = neg_samples.groupby(group_col, group_keys=False).apply(
            lambda x: x.sample(frac=return_ratio, random_state=42))

        train_samples.extend(stratified_pos_sample[id_col].values)
        train_samples.extend(stratified_neg_sample[id_col].values)

        test_samples.extend(df_query.loc[~df_query[id_col].isin(train_samples), id_col].values)

    # Remove queries with insufficient samples for stratified sampling from the train and test sets
    train_samples = [uid for uid in train_samples if uid not in drop_qid]
    test_samples = [uid for uid in test_samples if uid not in drop_qid]

    return train_samples, test_samples



def check_nan(df, cols_train):
    for col in cols_train:
        if df[col].isnull().values.any():
            return True
    return False


def norm_features(features_cols, data):
    # normalize features
    for f in features_cols:
        if min(data[f]) < 0:
            min_value = min(data[f])
            data[f] = data[f].apply(lambda x: x - min_value)

    min_max_scaler = preprocessing.MinMaxScaler()
    data[features_cols] = min_max_scaler.fit_transform(data[features_cols].values)
    return data


def writeToTXT(file_name_with_path, _df):
    # try:
    #     _df.to_csv(file_name_with_path, header=False, index=False, sep=' ')
    # except FileNotFoundError:
    directory = os.path.dirname(file_name_with_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    print("Make folder ", directory)
    _df.to_csv(file_name_with_path, header=False, index=False, sep=' ')