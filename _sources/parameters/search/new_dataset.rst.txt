Parameter settings for dataset
=====================================================================================================================================
(create a new yaml file in ~/search/properties/dataset/your_new_dataset.yaml)

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Dataset
''''''''''''''''''
- ``name (str)`` : The name of the dataset.
- ``path (str)`` : The path to the dataset.

Column definition
''''''''''''''''''
- ``query_col (str)`` : The query column name; if not present, it will be added.
- ``id_col (str)`` : The ID column name.
- ``score_col (str)`` : The score column name.
- ``feature_cols (list)`` : The list of column names for the features.
- ``sensitive_cols (list)`` : The list of sensitive columns to be considered.

Train-Test split
''''''''''''''''''
- ``k_fold (int)`` : The number of k-fold stratified splits.
- ``ratio_split (float)`` : The ratio of the test set.
- ``pos_th (int)`` : The threshold for considering negative samples during training.