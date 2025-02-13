Parameter settings for pre-processing models
=====================================================================================================================================
(Default values are in ~/search/Pre-processing.yaml)

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Pre-processing required parameters
''''''''''''''''''
- ``preprocessing_model (str)`` : The name of the pre-processing model, which automatically loads the configs from search/properties/models/<model_name>.yaml.

Ranklib parameters
''''''''''''''''''
- ``name (str``) : The name of the ranker class.
- ``ranker_path (str)`` : The path to the Ranklib library.
- ``rel_max (int)`` : The maximum relevance to be assigned to items.
- ``ranker (str)`` : The name of the ranking model.
- ``ranker_id (int)`` : The ID of the ranking model as per the Ranklib library.
- ``metric (str)`` : The evaluation metric.
- ``top_k (int)`` : The evaluation at top-k.
- ``lr (float)`` : The learning rate for training.
- ``epochs (int)`` : The number of epochs for training.
- ``train_data (list)`` : The data types for training ["original", "fair"].
- ``test_data (list)`` : The data types for testing ["original", "fair"].

Evaluation parameters
''''''''''''''''''
- ``metrics (list)`` : The list of fairness evaluation metrics having the following values ["diversity", "select_rate", "exposure", "individual", "igf"].
- ``rankings (list)`` : The list of rankings to perform the evaluation on, having the following values: "score_col" - original data, "<score_col>__<score_col>" - trained and tested on original data, "<score_col>_fair" - transformed data, "<score_col>fair_<score_col>_fair" - trained and tested on transformed data.
- ``k_list (list)`` : The top-k values for evaluation.

