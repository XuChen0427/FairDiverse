Parameter settings data pre-processing
========================================

(Default values are in ~/recommendation/properties/dataset.yaml)

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Required parameters
----------------------

Cache set ups
''''''''''''''''''
- ``reprocess (bool)`` : Should the preprocessing be redone based on the new parameters instead of using the cached files in ~/recommendation/process_dataset


Filtering set ups
''''''''''''''''''
- ``item_val (int)`` : Retain items in the dataset if their total interactions with all users exceed item_val.
- ``user_val (int)`` : Retain users in the dataset if their total interactions with all items exceed user_val.
- ``group_val (int)`` : Retain item groups in the dataset if their total interactions with all users exceed group_val.
- ``group_aggregation_threshold (int)`` : If the number of items owned by a group is less than this value, those groups will be merged into a single group called the 'infrequent group.'
- ``sample_size (float)`` : Sample radio of the whole dataset to form a new subset dataset for training.


Connect set ups
''''''''''''''''''
- ``valid_ratio (float)`` : The radio for validate set.
- ``test_ratio (float)`` : The radio for test set.
- ``sample_num (int)`` : Negative sample numbers for ranking-based evaluation.
- ``history_length (int)`` : The truncated length of a user's interaction history with items.







