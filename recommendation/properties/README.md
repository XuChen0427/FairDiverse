## parameters settings

### Dataset
The default process settings is in dataset.yaml, where it stores:
* **item_val**: filter the items less than such value
* **user_val**: filter the users less than such value 
* **group_val:** filter the groups less than such value
* **group_aggregation_threshold**: less than such threshold group will be aggregated into one group
* **valid_ratio, test_ratio**: the radio of the valid test and test set

The default dataset settings is in dataset folder, where each yaml contains the 
* **user_id, item_id, group_id**: indicate the column of user, item and group
* **label_id**: indicate the label
* **label_threshold**: threshold means the label exceed the value will be regarded as 1, otherwise, it will be accounted into 0


