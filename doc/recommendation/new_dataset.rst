Parameter settings for your new dataset (create a new yaml file in ~/recommendation/properties/dataset/your_new_dataset.yaml)
=========================

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Required parameters
----------------------

Column set ups
''''''''''''''''''
- ``user_id (str)`` : The column name used to identify the user.
- ``item_id (str)`` : The column name used to identify the item.
- ``group_id (str)`` : The column name used to identify the group.
- ``label_id (str)`` : The column name used to identify the label.
- ``timestamp (str)`` : The column name used to identify the timestamp.
- ``label_threshold (int)`` : Threshold means the label exceed the value will be regarded as 1, otherwise, it will be accounted into 0.


LLMs-based set ups
''''''''''''''''''
- ``text_id (str)`` : The column name used to identify textual information of item (e.g. new title).
- ``item_domain (str)`` : The item domain name used to generate prompts.












