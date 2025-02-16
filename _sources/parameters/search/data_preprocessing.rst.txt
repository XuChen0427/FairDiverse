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

Data directory set ups
''''''''''''''''''
- ``ground_truth (str)`` : The path of ground truth data, default as ground_truth\.
- ``doc_content_dir (str)`` : The path of document content, default as clueweb09_doc_content_dir\.
- ``query_suggestion (str)`` : The path of query suggestion, default as query_suggestion.xml.

Embedding set ups
''''''''''''''''''
- ``embedding_dir (str)`` : The path of query and documents' embedding, default as embedding\.
- ``embedding_type (str)`` : The embedding type of the query and documents, default as doc2vec.
- ``embedding_length (int)`` : The embedding length of the query and documents, default as 100.
