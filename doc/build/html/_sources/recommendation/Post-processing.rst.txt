Parameter settings for post-processing models
=====================================================================================================================================

(Default values are in ~/recommendation/Post-processing.yaml)

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Post-processing required parameters
--------------------------------------------

Model set ups
''''''''''''''''''
- ``ranking_store_path (str)`` : The log path of the in-processing models, which stores the ranking scores for re-ranking
- ``model (str)`` : The post-processing model name.



Log set ups
''''''''''''''''''
- ``log_name (str)`` : The running log name, which will create a new dictionary ~recommendation\log\log_name\ and store the test results and all parameters into it.



Evaluation set ups
''''''''''''''''''
- ``fairness_metrics (list)`` : The fairness metrics you want to record
- ``fairness_type (str)`` : The fairness computational methods, choosed among ["Exposure", "Utility"]. Exposure means computes fairness on the item exposure, Uility means computes fairness on the ranking scores.



For other evaluation settings, please reference to parameters in evaluation.rst
