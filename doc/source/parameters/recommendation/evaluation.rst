Parameter settings for evaluations
====================================

(Default values are in ~/recommendation/properties/evaluation.yaml)


The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Evaluation required parameters
---------------------------------

Evaluation process set ups
''''''''''''''''''''''''''''
- ``eval_step (int)`` : How many epochs apart should the model be evaluated on the validation set.
- ``eval_type (str)`` : Evaluation types can be chosen from ['ranking', 'CTR'], with ranking being the most commonly used. Most models do not support CTR tasks.
- ``eval_batch_size (int)`` : The batch size used to conduct evaluation.


Evaluation metric set ups
''''''''''''''''''''''''''
- ``watch_metric (str)`` : During training, the model should be saved for testing when the watch_metric on the validation set reaches its highest value.
- ``topk (list)`` : The evaluation ranking list size list, such as [5,10, 20]
- ``store_scores (bool)`` : Decide whether to save ranking scores for the post-processing (re-ranking) step.
- ``decimals (int)`` : Number of decimal places retained of evaluation metrics.
- ``mmf_eval_ratio (float)`` : The parameters of the MMF metric define the how many ratio of the worst-off group's utility compared to the utilities of all groups.
- ``fairness_type (str)`` : The fairness computational methods, choosed among ["Exposure", "Utility"]. Exposure means computes fairness on the item exposure, Uility means computes fairness on the ranking scores.





