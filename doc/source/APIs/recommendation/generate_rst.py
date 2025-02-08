import os

content = '''.. automodule:: {}
   :members:
   :undoc-members:
   :show-inheritance:
'''


file_name = ["evaluator", "metric", "llm_rec", "process_data", "reranker", "trainer", "sampler",
             "utils.group_utils", "rank_model.Abstract_Ranker", "rerank_model.Abstract_Reranker"]

file_template = "recommendation.{}.rst"

for k in file_name:
    with open(file_template.format(k), 'w') as f:
        f.write(str(content.format("recommendation.{}".format(k))))

