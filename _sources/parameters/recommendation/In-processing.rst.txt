Parameter settings for in-processing models
==================================================================================================================================

(Default values are in ~/recommendation/In-processing.yaml)

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

In-processing required parameters
--------------------------------------------

Base model set ups
''''''''''''''''''''''''
- ``model (str)`` : The base model name.
- ``data_type (str)`` : The base model training style, choosen from ['point', 'pair', 'sequential']. Note not all model support all types.

In-processing model set ups
'''''''''''''''''''''''''''''
- ``fair-rank (bool)`` : Bool variable for determining whether add fairness- and diversity-aware modules into base models.
- ``rank_model (str)`` : The fairness- and diversity-aware model name.


LLM setups
''''''''''''''''''
- ``use_llm (bool)`` : Bool variable for determining whether to use LLM for fairness ranking.
- ``llm_type (str)`` : Variables that determine the method of calling the large model, choosen from ['api', 'local', 'vllm'].
- ``llm_name (str)`` : The name of the LLM to be called, which must match the key in 'llm_path_dict' in ~/recommendation/properties/models/LLM.yaml.
- ``grounding_model (str)`` : The name of the grounding model which is used to ground the LLM generated answer to real items. It should match the key in 'llm_path_dict' in ~/recommendation/properties/models/LLM.yaml.
- ``saved_embs_filename (str)``: The file name which store the item names as embeddings using the grounding model. Input None if you don't need to save it.
- ``fair_prompt (str)``: Fairness prompts which is input to the large model to generate different types of fairness-aware recommendations.


Log set ups
''''''''''''''''''
- ``log_name (str)`` : The running log name, which will create a new dictionary ~recommendation\log\log_name\ and store the test results and all parameters into it.




For evaluation settings, please reference to parameters in evaluation.rst
