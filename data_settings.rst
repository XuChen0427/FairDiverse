Data settings
=========================

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

In-processing setups
----------------------


LLM setups
''''''''''''''''''
- ``use_llm (bool)`` : Bool variable for determining whether to use LLM for fairness ranking.
- ``llm_type (str)`` : Variables that determine the method of calling the large model, choosen from ['api', 'local', 'vllm'].
- ``llm_name (str)`` : The name of the LLM to be called, which must match the key in 'llm_path_dict' in Item-Fair/Item-Fair-IR/recommendation/properties/models/LLM.yaml.
- ``grounding_model (str)`` : The name of the grounding model which is used to ground the LLM generated answer to real items. It should match the key in 'llm_path_dict' in Item-Fair/Item-Fair-IR/recommendation/properties/models/LLM.yaml.
- ``saved_embs_filename (str)``: The file name which store the item names as embeddings using the grounding model. Input None if you don't need to save it.
- ``fair_prompt (str)``: Fairness prompts which is input to the large model to generate different types of fairness-aware recommendations.

