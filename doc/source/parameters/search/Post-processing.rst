Parameter settings for post-processing models
=====================================================================================================================================
(Default values are in ~/search/Post-processing.yaml)

The benchmark provides several arguments for describing:

- Basic setting of the parameters

See below for the details:

Post-processing required parameters
''''''''''''''''''
- ``model (str)`` : The name of the pre-processing model, which automatically loads the configs from search/properties/models/<model_name>.yaml.
- ``model_save_dir (str)`` : The save path of model.
- ``tmp_dir (str)`` : The temperature path of files.
- ``mode (str)`` : Train or test mode.
- ``best_model_list (list)`` : The best model path list for test mode.

Supervised and unsupervised model train parameters
''''''''''''''''''
- ``device (str)`` : used device (cpu or gpu) to run the codes.
- ``epoch (int)`` : training epochs.
- ``batch_size (int)`` : training batch size.
- ``learning_rate (float)`` : learning rate of optimizing the models.
- ``dropout (float)`` : The dropout for training.
- ``loss (str)`` : The loss type for training.
- ``lambda (float)`` : The hyperparameter lambda in PM2 and xQuAD for balance relevance and diversity.

LLMs-based model parameters
''''''''''''''''''
- ``prompts_dir (str)`` : The path of LLMs-based model's prompts, default as prompts\llm_rerank.txt.
- ``model_name (str)`` : The name of LLMs-based backbone model, default as gpt-4o.
- ``api_url (str)`` : The api url of LLM API.
- ``api_key (str)`` : The api key of LLM API.
- ``max_new_tokens (int)`` : The max new tokens for LLMs-based model, default as 1024.
- ``temperature (float)`` : The temperature for LLM generation.
- ``top_p (float)`` : The top_p for LLM generation.

Evaluation parameters
''''''''''''''''''
- ``eval_step (int)`` : How many epochs apart should the model be evaluated on the validation set.
- ``eval_batch_size (int)`` : The batch size used to conduct evaluation.

Log set ups
''''''''''''''''''
- ``log_name (str)`` : The running log name, which will create a new dictionary ~search\log\log_name\ and store the test results and all parameters into it.


