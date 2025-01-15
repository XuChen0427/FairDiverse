## Fair-IR
__Fair-IR__ is a toolkit for reproducing and developing fairness-aware Information Retrieval (IR) tasks.

### Requirements
```
python>=3.7.0
numpy>=1.20.3
torch>=1.11.0
```

### Quick-start
With the source code, you can use the provided script for initial usage of our library:
```
python main.py
```

For the search task, you can begin with:
```
python main.py --task search --stage re-ranking --dataset clueweb09 --train_config_file train.yaml
```

### Datasets
For the recommendation dataset, we utilize the dataset format in [Recbole Datasets](https://recbole.io/dataset_list.html).
For the search dataset, we utilize the ClueWeb dataset (http://boston.lti.cs.cmu.edu/Services/clueweb09_batch/)



