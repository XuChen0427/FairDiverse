## FairDiverse

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

__FairDiverse__ is a toolkit for reproducing and developing fairness- and diversity-aware Information Retrieval (IR) tasks.

![FairDiverse pipelines](img/pipeline.png)


## General Requirements
```
python>=3.7.0
numpy>=1.20.3
torch>=1.11.0
```

#### For LLMs-based ranking models
Require Linux system
```
vllm>=0.6.6
```



## Quick-start
With the source code, you can use the provided script and toy dataset steam for initial usage of our library.

#### Recommendation tasks:
For in-processing methods, please run

```
python main.py --task recommendation --stage in-processing --dataset steam --train_config_file In-processing.yaml
```

For post-processing methods, please run
```
python main.py --task recommendation --stage post-processing --dataset steam --train_config_file Post-processing.yaml
```

#### Search tasks:

For the post-processing methods, you can begin with:
```
python main.py --task search --stage post-processing --dataset clueweb09 --train_config_file train.yaml
```

## Datasets
For the recommendation dataset, we utilize the dataset format in [Recbole Datasets](https://recbole.io/dataset_list.html).

For the search dataset, we utilize the [ClueWeb dataset](http://boston.lti.cs.cmu.edu/Services/clueweb09_batch/).



## License
FairDiverse uses [MIT License](./LICENSE). All data and code in this project can only be used for academic purposes.
