## FairDiverse

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

__FairDiverse__ is a toolkit for reproducing and developing fairness- and diversity-aware Information Retrieval (IR) tasks.

![FairDiverse pipelines](img/pipeline.png)

## Contact
We welcome the contributors to join our toolkit implementation! Any information, please contact:

[Chen Xu](https://xuchen0427.github.io/): [Email Me](xc_chen@ruc.edu.cn)


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

#### For post-processing method RAIF
Require Gurobi license 
```
mip>=1.15.0
gurobipy>=12.0.1
```

## Quick-start
![FairDiverse pipelines](img/usage.png)

With the source code,  you can start three steps: 

1. Download the datasets and check the default parameters of the four stages of pipelines (we provide a toy dataset steam already).

2. Set your custom configuration file to execute the pipeline (we already provide a template file).

3. Run the shell command, with the task, stage, dataset, and your custom configuration file specifying (you can directly run the command).

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

## Implemented Models

### Recommendation tasks

#### Base models

| Types    | Models                                                | Descriptions                                                      |
|----------|-------------------------------------------------------|-------------------------------------------------------------------|
| Non-LLMs | [DMF](https://dl.acm.org/doi/10.5555/3172077.3172336) | optimizes the matrix factorization with the deep neural networks. |
| Non-LLMs | [BPR](https://doi.org/10.1145/3543507.3583355)        | optimizes pairwise ranking via implicit feedback.                 |                                                         |
| Non-LLMs | [GRU4Rec](https://arxiv.org/abs/1606.08117)    | employs gated recurrent units (GRUs) for session-based recommendations.                                                           |
| Non-LLMs | [SASRec](https://arxiv.org/abs/1808.09781)     | leverages self-attention mechanisms to model sequential user behavior.                                                           |
| LLMs     | [LLama3](https://arxiv.org/abs/2407.21783)     | utilizing rank-specific prompts to conduct ranking tasks under LLMs                                                           |
| LLMs     | [Qwen2](https://arxiv.org/abs/2309.16609)      | utilizing rank-specific prompts to conduct ranking tasks under LLMs                                                           |


#### In-processing models





## License
FairDiverse uses [MIT License](./LICENSE). All data and code in this project can only be used for academic purposes.
