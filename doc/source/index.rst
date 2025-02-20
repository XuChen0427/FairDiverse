.. FairDiverse documentation master file.
.. title:: FairDiverse v0.0.1

=========================================================

`HomePage <https://github.com/XuChen0427/FairDiverse>`_ |

Introduction
-------------------------
FairDiverse is a unified, comprehensive and efficient benchmark toolkit for fairnes-aware and diversity-aware IR models.
It aims to help the researchers to reproduce and develop IR models.

In the lastest release, our library includes 30+ algorithms covering four major categories:

- Pre-processing models
- In-processing models
- Post-processing models
- IR base models


We design a unified pipelines.

.. image:: C:/lab/P-fairness_project/img/pipeline.png
    :width: 600
    :align: center


For the usage, we use following steps:


.. image:: C:/lab/P-fairness_project/img/usage.png
    :width: 600
    :align: center

The utilized parameters in each config files can be found in following docs:

Parameter Descriptions
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Recommendation Parameters

   parameters/recommendation/data_preprocessing
   parameters/recommendation/evaluation
   parameters/recommendation/new_dataset
   parameters/recommendation/In-processing
   parameters/recommendation/Post-processing

.. toctree::
   :maxdepth: 2
   :caption: Search Parameters

   parameters/search/new_dataset
   parameters/search/data_preprocessing
   parameters/search/Pre-processing
   parameters/search/Post-processing

Custom Your Models (APIs)
----------------------------



For the develop your own recommendation model, you can use following steps:


.. image:: C:/lab/P-fairness_project/img/rec_develop_steps.png
    :width: 600
    :align: center

.. toctree::
   :maxdepth: 2
   :caption: Recommendation develop APIs

   APIs/recommendation/recommendation.reranker
   APIs/recommendation/recommendation.trainer
   APIs/recommendation/recommendation.rank_model.Abstract_Ranker
   APIs/recommendation/recommendation.rerank_model.Abstract_Reranker

.. toctree::
   :maxdepth: 2
   :caption: Recommendation other APIs

   APIs/recommendation/recommendation.evaluator
   APIs/recommendation/recommendation.llm_rec
   APIs/recommendation/recommendation.metric
   APIs/recommendation/recommendation.process_data
   APIs/recommendation/recommendation.sampler
   APIs/recommendation/recommendation.utils.group_utils


.. toctree::
   :maxdepth: 4
   :caption: Search develop APIs

   APIs/search/search.trainer_preprocessing_ranker
   APIs/search/search.preprocessing_model.fair_model
   APIs/search/search.postprocessing_model.base
   APIs/search/search.trainer.base


.. toctree::
   :maxdepth: 4
   :caption: Search other APIs

   APIs/search/search.fairness_evaluator
   APIs/search/search.preprocessing_model.utils
   APIs/search/search.ranker_model.ranker
   APIs/search/search.ranklib_ranker
   APIs/search/search.utils.loss
   APIs/search/search.utils.process_dataset
   APIs/search/search.utils.utils
   APIs/search/search.llm_model.api_llm
   APIs/search/search.utils.div_type






The Team
------------------
FairDiverse is developed and maintained by `RUC, UvA`.

Here is the list of our lead developers in each development phase.

======================   ===============   =============================================
Time                     Version 	        Lead Developers
======================   ===============   =============================================
Nov. 2024 ~ Feb. 2025    v0.0.1            `Chen Xu <https://github.com/XuChen0427>`_, `Zhirui Deng <https://github.com/DengZhirui>`_, `Clara Rus <https://github.com/ClaraRus>`_
======================   ===============   =============================================

License
------------
FairDiverse uses `MIT License <https://github.com/XuChen0427/FairDiverse/blob/master/LICENSE>`_.