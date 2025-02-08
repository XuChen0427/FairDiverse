.. FairDiverse documentation master file.
.. title:: FairDiverse v0.1.1

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

.. toctree::
   :maxdepth: 1
   :caption: Parameter Settings

.. toctree::
   :maxdepth: 2
   :caption: Recommendation

   parameters/recommendation/data_preprocessing
   parameters/recommendation/evaluation
   parameters/recommendation/new_dataset
   parameters/recommendation/In-processing
   parameters/recommendation/Post-processing


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





The Team
------------------
FairDiverse is developed and maintained by `RUC, UvA`.

Here is the list of our lead developers in each development phase. They are the souls of RecBole and have made outstanding contributions.

======================   ===============   =============================================
Time                     Version 	        Lead Developers
======================   ===============   =============================================
Nov. 2024 ~ Feb. 2025    v0.1.1            `Chen Xu <https://github.com/XuChen0427>`_, `Zhirui Deng <https://github.com/DengZhirui>`_, `Clara Rus <https://github.com/ClaraRus>`_
======================   ===============   =============================================

License
------------
FairDiverse uses `MIT License <https://github.com/XuChen0427/FairDiverse/blob/master/LICENSE>`_.