from search.postprocessing_model.xQuAD import xQuAD
from search.postprocessing_model.DESA import DESA
from search.postprocessing_model.PM2 import PM2
from search.postprocessing_model.DALETOR import DALETOR
from search.postprocessing_model.base import BasePostProcessModel, BasePostProcessUnsupervisedModel

diversity_method_mapping = {
    'DESA': DESA,
    'DALETOR': DALETOR,
    'PM2': PM2,
    'xQuAD': xQuAD
}