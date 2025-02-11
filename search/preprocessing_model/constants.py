from search.preprocessing_model.CIFRank import CIFRank
from search.preprocessing_model.gFair import gFair
from search.preprocessing_model.iFair import iFair
from search.preprocessing_model.LFR import LFR

fairness_method_mapping = {
    'CIFRank': CIFRank,
    'gFair': gFair,
    'iFair': iFair,
    'LFR': LFR
}