from abc import ABC, abstractmethod
import torch.nn as nn

class BasePostProcessModel(nn.Module, ABC):
    def __init__(self, dropout=0.1):
        r"""
        A base class for post-processing supervised models.

        :param dropout: The dropout rate used in the model to reduce overfitting. Default is 0.1.
        """
        super(BasePostProcessModel, self).__init__()
        self.dropout = dropout
    
    @abstractmethod
    def fit(self):
        r"""
        Train the post-processing model.
        """
        pass

class BasePostProcessUnsupervisedModel(ABC):
    def __init__(self, top_k):
        r"""
        A base class for unsupervised post-processing models.

        :param top_k: The number of top-ranked items to retain after reranking.
        """
        super(BasePostProcessUnsupervisedModel, self).__init__()
        self.top_k = top_k
    
    @abstractmethod
    def rerank(self):
        r"""
        Reranks documents based on an unsupervised criterion.
        """
        pass