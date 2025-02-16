from abc import ABC, abstractmethod
import torch.nn as nn

class BasePostProcessModel(nn.Module, ABC):
    def __init__(self, dropout=0.1):
        super(BasePostProcessModel, self).__init__()
        self.dropout = dropout
    
    @abstractmethod
    def fit(self):
        pass

class BasePostProcessUnsupervisedModel(ABC):
    def __init__(self, top_k):
        super(BasePostProcessUnsupervisedModel, self).__init__()
        self.top_k = top_k
    
    @abstractmethod
    def rerank(self):
        pass