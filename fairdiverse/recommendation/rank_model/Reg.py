import numpy as np
from .Abstract_Ranker import Abstract_Regularizer
import torch.nn.functional as F
import torch
r"""
Reg
################################################
Reference: Toshihiro Kamishima and Shotaro Akaho. 2017. Considerations on recommendation independence for a fnd-good-items task

##Note that in this code, we extend the two-group cases into multi-group cases

"""


class Reg(Abstract_Regularizer):
    def __init__(self, config, group_weight):
        super().__init__(config)


    def fairness_loss(self, input_dict):
        losses = input_dict['scores'] #[B]
        return torch.var(losses)



