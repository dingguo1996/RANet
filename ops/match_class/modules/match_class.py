from torch.nn.modules.module import Module
from ..functions.match_class import MatchClassFunction


class MatchClass(Module):

    def __init__(self):
        super(MatchClass, self).__init__()


    def forward(self, class_pred_softmax, class_max_prob_A_index):
        return MatchClassFunction.apply(class_pred_softmax, class_max_prob_A_index)
