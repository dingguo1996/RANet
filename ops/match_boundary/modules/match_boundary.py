from torch.nn.modules.module import Module
from ..functions.match_boundary import MatchBoundaryFunction


class MatchBoundary(Module):

    def __init__(self):
        super(MatchBoundary, self).__init__()

    def forward(self, prob_boundary):
        return MatchBoundaryFunction.apply(prob_boundary)
