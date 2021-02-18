from torch.nn.modules.module import Module
from ..functions.split_repscore import SplitRepscoreFunction


class SplitRepscore(Module):

    def __init__(self):
        super(SplitRepscore, self).__init__()

    def forward(self, repscore_map, region_map):
        return SplitRepscoreFunction.apply(repscore_map, region_map)
