from torch.nn.modules.module import Module
from ..functions.intra_collection import IntraCollectionFunction


class IntraCollection(Module):

    def __init__(self):
        super(IntraCollection, self).__init__()


    def forward(self, rep_feat, feat, vtopk_table, region_map):
        return IntraCollectionFunction.apply(rep_feat, feat, vtopk_table, region_map)
