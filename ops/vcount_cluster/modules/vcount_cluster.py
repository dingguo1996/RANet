from torch.nn.modules.module import Module
from ..functions.vcount_cluster import VcountClusterFunction


class VcountCluster(Module):

    def __init__(self):
        super(VcountCluster, self).__init__()

    def forward(self, region_attention_table, region_map):
        return VcountClusterFunction.apply(region_attention_table, region_map)
