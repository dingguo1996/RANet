from torch.nn.modules.module import Module
from ..functions.follow_cluster import FollowClusterFunction


class FollowCluster(Module):

    def __init__(self, threshold=0.9):
        super(FollowCluster, self).__init__()
        self.threshold = threshold

    def forward(self, class_cluster_table):
        return FollowClusterFunction.apply(class_cluster_table, self.threshold)
