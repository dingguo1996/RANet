from torch.autograd import Function

from .. import follow_cluster_cuda


class FollowClusterFunction(Function):

    @staticmethod
    def forward(ctx, class_cluster_table, threshold):
        ctx.class_cluster_table = class_cluster_table
        ctx.feature_size = class_cluster_table.size()

        batch_size, data_height, data_width = class_cluster_table.size()

        follow_index = class_cluster_table.new_zeros(batch_size, data_height).int()
        if class_cluster_table.is_cuda:
            follow_cluster_cuda.forward(class_cluster_table, threshold, follow_index)
        else:
            raise NotImplementedError
        return follow_index

    @staticmethod
    def backward(ctx, grad_follow_index):
        return None, None

follow_cluster = FollowClusterFunction.apply
