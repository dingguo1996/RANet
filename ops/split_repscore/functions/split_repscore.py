from torch.autograd import Function

from .. import split_repscore_cuda


class SplitRepscoreFunction(Function):

    @staticmethod
    def forward(ctx, repscore_map, region_map):
        ctx.repscore_map = repscore_map
        ctx.region_map = region_map
        ctx.feature_size = repscore_map.size()

        data_length, = repscore_map.size()
        data_cluster = region_map.max().item() + 1

        pric_table = repscore_map.new_zeros(data_cluster, data_length) #pos_repscore_in_cluster_table
        if repscore_map.is_cuda:
            split_repscore_cuda.forward(repscore_map, region_map, pric_table)
        else:
            raise NotImplementedError
        return pric_table

    @staticmethod
    def backward(ctx, grad_pric_table):
        return None, None

split_repscore = SplitRepscoreFunction.apply
