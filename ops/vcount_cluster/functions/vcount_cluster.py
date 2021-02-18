from torch.autograd import Function

from .. import vcount_cluster_cuda


class VcountClusterFunction(Function):

    @staticmethod
    def forward(ctx, region_attention_table, region_map):
        ctx.region_attention_table = region_attention_table
        ctx.region_map = region_map
        ctx.region_attention_table_size = region_attention_table.size()

        data_height, data_width = region_attention_table.size()
        data_cluster = region_map.max().item() + 1

        pvic_table = region_attention_table.new_zeros(data_cluster, data_height) #pos_value_in_cluster_table
        if region_attention_table.is_cuda:
            vcount_cluster_cuda.forward(region_attention_table, region_map, pvic_table)
        else:
            raise NotImplementedError
        return pvic_table

    @staticmethod
    def backward(ctx, grad_pvic_table):
        region_attention_table = ctx.region_attention_table
        region_map = ctx.region_map
        region_attention_table_size = ctx.region_attention_table_size
        assert (region_attention_table_size is not None and grad_pvic_table.is_cuda)

        data_height, data_width = region_attention_table_size
        grad_region_attention_table = None
        if ctx.needs_input_grad[0]:
            grad_region_attention_table = grad_pvic_table.new_zeros(data_height, data_width)
            vcount_cluster_cuda.backward(grad_pvic_table.contiguous(), region_map, grad_region_attention_table)
        return grad_region_attention_table, None

vcount_cluster = VcountClusterFunction.apply
