from torch.autograd import Function

from .. import intra_collection_cuda


class IntraCollectionFunction(Function):

    @staticmethod
    def forward(ctx, rep_feat, feat, vtopk_table, region_map):
        ctx.rep_feat = rep_feat
        ctx.feat = feat
        ctx.vtopk_table = vtopk_table
        ctx.region_map = region_map
        ctx.feature_size = rep_feat.size()

        num_rep_pixels, num_channels = rep_feat.size()

        collect_rep_feat = rep_feat.new_ones(num_rep_pixels, num_channels)
        if rep_feat.is_cuda:
            intra_collection_cuda.forward(rep_feat, feat, vtopk_table, region_map, collect_rep_feat)
        else:
            raise NotImplementedError
        return collect_rep_feat

    @staticmethod
    def backward(ctx, grad_collect_rep_feat):
        rep_feat = ctx.rep_feat
        feat = ctx.feat
        vtopk_table = ctx.vtopk_table
        region_map = ctx.region_map
        assert (rep_feat is not None and grad_collect_rep_feat.is_cuda)

        num_rep_pixels, num_channels = rep_feat.size()
        num_pixels, _ = feat.size()
        grad_rep_feat = None
        grad_feat = None
        if ctx.needs_input_grad[0]:
            grad_rep_feat = grad_collect_rep_feat.new_zeros(num_rep_pixels, num_channels)
            grad_feat = grad_collect_rep_feat.new_zeros(num_pixels, num_channels)
            intra_collection_cuda.backward(grad_collect_rep_feat.contiguous(), rep_feat, feat,
                                           vtopk_table, region_map, grad_rep_feat, grad_feat)

        return grad_rep_feat, grad_feat, None, None, None


intra_collection = IntraCollectionFunction.apply
