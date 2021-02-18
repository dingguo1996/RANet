from torch.autograd import Function

from .. import match_boundary_cuda


class MatchBoundaryFunction(Function):

    @staticmethod
    def forward(ctx, prob_boundary):
        ctx.prob_boundary = prob_boundary
        ctx.feature_size = prob_boundary.size()

        batch_size, num_channels, data_height, data_width = prob_boundary.size()

        table_boundary = prob_boundary.new_ones(batch_size, data_height * data_width, data_height * data_width)
        index_output = prob_boundary.new_zeros(batch_size, data_height * data_width, data_height * data_width).int()
        table_near = prob_boundary.new_zeros(batch_size, data_height * data_width, data_height * data_width).int()
        if prob_boundary.is_cuda:
            match_boundary_cuda.forward(prob_boundary, table_boundary, index_output)
        else:
            raise NotImplementedError
        ctx.save_for_backward(index_output)
        return table_boundary

    @staticmethod
    def backward(ctx, grad_table_boundary):
        prob_boundary = ctx.prob_boundary
        feature_size = ctx.feature_size
        assert (feature_size is not None and grad_table_boundary.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        index_output = ctx.saved_tensors[0]
        grad_prob_boundary_input = None
        if ctx.needs_input_grad[0]:
            grad_prob_boundary_input = grad_table_boundary.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            match_boundary_cuda.backward(grad_table_boundary.contiguous(), index_output, prob_boundary, grad_prob_boundary_input)

        return grad_prob_boundary_input


match_boundary = MatchBoundaryFunction.apply
