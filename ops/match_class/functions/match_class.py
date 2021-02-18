from torch.autograd import Function

from .. import match_class_cuda


class MatchClassFunction(Function):

    @staticmethod
    def forward(ctx, class_pred_softmax, class_max_prob_A_index):
        ctx.class_pred_softmax = class_pred_softmax
        ctx.class_max_prob_A_index = class_max_prob_A_index
        ctx.feature_size = class_pred_softmax.size()

        batch_size, num_channels, data_height, data_width = class_pred_softmax.size()

        confidence_output = class_pred_softmax.new_ones(batch_size, data_height * data_width, data_height * data_width)
        if class_pred_softmax.is_cuda:
            match_class_cuda.forward(class_pred_softmax, class_max_prob_A_index, confidence_output)
        else:
            raise NotImplementedError
        return confidence_output

    @staticmethod
    def backward(ctx, grad_confidence_output):
        class_pred_softmax = ctx.class_pred_softmax
        class_max_prob_A_index = ctx.class_max_prob_A_index
        feature_size = ctx.feature_size
        assert (feature_size is not None and grad_confidence_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        grad_class_pred_softmax_input = None
        if ctx.needs_input_grad[0]:
            grad_class_pred_softmax_input = grad_confidence_output.new_zeros(batch_size, num_channels, data_height,
                                                          data_width)
            match_class_cuda.backward(grad_confidence_output.contiguous(), class_pred_softmax, class_max_prob_A_index,
                                      grad_class_pred_softmax_input)

        return grad_class_pred_softmax_input, None


match_class = MatchClassFunction.apply
