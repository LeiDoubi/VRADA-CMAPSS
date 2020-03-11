from torch.autograd import Function
# Modified based on the https://github.com/fungtion/DANN/blob/master/models/functions.py


class ReverseGradient(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.alpha)
