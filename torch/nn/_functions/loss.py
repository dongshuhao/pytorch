import torch
from torch.autograd import Function

import torch
from torch.autograd import Function

class ContrastiveLoss(Function):
    def __init__(self, margin=1, size_average=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, y):
        assert input1.size() == input2.size(), "Input sizes must be equal."

        self.l22 = input1.new() #l22 = ||a - b||^2
        self.l21 = input1.new() #l21 = ||a - b||

        self._outputs = input1.new()
        _idx = input1.new().byte()

        #l22 = ||a - b||^2
        _diff = torch.abs(input1 - input2)
        self.l22 = torch.pow(_diff, 2).sum(dim=1)   # l22 = ||a - b||^2

        self.l21 = torch.sqrt(self.l22)             # l21 = ||a - b||

        #_output = l22
        self._outputs.resize_as_(self.l22).copy_(self.l22)
        self._outputs = self._outputs.select(1, 0) #_output = first column and it one column vector

        torch.eq(y, -1, out=_idx) #y==-1
        self._outputs[_idx] = self._outputs[_idx].sqrt_().mul_(-1).add_(self.margin).clamp_(min=0) #max(0, m - ||a - b||)
        self._outputs[_idx] = self._outputs[_idx].pow_(2)

        #_output = 1/2 * _output
        self._outputs.mul_(1.0 / 2.0)
        # print('same dis : diff dis  {} : {}'.format(self.l21[_idx == 0].mean(), self.l21[_idx].mean()))

        output = self._outputs.sum() #sum

        if self.size_average:
            output = output / y.size(0)  #mean

        self.save_for_backward(input1, input2, y)
        return input1.new((output,))

    def backward(self, grad_output):
        # import pdb
        # pdb.set_trace()
        v1, v2, y = self.saved_tensors

        buffer = v1.new()
        bufferl21 = v1.new()
        _idx = v1.new().byte()

        gw1 = grad_output.new()
        gw2 = grad_output.new()
        gw1.resize_as_(v1).copy_(v2) #gw1 = b
        gw2.resize_as_(v2).copy_(v1) #gw2 = a

        gw1.mul_(-1).add_(v1) #a' = sum(a - b)
        gw2.mul_(-1).add_(v2) #b' = sum(b - a)

        torch.eq(self._outputs, 0, out=_idx) #find _output < 0 because loss>0
        _idx = _idx.view(-1, 1).expand(gw1.size())
        gw1[_idx] = 0
        gw2[_idx] = 0

        #y == -1
        epsilon = 1e-12  # avoid div 0

        torch.eq(y, -1, out=_idx)
        _idx = _idx.view(-1, 1).expand(gw2.size())
        torch.add(self.l21, -self.margin, out=buffer)
        torch.add(self.l21, epsilon, out=bufferl21)
        buffer = buffer.expand(gw1.size())
        bufferl21 = bufferl21.expand(gw1.size())

        gw1[_idx] = gw1[_idx].mul_(buffer[_idx]).div_(bufferl21[_idx])
        # gw2[_idx] = gw2[_idx].mul_(buffer[_idx]).mul_(-1).div_(bufferl21[_idx])
        gw2[_idx] = gw2[_idx].mul_(buffer[_idx]).div_(bufferl21[_idx])

        if self.size_average:
            gw1.div_(y.size(0))
            gw2.div_(y.size(0))

        grad_output_val = grad_output[0] #current = lastgrad*dervative
        if grad_output_val != 1:
            gw1.mul_(grad_output_val)
            gw2.mul_(grad_output_val)

        return gw1, gw2, None


class CosineEmbeddingLoss(Function):
    def __init__(self, margin=0, size_average=True):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, y):
        self.w1 = input1.new()
        self.w22 = input1.new()
        self.w = input1.new()
        self.w32 = input1.new()
        self._outputs = input1.new()

        _idx = input1.new().byte()

        buffer = torch.mul(input1, input2)
        torch.sum(buffer, 1, out=self.w1, keepdim=True)

        epsilon = 1e-12
        torch.mul(input1, input1, out=buffer)
        torch.sum(buffer, 1, out=self.w22, keepdim=True).add_(epsilon)

        self._outputs.resize_as_(self.w22).fill_(1)
        torch.div(self._outputs, self.w22, out=self.w22)
        self.w.resize_as_(self.w22).copy_(self.w22)

        torch.mul(input2, input2, out=buffer)
        torch.sum(buffer, 1, out=self.w32, keepdim=True).add_(epsilon)
        torch.div(self._outputs, self.w32, out=self.w32)
        self.w.mul_(self.w32)
        self.w.sqrt_()

        torch.mul(self.w1, self.w, out=self._outputs)
        self._outputs = self._outputs.select(1, 0)

        torch.eq(y, -1, out=_idx)
        self._outputs[_idx] = self._outputs[_idx].add_(-self.margin).clamp_(min=0)
        torch.eq(y, 1, out=_idx)
        self._outputs[_idx] = self._outputs[_idx].mul_(-1).add_(1)

        output = self._outputs.sum()

        if self.size_average:
            output = output / y.size(0)

        self.save_for_backward(input1, input2, y)
        return input1.new((output,))

    def backward(self, grad_output):
        v1, v2, y = self.saved_tensors

        buffer = v1.new()
        _idx = v1.new().byte()

        gw1 = grad_output.new()
        gw2 = grad_output.new()
        gw1.resize_as_(v1).copy_(v2)
        gw2.resize_as_(v1).copy_(v1)

        torch.mul(self.w1, self.w22, out=buffer)
        gw1.addcmul_(-1, buffer.expand_as(v1), v1)
        gw1.mul_(self.w.expand_as(v1))

        torch.mul(self.w1, self.w32, out=buffer)
        gw2.addcmul_(-1, buffer.expand_as(v1), v2)
        gw2.mul_(self.w.expand_as(v1))

        torch.le(self._outputs, 0, out=_idx)
        _idx = _idx.view(-1, 1).expand(gw1.size())
        gw1[_idx] = 0
        gw2[_idx] = 0

        torch.eq(y, 1, out=_idx)
        _idx = _idx.view(-1, 1).expand(gw2.size())
        gw1[_idx] = gw1[_idx].mul_(-1)
        gw2[_idx] = gw2[_idx].mul_(-1)

        if self.size_average:
            gw1.div_(y.size(0))
            gw2.div_(y.size(0))

        grad_output_val = grad_output[0]
        if grad_output_val != 1:
            gw1.mul_(grad_output_val)
            gw2.mul_(grad_output_val)

        return gw1, gw2, None


class HingeEmbeddingLoss(Function):
    def __init__(self, margin=1, size_average=True):
        super(HingeEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        buffer = input.new()
        buffer.resize_as_(input).copy_(input)
        buffer[torch.eq(target, -1.)] = 0
        output = buffer.sum()

        buffer.fill_(self.margin).add_(-1, input)
        buffer.clamp_(min=0)
        buffer[torch.eq(target, 1.)] = 0
        output += buffer.sum()

        if self.size_average:
            output = output / input.nelement()

        self.save_for_backward(input, target)
        return input.new((output,))

    def backward(self, grad_output):
        input, target = self.saved_tensors
        grad_input = input.new().resize_as_(input).copy_(target)
        grad_input[torch.mul(torch.eq(target, -1), torch.gt(input, self.margin))] = 0

        if self.size_average:
            grad_input.mul_(1. / input.nelement())

        if grad_output[0] != 1:
            grad_input.mul_(grad_output[0])

        return grad_input, None


class MarginRankingLoss(Function):
    def __init__(self, margin=1, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, y):
        _output = input1.clone()
        _output.add_(-1, input2)
        _output.mul_(-1).mul_(y)
        _output.add_(self.margin)
        _output.clamp_(min=0)
        output = _output.sum()

        if self.size_average:
            output = output / y.size(0)

        self.save_for_backward(input1, input2, y)
        return input1.new((output,))

    def backward(self, grad_output):
        input1, input2, y = self.saved_tensors
        grad_input1 = input1.new().resize_as_(input1)
        grad_input2 = input2.new().resize_as_(input2)

        dist = input1.clone()
        dist.add_(-1, input2)
        dist.mul_(-1).mul_(y)
        dist.add_(self.margin)
        mask = dist.ge(0)

        grad_input1.copy_(mask)
        grad_input1.mul_(-1).mul_(y)
        grad_input2.copy_(mask)
        grad_input2.mul_(y)

        if self.size_average:
            grad_input1.div_(y.size(0))
            grad_input2.div_(y.size(0))

        return grad_input1, grad_input2, None
