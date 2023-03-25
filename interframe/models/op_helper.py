import torch
import numpy as np
import linklink as link


def count_flops(model, input_shape):

    flops_dict = {}

    def make_conv2d_hook(name):

        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[1] / m.stride[1] / m.groups
            flops_dict[name] = int(flops)

        return conv2d_hook

    hooks = []

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)

    input = torch.zeros(*input_shape)

    model.eval()
    with torch.no_grad():
        _ = model(input)

    model.train()
    total_flops = 0
    for k, v in flops_dict.items():
        print('module {}: {}M'.format(k, v/1e6))
        total_flops += v

    print('total FLOPS: {:.2f}M'.format(total_flops/1e6))

    for h in hooks:
        h.remove()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count

