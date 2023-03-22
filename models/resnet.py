import random
import torch.linalg
import torch.nn as nn
from torch.nn import functional as F


def fire_function(gamma):
    class ZIF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = (input >= 0).float()
            ctx.save_for_backward(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (input, ) = ctx.saved_tensors
            grad_input = grad_output.clone()
            tmp = (input.abs() < gamma/2).float() / gamma
            grad_input = grad_input * tmp
            return grad_input, None

    return ZIF.apply


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = 0

    def forward(self, x):
        self.mem = self.mem * self.tau + x
        spike = fire_function(self.gamma)(self.mem - self.thresh)
        self.mem = (1 - spike) * self.mem
        return spike


class ASConv2d(nn.Conv2d):

    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=0, bias=False):
        super(ASConv2d, self).__init__(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                       groups=1, bias=bias, dilation=1)
        self.use_ann = True

    def initialize(self):
        # initialize the U, V, sigma_a/snn based on the weight
        c_o, c_in, k, k = self.weight.shape
        org_weight = self.weight.data.reshape(k, k, c_o, c_in)
        U, sigma, Vh = torch.linalg.svd(org_weight, full_matrices=False)
        self.register_parameter('U', nn.Parameter(U))
        self.register_parameter('V', nn.Parameter(Vh))
        self.register_parameter('sigma_ann', nn.Parameter(sigma))
        self.register_parameter('sigma_snn', nn.Parameter(sigma.clone()))

    def forward(self, x):
        if self.use_ann is True:
            weight_ann = (self.U @ torch.diag_embed(self.sigma_ann) @ self.V).permute([2, 3, 0, 1])
            return self._conv_forward(x, weight_ann, self.bias)
        else:
            weight_snn = (self.U @ torch.diag_embed(self.sigma_snn) @ self.V).permute([2, 3, 0, 1])
            return self._conv_forward(x, weight_snn, self.bias)


class ASLinear(nn.Linear):

    def __init__(self, *args):
        super(ASLinear, self).__init__(*args)
        self.use_ann = True

    def initialize(self):
        org_weight = self.weight.data
        self.register_parameter('weight_ann', nn.Parameter(org_weight))
        self.register_parameter('weight_snn', nn.Parameter(org_weight.clone()))
        delattr(self, 'weight')

    def forward(self, x):
        if self.use_ann is True:
            return F.linear(x, self.weight_ann, self.bias)
        else:
            return F.linear(x, self.weight_snn, self.bias)


class ASBatchNorm2d(nn.Module):
    def __init__(self, n_channel):
        super(ASBatchNorm2d, self).__init__()
        self.use_ann = True
        self.bn_ann = nn.BatchNorm2d(n_channel)
        self.bn_snn = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        if self.use_ann:
            return self.bn_ann(x)
        else:
            return self.bn_snn(x)


class ASAct(nn.Module):
    def __init__(self):
        super(ASAct, self).__init__()
        self.use_ann =True
        self.act_ann = nn.ReLU(True)
        self.act_snn = LIFSpike()

    def forward(self, x):
        if self.use_ann:
            return self.act_ann(x)
        else:
            return self.act_snn(x)


def conv3x3(in_planes, out_planes, stride=1):
    return ASConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    return ASConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out // 4
    return nn.Sequential(
        ASConv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        ASBatchNorm2d(middle_channel),
        ASAct(),

        ASConv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        ASBatchNorm2d(middle_channel),
        ASAct(),

        ASConv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        ASBatchNorm2d(channel_out),
        ASAct(),
    )


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ASBatchNorm2d(planes)
        self.relu1 = ASAct()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ASBatchNorm2d(planes)
        self.downsample = downsample
        self.relu2 = ASAct()
        self.stride = stride

    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu2(output)
        return output


class Multi_ResNet(nn.Module):
    """Resnet model
    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, num_classes=1000):
        super(Multi_ResNet, self).__init__()
        self.inplanes = 64
        self.T = 2
        self.conv1 = ASConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ASBatchNorm2d(self.inplanes)
        self.relu = ASAct()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bottleneck1_1 = branchBottleNeck(64 * block.expansion, 512 * block.expansion, kernel_size=8)
        self.middle_fc1 = ASLinear(512 * block.expansion, num_classes)

        self.bottleneck2_1 = branchBottleNeck(128 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.middle_fc2 = ASLinear(512 * block.expansion, num_classes)

        self.bottleneck3_1 = branchBottleNeck(256 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.middle_fc3 = ASLinear(512 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ASLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.named_modules():
            if isinstance(m, (ASConv2d, ASLinear)):
                print('Initialize layer: {}'.format(name))
                m.initialize()

    def use_ann_mode_tag(self, tag=True):
        self.use_ann = tag
        for m in self.modules():
            if isinstance(m, (ASConv2d, ASLinear, ASBatchNorm2d, ASAct)):
                m.use_ann = tag

    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                ASBatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes))

        return nn.Sequential(*layer)

    def one_time_forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        middle_output1 = self.bottleneck1_1(x)
        middle_output1 = self.avgpool(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        x = self.layer2(x)
        middle_output2 = self.bottleneck2_1(x)
        middle_output2 = self.avgpool(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x = self.layer3(x)
        middle_output3 = self.bottleneck3_1(x)
        middle_output3 = self.avgpool(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)

        x = self.layer4(x)
        x = self.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea

    def forward(self, x, snn_only=False):
        for m in self.modules():
            if isinstance(m, LIFSpike):
                m.mem = 0

        if snn_only:
            self.use_ann_mode_tag(tag=False)
            all_outputs = []
            for i in range(self.T):
                outputs = self.one_time_forward(x)
                all_outputs += [outputs]
            return [sum([outputs[i] for outputs in all_outputs]) for i in range(8)]
        else:
            self.use_ann_mode_tag(True)
            ann_outputs = self.one_time_forward(x)
            self.use_ann_mode_tag(tag=False)
            all_outputs = []
            for i in range(self.T):
                outputs = self.one_time_forward(x)
                all_outputs += [outputs]
            snn_outputs = [sum([outputs[i] for outputs in all_outputs]) for i in range(8)]
            return ann_outputs, snn_outputs


def multi_resnet34_kd(num_classes=1000):
    return Multi_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def multi_resnet18_kd(num_classes=1000):
    return Multi_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

