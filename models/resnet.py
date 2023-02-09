import random
import torch.linalg
from models.layers import *
from IPython import embed

class ASConv2d(nn.Conv2d):

    def __init__(self, in_planes, out_planes, stride=1, ksize=3, padding=1):
        super(ASConv2d, self).__init__(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=padding,
                                       groups=1, bias=False, dilation=1)
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


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SeqToANNContainer(ASConv2d(inplanes, planes, stride))
        self.bn1_ann = nn.BatchNorm2d(planes)
        self.bn1_snn = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.spike1 = LIFSpike()
        self.relu1 = nn.ReLU(True)

        self.conv2 = SeqToANNContainer(ASConv2d(planes, planes))
        self.bn2_ann = nn.BatchNorm2d(planes)
        self.bn2_snn = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.spike2 = LIFSpike()
        self.relu2 = nn.ReLU(True)

        self.use_ann = True

        if (stride != 1 or inplanes != planes):
            # Projection also with pre-activation according to paper.
            self.downsample = SeqToANNContainer(ASConv2d(inplanes, planes, stride=2, ksize=1, padding=0))
            self.downbn_ann = nn.BatchNorm2d(planes)
            self.downbn_snn = SeqToANNContainer(nn.BatchNorm2d(planes))

    def forward(self, x):
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.downbn_ann(residual) if self.use_ann else self.downbn_snn(residual)
        else:
            residual = x

        if not self.use_ann:
            out = self.bn2_snn(self.conv2(self.spike1(self.bn1_snn(self.conv1(x)))))
            return self.spike2(out+residual)
        else:
            out = self.bn2_ann(self.conv2(self.relu1(self.bn1_ann(self.conv1(x)))))
            return self.relu2(out+residual)


class ResStage(nn.Module):

    def __init__(self, inplane, plane, stride=1, rep=2, num_class=10):
        super().__init__()
        assert rep >= 1
        self.feature1 = BasicBlock(inplane, plane, stride=stride)
        self.features = nn.Sequential(*[BasicBlock(plane, plane) for i in range(rep-1)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_snn = nn.Linear(plane, num_class)
        self.fc_ann = nn.Linear(plane, num_class)
        self.use_ann = True

    def forward(self, x):
        out = self.features(self.feature1(x))
        #embed()
        if not self.use_ann:
            logit = self.avg_pool(out.mean(1))
            logit = logit.flatten(1)
            pred = self.fc_snn(logit)
        else:
            logit = self.avg_pool(out).flatten(1)
            pred = self.fc_ann(logit)

        return out, logit, pred


class ResNet(nn.Module):

    def __init__(self, block, num_classes=10, in_c=3):
        super().__init__()
        #self.conv1 = SeqToANNContainer(nn.Conv2d(in_c, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv1 = SeqToANNContainer(ASConv2d(in_c, 64, ksize=3, stride=1, padding=1))
        #self.conv1 = ASConv2d(in_c, 64, ksize=3, stride=1, padding=1)
        self.bn_ann = nn.BatchNorm2d(64)
        self.bn_snn = SeqToANNContainer(nn.BatchNorm2d(64))
        self.spike = LIFSpike()
        self.relu = nn.ReLU(True)
        self.stage1 = ResStage(inplane=64, plane=64, stride=1, rep=block[0], num_class=num_classes)
        self.stage2 = ResStage(inplane=64, plane=128, stride=2, rep=block[1], num_class=num_classes)
        self.stage3 = ResStage(inplane=128, plane=256, stride=2, rep=block[2], num_class=num_classes)
        self.stage4 = ResStage(inplane=256, plane=512, stride=2, rep=block[3], num_class=num_classes)
        self.use_ann = True
        self.T = 4
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
        self.mse_loss = nn.MSELoss()
        self.lambda1 = 1
        self.lambda2 = 0.25

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
        for m in self.modules():
            if isinstance(m, ASConv2d):
                m.initialize()

    def use_ann_mode_tag(self, tag=True):
        self.use_ann = tag
        for m in self.modules():
            if isinstance(m, (ASConv2d, BasicBlock, ResStage)):
                m.use_ann = tag

    def train_forward(self, x, y, ):
        #embed()
        self.use_ann_mode_tag(True)
        out = self.relu(self.bn_ann(self.conv1(x)))
        out, logit1, pred1 = self.stage1(out)
        out, logit2, pred2 = self.stage2(out)
        out, logit3, pred3 = self.stage3(out)
        out, logit4, pred4 = self.stage4(out)

        self.use_ann_mode_tag(False)
        #embed()
        x2 = add_dimention(x.clone(), self.T)
        outs = self.spike(self.bn_snn(self.conv1(x2)))
        outs, logit1s, pred1s = self.stage1(outs)
        outs, logit2s, pred2s = self.stage2(outs)
        outs, logit3s, pred3s = self.stage3(outs)
        outs, logit4s, pred4s = self.stage4(outs)

        # ce_loss
        #loss_ce = sum([eval('self.ce_loss(pred{}, y)'.format(i)) for i in range(1, 5)])
        loss_ce = sum([self.ce_loss(pred1, y),self.ce_loss(pred2, y),self.ce_loss(pred3, y),self.ce_loss(pred4, y)])
        #loss_ce += sum([eval('self.ce_loss(pred{}s, y)'.format(i)) for i in range(1, 5)])
        loss_ce = loss_ce + sum([self.ce_loss(pred1s, y),self.ce_loss(pred2s, y),self.ce_loss(pred3s, y),self.ce_loss(pred4s, y)])
        # kl loss
        #loss_kl = sum([eval('self.kl_loss(pred{}s, pred{}.detach())'.format(i, i)) for i in range(1, 5)])
        loss_kl = sum([self.kl_loss(pred1s, pred1.detach()),self.kl_loss(pred2s, pred2.detach()),self.kl_loss(pred3s, pred3.detach()),self.kl_loss(pred4s, pred4.detach())])
        # norm loss
        #loss_norm = sum([eval('self.mse_loss(pred{}s, pred{}.detach())'.format(i, i)) for i in range(1, 5)])
        loss_norm = sum([self.mse_loss(pred1s, pred1.detach()),self.mse_loss(pred2s, pred2.detach()),self.mse_loss(pred3s, pred3.detach()),self.mse_loss(pred4s, pred4.detach())])
        loss = loss_ce + self.lambda1 * loss_kl + self.lambda2 * loss_norm
        #loss = sum([self.ce_loss(pred1, y),self.ce_loss(pred2s, y),self.ce_loss(pred3s, y),self.ce_loss(pred4s, y)])
        return loss

    def test_forward(self, x):
        # used for test
        self.use_ann_mode_tag(False)
        x = add_dimention(x, self.T)
        outs = self.spike(self.bn_snn(self.conv1(x)))
        outs, _, _ = self.stage1(outs)
        outs, _, _ = self.stage2(outs)
        outs, _, _ = self.stage3(outs)
        outs, logits, pred = self.stage4(outs)
        return pred


def resnet18(*args, **kwargs):
    return ResNet([2, 2, 2, 2], *args, **kwargs)


def resnet34(*args, **kwargs):
    return ResNet([3, 4, 6, 3], *args, **kwargs)

