import torch
import torch.nn as nn


class styleLoss(nn.Module):
    def forward(self, input, target):
        # 分别计算结果图和风格图的Gram矩阵
        ib, ic, ih, iw = input.size()
        iF = input.view(ib, ic, -1)
        iMean = torch.mean(iF, dim=2)
        iCov = GramMatrix()(input)

        tb, tc, th, tw = target.size()
        tF = target.view(tb, tc, -1)
        tMean = torch.mean(tF, dim=2)
        tCov = GramMatrix()(target)

        # 对每一层的Gram矩阵差异计算损失，这里多计算了一个均值差异
        loss = nn.MSELoss(size_average=False)(iMean, tMean) + nn.MSELoss(size_average=False)(iCov, tCov)
        return loss / tb


class GramMatrix(nn.Module):
    """
    计算Gram矩阵，先将输入的feature map拉直为(batch=1, channel=32, *)，再计算每一个channel的协方差
    因此一个feature map的Gram矩阵的输出结果的size只和channel数相关( = channel * channel)
    """
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h * w)  # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f, f.transpose(1, 2))  # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c * h * w)


class LossCriterion(nn.Module):
    def __init__(self, style_layers, content_layers, style_weight, content_weight):
        super(LossCriterion, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        # 内容损失默认只有一层，可以和风格损失一样设置多层
        self.contentLosses = [nn.MSELoss()] * len(content_layers)
        self.styleLosses = [styleLoss()] * len(style_layers)

    def forward(self, tF, sF, cF):
        # 内容损失
        totalContentLoss = 0
        for i, layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i, cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # 风格损失
        totalStyleLoss = 0
        for i, layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i, sf_i)

        # 总损失是两者加权相加，权重在初始化时的style_weight, content_weight
        totalStyleLoss = totalStyleLoss * self.style_weight
        loss = totalStyleLoss + totalContentLoss

        return loss, totalStyleLoss, totalContentLoss
