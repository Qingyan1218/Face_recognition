import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(nn.Module):
    """继承自Function的损失函数计算必须是静态方法，即不含self的函数，
    这里需要改成继承自nn.Module，该写法同常规一致"""

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative, reduction = None):
        # 计算正样本间的距离
        positive_dis = self.pdist(anchor,positive)
        # 计算正负样本间的距离
        negative_dis = self.pdist(anchor,negative)
        # triplet loss的定义
        original_loss = positive_dis - negative_dis + self.margin
        # 小于0.0的值是不需要优化的，因此损失是0.0
        # loss[loss<0] = 0.0
        loss = torch.clamp(original_loss, min=0.0)
        if reduction:
            loss = torch.mean(loss)
        return loss

if __name__ == '__main__':
    # torch.manual_seed(20) # tensor(0.7353, grad_fn=<MeanBackward0>)
    anchor = torch.randn(4, 128, requires_grad=True)
    positive = torch.randn(4, 128, requires_grad=True)
    negative = torch.randn(4, 128, requires_grad=True)
    result = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, reduction='mean')
    print(result)
    tri_loss = TripletLoss(1.0)
    result = tri_loss(anchor, positive, negative , reduction = 'mean')
    print(result)