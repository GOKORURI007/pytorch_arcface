import math

import torch
from torch import nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False, **kwargs):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward1(self, embedding: torch.Tensor, ground_truth):
        """
        This implementation is from https://github.com/deepinsight/insightface, which takes
        53.55467200570274 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0 + 1e-7, 1 - 1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale

        loss = self.ce(output, ground_truth)
        return loss

    def forward2(self, embedding: torch.Tensor, ground_truth):
        """
        This implementation is from https://github.com/deepinsight/insightface, which takes
        75.16696600941941 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        Please noted that, different with forward1&3, this implementation ignore the samples that
        caused \theta + m > \pi to happen.
        """
        embedding = F.normalize(embedding)
        w = F.normalize(self.weight)
        cosine = F.linear(embedding, w)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        mask = torch.gather(cosine, 1, ground_truth.view(-1, 1)).view(-1)
        mask = torch.where(mask.acos_() + self.margin > math.pi, 0, 1)
        mask = torch.where(mask != 0)[0]
        m_hot = torch.zeros(mask.shape[0], cosine.shape[1], device=cosine.device)
        m_hot.scatter_(1, ground_truth[mask, None], self.margin)
        cosine.acos_()
        cosine += m_hot
        cosine.cos_().mul_(self.scale)
        loss = self.ce(cosine[mask], ground_truth[mask])
        return loss

    def forward3(self, embedding: torch.Tensor, ground_truth):
        """
        This implementation is modified from forward1, which takes
        54.52143200091086 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cosine, 1, ground_truth.view(-1, 1))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, cosine)
        else:
            phi = torch.where(pos > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), phi)
        cosine += one_hot
        cosine *= self.scale
        loss = self.ce(cosine, ground_truth)
        return loss
