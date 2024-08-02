import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        scores = self.layers(features)
        return scores


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class Domain_Discriminator(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Domain_Discriminator, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, features, inv):
        if inv is True:
            features = GradReverse.apply(features)
        y = self.class_classifier(features)
        return y


class Predictor(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False))
        
    def forward(self, x):
        p = self.predictor(x)
        return p

class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle =8192, k = 1024, hard=False):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k
        self.hard = hard

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       mask = self.bn(self.layers(f))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=self.hard)
           z = torch.maximum(mask,z)
       return z
    
