import math

import torch
from torch import nn


class GMM_Model(nn.Module):
    def __init__(self, N, K, mean=None, var=None, prior=None):
        super(GMM_Model, self).__init__()
        if mean is not None:
            self.mean = nn.Parameter(torch.from_numpy(mean).view(1, N, K))
            self.std = nn.Parameter(torch.sqrt(torch.from_numpy(var)).view(1, N, K))
        else:
            self.mean = nn.Parameter(torch.randn(1, N, K))
            self.std = nn.Parameter(torch.ones(1, N, K))
        self.N = N
        self.K = K

    def get_para(self):
        return self.mean, self.std

    def log_prob(self, data_mean, data_logvar, cond_prob, weight):
        term1 = torch.sum(-torch.log((self.std ** 2) * 2 * math.pi), dim=1) * 0.5
        term2 = torch.sum(-torch.div(
            torch.pow(data_mean.view(-1, self.N, 1) - self.mean, 2) + torch.exp(data_logvar).view(-1, self.N, 1),
            self.std ** 2), dim=1) * 0.5
        prob = term1 + term2
        log_p1 = torch.sum(torch.mul(prob, cond_prob), dim=1)
        log_p = torch.sum(torch.mul(log_p1, weight))
        return log_p

    def compute_prob(self, data):
        prob = torch.exp(torch.sum(
            -torch.log((self.std ** 2) * 2 * math.pi) - torch.div(torch.pow(data.view(-1, self.N, 1) - self.mean, 2),
                                                                  self.std ** 2), dim=1) * 0.5)
        pc = torch.div(prob, (torch.sum(prob, dim=-1)).view(-1, 1) + 1e-10)
        return pc

    def compute_entropy(self, inputs, weight):
        entropy1 = torch.sum(-torch.mul(inputs, torch.log(inputs + 1e-10)), dim=-1)
        entropy = torch.sum(torch.mul(entropy1, weight))
        return entropy

    def reg(self):
        return torch.sum(
            torch.pow(torch.sum(torch.pow(self.mean, 2), dim=1) - self.K * torch.ones(1, self.K).cuda(), 2))

    def forward(self):
        pass


def training_p(num_neighbors):
    activation = nn.ReLU()
    resume = ""


if __name__ == '__main__':
    training_p(21)
