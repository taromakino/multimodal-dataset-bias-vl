import torch


EPSILON = 1e-8


def diag_gaussian_log_prob(x, mu, var, device):
    c = 2 * torch.pi * torch.ones(1).to(device)
    return (-0.5 * (torch.log(c) + var.log() + (x - mu).pow(2).div(var))).sum(-1)