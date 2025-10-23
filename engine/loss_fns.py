import torch

# def cwt_loss(inputs, positive, negative, temperature=1.):
#     eps = torch.finfo(inputs.dtype).eps
#     self_dist = torch.exp((inputs*positive).sum(-1)/temperature)
#     neg_dist = torch.exp(torch.mm(inputs, negative.T)/temperature).sum(-1)
#     return - (self_dist/(self_dist + neg_dist + eps)).log().mean()


def cwt_loss(inputs, positive, negative, temperature=1.0):
    log_self_dist = (inputs * positive).sum(-1) / temperature
    log_neg_dist = torch.mm(inputs, negative.T) / temperature

    ratio = log_neg_dist - log_self_dist[..., :, None]
    ratio = torch.cat((torch.zeros_like(log_self_dist)[..., :, None], ratio), 1)
    return torch.logsumexp(ratio, -1).mean()
