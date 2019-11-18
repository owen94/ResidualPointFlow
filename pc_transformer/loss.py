import torch

from chamfer_distance import ChamferDistance


def cd_loss(x, y, alpha=0.5):
    dist1, dist2 = ChamferDistance()(x, y)

    return alpha * torch.mean(dist1) + (1 - alpha) * torch.mean(dist2)


def cd_margin_loss(x, y, thres=0.01, alpha=0.5):
    dist1, dist2 = ChamferDistance()(x, y)
    dist1, dist2 = dist1[dist1 > thres], dist2[dist2 > thres]

    loss = torch.tensor(0).to(x)
    if dist1.size(0) > 0:
        loss += alpha * torch.mean(dist1)
    if dist2.size(0) > 0:
        loss += (1 - alpha) * torch.mean(dist2)
    return loss


def get_loss(loss_name, args):
    if loss_name == 'cd':
        return lambda x, y: cd_loss(x, y, alpha=args.alpha)
    elif loss_name == 'cd_margin':
        return lambda x, y: cd_margin_loss(x, y, thres=args.thres, alpha=args.alpha)
    else:
        raise NotImplementedError