import torch
from typing import Tuple


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411

    Args:
        output (torch.tensor): model output of the batch
        target (torch.tensor): labels of the batch
        topk (tuple of int, optional):
            k for computing top-k accuracy. Defaults to (1,).
                topk=(1,) returns top1
                topk=(1,5) returns list of [top1, top5]

    Returns:
        int or List[int]: top-k accuracy
    """
    assert topk[0] == 1, "topk[0] should be top1"
    assert len(topk) >= 1

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res if len(res) > 1 else res[0]
