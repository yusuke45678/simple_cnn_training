import torch
from typing import Tuple, Literal


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float | torch.Tensor, n: int = 1):
        """update the statistics

        Args:
            value (float or torch.Tensor): a value used for averaging
            n (int, optional): multiplier of the curent value for averaging. Defaults to 1.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class AvgMeterLossTopk:
    def __init__(
        self,
        mode_name: Literal["train", "val"],
        topk: Tuple[int] = (1, 5)
    ):
        self.mode_name = mode_name
        self.topk = topk
        self.loss_meter = AverageMeter()
        self.topk_meter = []
        for _ in topk:
            self.topk_meter.append(AverageMeter())

    def update(
        self,
        loss: float | torch.Tensor,
        topk_value: Tuple[float | torch.Tensor],
        batch_size: int = 1
    ):
        self.loss_meter.update(loss, batch_size)
        for meter, value in zip(self.topk_meter, topk_value):
            meter.update(value, batch_size)

    def get_set_postfix_str(self, global_step: int) -> str:
        """generate string for tqdm pbar.set_postfix_str

        Args:
            global_step (int): global_step

        Returns:
            str: set_postfix_str string
        """
        postfix_str = f"step={global_step:d}, "
        postfix_str += f"loss={self.loss_meter.value:6.4e}"
        postfix_str += f"({self.loss_meter.avg:6.4e}), "
        for meter, k in zip(self.topk_meter, self.topk):
            postfix_str += f"top{k}={meter.value:6.2f}({meter.avg:6.2f}), "
        return postfix_str

    def get_step_metrics_dict(self) -> dict:
        """generate dict of step value statistics
        for comet_ml.experiment.log_metrics

        Returns:
            dict: dict to be logged by log_metrics
        """
        metrics_dict = {
            f"{self.mode_name}_loss_step": self.loss_meter.value,
        }
        for meter, k in zip(self.topk_meter, self.topk):
            metrics_dict[f"{self.mode_name}_top{k}_step"] = meter.value
        return metrics_dict

    def get_epoch_metrics_dict(self) -> dict:
        """generate dict of epoch avgerage statistics
        for comet_ml.experiment.log_metrics

        Returns:
            dict: dict to be logged by log_metrics
        """
        metrics_dict = {
            f"{self.mode_name}_loss_epoch": self.loss_meter.avg,
        }
        for meter, k in zip(self.topk_meter, self.topk):
            metrics_dict[f"{self.mode_name}_top{k}_epoch"] = meter.avg
        return metrics_dict
