from typing import Tuple

from tqdm import tqdm


class TqdmLossTopK(tqdm):

    def __init__(self):
        super().__init__()
        self.postfix_str = ""

    def set_postfix_str_loss_topk(
        self,
        global_step: int,
        loss: float,
        topk_values: Tuple[float, ...],
        topk: Tuple[int, ...] = (1, 5),
    ) -> str:
        assert len(topk) == len(topk_values)

        self.add_step(global_step)
        self.add_loss(loss)
        self.add_topk(topk_values, topk)

        self.set_postfix_str(self.postfix_str)
        return self.postfix_str

    def add_step(self, step: int) -> None:
        self.postfix_str += f"step={step:d}, "

    def add_loss(self, loss: float) -> None:
        self.postfix_str += f"loss={loss:6.4e} "

    def add_topk(
        self,
        topk_values: Tuple[float, ...],
        topk: Tuple[int, ...]
    ) -> None:
        for k, value in zip(topk, topk_values):
            self.postfix_str += f"top{k}={value:6.2f} "
