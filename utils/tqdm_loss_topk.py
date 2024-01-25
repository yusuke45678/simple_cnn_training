from typing import Tuple

from tqdm import tqdm


class TqdmLossTopK(tqdm):

    def set_postfix_str_loss_topk(
        self,
        global_step: int,
        loss: float,
        topk_values: Tuple[float, ...],
        topk: Tuple[int, ...] = (1, 5),
    ) -> str:
        assert len(topk) == len(topk_values)

        postfix_str = f"step={global_step:d}, "
        postfix_str += f"loss={loss:6.4e} "
        for k, value in zip(topk, topk_values):
            postfix_str += f"top{k}={value:6.2f} "

        self.set_postfix_str(postfix_str)

        return postfix_str
