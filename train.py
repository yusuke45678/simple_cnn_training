from dataclasses import dataclass

import comet_ml
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import (
    compute_topk_accuracy,
    AvgMeterLossTopk,
    TqdmLossTopK,
)
from model import ClassificationBaseModel, get_device

from collections import defaultdict
from loss_analysis import get_high_loss_videos

import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    grad_accum_interval: int
    log_interval_steps: int


@dataclass
class TrainOutput:
    loss: float
    top1: float
    train_step: int
    high_loss_videos: list  # 高損失動画情報を追加
    logits_labels: dict  # 追加：動画IDをキーに予測確率とラベルのリストを記録する辞書


class LoggerChecker:
    def __init__(self, log_interval_steps):
        self.log_interval_steps = log_interval_steps

    def should_log(self, step):
        return step % self.log_interval_steps == 0


class OptimizerChecker:
    def __init__(self, grad_accum_interval):
        self.grad_accum_interval = grad_accum_interval

    def should_zero_grad(self, batch_index):
        return (
            self.grad_accum_interval == 1
            or batch_index % self.grad_accum_interval == 1
        )

    def should_update(self, batch_index):
        return batch_index % self.grad_accum_interval == 0


def train(
    model: ClassificationBaseModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    current_train_step: int,
    current_epoch: int,
    logger: comet_ml.Experiment,
    train_config: TrainConfig
) -> TrainOutput:
    """training loop for one epoch

    Args:
        model (ClassificationBaseModel): CNN model
        optimizer (Optimizer): optimizer
        scheduler (LRScheduler): learning rate (lr) scheduler
        loader (DataLoader): training dataset loader
        current_train_step (int): current step for training
        current_epoch (int): current epoch
        logger (comet_ml.Experiment): comet logger
        train_config (TrainInfo): information for training

    Returns:
        TrainOutput: train loss, train top1, steps for training
    """

    train_meters = AvgMeterLossTopk("train")

    # 動画IDと動画パスを管理する辞書
    video_id_to_path = {}
    video_loss_dict = defaultdict(list)  # key {loss{}, 認識(correct)(0.1){}}

    logits_labels = defaultdict(list)   # 予測確率とラベルの記録用

    # 動画IDとパスの対応を収集
    for idx in range(len(train_loader.dataset)):
        sample = train_loader.dataset[idx]
        video_id_to_path[f"video{idx}"] = sample["video_path"]

    model.train()
    device = get_device(model)
    optimizer_checker = OptimizerChecker(train_config.grad_accum_interval)
    logger_checker = LoggerChecker(train_config.log_interval_steps)

    # logits_labels = {}

    with TqdmLossTopK(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            leave=False,
            unit='step',
    ) as progress_bar_step:
        progress_bar_step.set_description("[train    ]")
        for batch_index, batch in progress_bar_step:

            # (BCHW, B) for images or (BCTHW, B) for videos torch.Size([8, 3, 16, 224, 224])
            data, labels = batch
            # data = batch["video"]
            # labels = batch["label"]

            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.size(0)

            # 動画IDを生成
            # video_ids = [f"batch{batch_index}_video{i}" for i in range(batch_size)]
            video_ids = [f"video{batch_index * batch_size + i}" for i in range(batch_size)]

            if optimizer_checker.should_zero_grad(batch_index):
                optimizer.zero_grad()

            outputs = model(data, labels=labels)
            loss = outputs.loss.mean()  # maen() is only for dp to gather loss
            loss.backward()

            train_topk = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))  # labelとlogitから予測できたかどうかを記録
            train_meters.update(loss, train_topk, batch_size)

            # losses = outputs.loss  # バッチ内の各サンプルの損失 (形状: [バッチサイズ])
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # 個々の損失を計算
            losses = loss_fn(outputs.logits, labels)              # 各サンプルの損失
            # loss = losses.mean()                                  # 平均損失を計算
            # loss.backward()                                       # 平均損失で逆伝播

            # ソフトマックスを使って予測確率を計算
            probabilities = F.softmax(outputs.logits, dim=-1)

            # 各動画の損失を記録
            for i, (video_id, sample_loss) in enumerate(zip(video_ids, losses)):
                video_loss_dict[video_id].append(sample_loss.item())  # 損失を記録
                # logits_labels[video_id] = (outputs.logits[i], labels[i])  # logits_labels に logits とラベルを記録
                logits_labels[video_id].append({
                    "probabilities": probabilities[i].tolist(),  # 予測確率
                    "label": labels[i].item()  # ラベル
                })

            # lossを動画IDごとに記録
            # for i, video_id in enumerate(video_ids):
            #     video_loss_dict[video_id].append(loss.item())
            #     logits_labels[video_id] = (outputs.logits[i], labels[i])  # 追加

            if logger_checker.should_log(current_train_step):
                progress_bar_step.set_postfix_str_loss_topk(
                    current_train_step, loss, train_topk
                )
                logger.log_metrics(
                    train_meters.get_step_metrics_dict(),
                    step=current_train_step,
                    epoch=current_epoch,
                )

            if optimizer_checker.should_update(batch_index):
                optimizer.step()
                current_train_step += 1

    scheduler.step()

    logger.log_metrics(
        train_meters.get_epoch_metrics_dict(),
        step=current_train_step,
        epoch=current_epoch,
    )

    # 高いlossを持つ動画を特定してログ
    # high_loss_videos = sorted(　.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:5]
    # log_message = f"High loss videos: {high_loss_videos}"
    # logger.log_text(log_message)
    # print(log_message)

    # 高いlossを持つ動画を特定
    # sorted_loss_videos = sorted(video_loss_dict.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:5]
    high_loss_videos = get_high_loss_videos(video_loss_dict, video_id_to_path, logits_labels)

    # 高いlossの動画ファイル名をターミナルに出力
    # log_message = f"Epoch {current_epoch} - High loss videos (top 5):"
    # logger.log_text(log_message)
    # for video_id, losses in sorted_loss_videos:
    #     avg_loss = sum(losses) / len(losses)  # lossの平均
    #     video_path = video_id_to_path.get(video_id, "Unknown video path")
    #     log_message = f"Video ID: {video_id}, Loss: {avg_loss}, Video Path: {video_path}"
    #     logger.log_text(log_message)

    return TrainOutput(
        loss=train_meters.loss_meter.avg,
        top1=train_meters.topk_meters[0].avg,  # top1: topk[0] should be 1
        train_step=current_train_step,
        high_loss_videos=high_loss_videos,
        logits_labels=dict(logits_labels)  # 記録を含めて出力
    )
