import argparse

from tqdm import tqdm

import torch
from torch import nn

from args import ArgParse
from dataset import configure_dataloader
from model import (
    configure_model,
    ModelConfig,
)
from setup import configure_optimizer, configure_scheduler

from logger import configure_logger
from utils import (
    save_to_checkpoint, save_to_comet,
    load_from_checkpoint,
)

from train import train, TrainConfig
from val import validation

import csv
import pandas as pd
import os
from datetime import datetime
import pickle


class TqdmEpoch(tqdm):
    def __init__(
            self,
            start_epoch: int,
            num_epochs: int,
            *args,
            **kwargs):
        super().__init__(
            range(start_epoch + 1, num_epochs + 1), *args, **kwargs
        )


def prepare_training(args: argparse.Namespace):
    """prepare training objects from args

    Args:
        args (argparse.Namespace): command line argmentrs

    Returns:
        a set of training objects
    """

    logger = configure_logger(
        logged_params=vars(args),
        model_name=args.model_name,
        disable_logging=args.disable_comet,
    )

    dataloaders = configure_dataloader(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=args.model_name,
        use_pretrained=args.use_pretrained,
        torch_home=args.torch_home,
        n_classes=dataloaders.n_classes,
    ))
    model = model.to(device)
    if args.use_dp:
        model = nn.DataParallel(model)  # type: ignore[assignment]

    optimizer = configure_optimizer(
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        model_params=model.parameters()
    )
    scheduler = configure_scheduler(
        optimizer=optimizer,
        use_scheduler=args.use_scheduler
    )

    train_config = TrainConfig(
        grad_accum_interval=args.grad_accum,
        log_interval_steps=args.log_interval_steps
    )

    if args.checkpoint_to_resume:
        (
            start_epoch,
            current_train_step,
            current_val_step,
            model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(  # type: ignore[assignment]
            args.checkpoint_to_resume,
            model,
            optimizer,
            scheduler,
            device
        )
    else:
        current_train_step = 1
        current_val_step = 1
        start_epoch = 0

    return (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    )


class ValidationChecker:
    def __init__(self, val_interval_epochs, num_epochs):
        self.val_interval_epochs = val_interval_epochs
        self.num_epochs = num_epochs

    def should_validate(self, current_epoch):
        return (
            current_epoch % self.val_interval_epochs == 0
            or current_epoch == self.num_epochs
        )


def main():

    args = ArgParse.get()

    (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    ) = prepare_training(args)

    val_checker = ValidationChecker(args.val_interval_epochs, args.num_epochs)

    train_output_high_loss_videos = []  # 追加

    with TqdmEpoch(
        start_epoch, args.num_epochs, unit='epoch',
    ) as progress_bar_epoch:
        for current_epoch in progress_bar_epoch:
            progress_bar_epoch.set_description(f"[epoch {current_epoch:03d}]")

            train_output = train(
                model,
                optimizer,
                scheduler,
                dataloaders.train_loader,
                current_train_step,
                current_epoch,
                logger,
                train_config
            )
            current_train_step = train_output.train_step

            # 保存する関数を追加
            # 関数を別.pyに書き込む
            # 保存先用のフォルダを作る json 形式で保存

            # 高い損失を持つ動画を表示
            print("High loss videos from training output:")
            for video_path, avg_loss, is_correct in train_output.high_loss_videos:
                # status = "Correct" if is_correct == 1 else "Incorrect"
                train_output_high_loss_videos.append((current_epoch, round(avg_loss, 4), is_correct, video_path))
                print(f"Loss: {avg_loss:.4f}, Correct: {is_correct}, Path: {video_path}")

            # 保存されたpickleファイルの読み込み
            with open("label_encoder.pkl", "rb") as f:
                train_output = pickle.load(f)

            # LabelEncoder を適切に復元
            label_encoder = train_output['label_encoder']  # 事前に LabelEncoder を保存している場合

            # 例: 動画ごとの予測詳細を出力
            for video_id, records in train_output['logits_labels'].items():
                print(f"Video ID: {video_id}")
                for record in records:
                    # 'record' から probabilies と label を取得
                    probabilities = record['probabilities']
                    label = record['label']  # 数値ラベル

                    # 数値ラベルを元の文字列ラベルに変換
                    original_label = label_encoder.inverse_transform([label])[0] if label is not None else None

                    # 出力
                    print(f"  Probabilities: {probabilities}")
                    print(f"  Label: {original_label}")

            if val_checker.should_validate(current_epoch):

                val_output = validation(
                    model,
                    dataloaders.val_loader,
                    current_val_step,
                    current_epoch,
                    logger,
                )
                current_val_step = val_output.val_step

                checkpoint_dict, _ = save_to_checkpoint(
                    args.save_checkpoint_dir,
                    current_epoch,
                    current_train_step,
                    current_val_step,
                    val_output.top1,
                    model,
                    optimizer,
                    scheduler,
                    logger
                )
                save_to_comet(
                    checkpoint_dict,
                    args.model_name,
                    logger
                )

    # 現在の日付と時刻を取得してファイル名に追加
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_folder = "my_log"
    csv_file_path = os.path.join(log_folder, f"{current_time}.csv")

    # データをDataFrameに変換
    df = pd.DataFrame(
        train_output_high_loss_videos,
        columns=["epoch", "Loss", "is_correct", "video_path"]
    )

    # CSVファイルに保存
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
