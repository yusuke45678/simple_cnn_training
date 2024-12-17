import os
# from typing import Tuple
import av  # PyAVを利用
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
import torch
from typing import List, Tuple
from torchvision.transforms import functional as F
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
import pickle


class MyVideoDataset(Dataset):
    def __init__(self, root_dir: str, num_frames: int, transform: transforms):
        """
        動画データセットクラス

        Args:
            root_dir (str): 動画が格納されたフォルダパス。
            num_frames (int): 各動画から抽出するフレーム数。
            transform (transforms): 前処理。
        """
        # self.dataset_iter = iter(self.data)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform

        # 動画ファイルをリスト化
        self.video_paths = []
        self.labels = []

        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith((".mp4", ".avi", ".mkv")):  # 動画ファイルの拡張子をチェック
                    # 動画のパスをリストに追加
                    self.video_paths.append(os.path.join(root, filename))
                    # 対応するラベルをリストに追加
                    # ラベルはファイルパスから"ApplyEyeMakeup"を抽出
                    label = root.split(os.sep)[-1]  # フォルダ名を抽出
                    self.labels.append(label)

        encoder = LabelEncoder()
        self.labels = encoder.fit_transform(self.labels)
        self.labels = torch.tensor(self.labels)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

        # 追加
        self.video_paths = self.video_paths[:200]
        self.labels = self.labels[:200]

        # 動画の総数を属性として保存
        self.num_videos = len(self.video_paths)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # PyAVを用いて動画フレームを読み込む
        container = av.open(video_path)
        frames = []
        frame_count = 0
        for frame in container.decode(video=0):
            # img = F.to_tensor(frame.to_image())
            # img = frame.to_image()
            img = frame.to_ndarray(format="rgb24")
            # if self.transform:
            #     img = self.transform(img)
            frames.append(img)
            frame_count += 1
            # 指定したフレーム数に達したら終了
            if frame_count >= self.num_frames:
                break

        # frames = [self.transform(frame) for frame in frames]

        # 変換を適用
        # frames = [self.transform(frame) for frame in frames]
        # frames = torch.stack(frames)
        # frames = torch.tensor(frame)
        # frames = frames.permute(1, 0, 2, 3)  # TCHW -> CTHW (Time, Channel, Height, Width）
        frames = np.stack(frames, axis=0)
        frames = torch.tensor(frames, dtype=torch.float32)  # THWC
        frames = frames.permute(3, 0, 1, 2)  # THWC -> CTHW

        # 変換を適用
        if self.transform:
            batch_dict = {"video": frames, "label": label, "video_path": video_path}  # データを辞書形式に
            batch_dict = self.transform(batch_dict)
            frames = batch_dict["video"]
            label = batch_dict["label"]
            video_path = batch_dict["video_path"]

        return {"video": frames, "label": label, "video_path": video_path}

    # def __iter__(self):
    #     # イテレータの初期化
    #     self.dataset_iter = iter(self.data)
    #     return self
#
    # def __next__(self):
    #     try:
    #         return next(self.dataset_iter)
    #     except StopIteration:
    #         raise StopIteration  # 正しく終了処理をする

# def video_folder(
#     root: str,
#     train_dir: str,
#     val_dir: str,
#     batch_size: int,
#     num_workers: int,
#     num_frames: int,
#     train_transform: transforms,
#     val_transform: transforms
# ) -> Tuple[DataLoader, DataLoader, int]:
#     """
#     動画フォルダからデータローダーを作成

#     Args:
#         root (str): データセットのルートディレクトリ。
#         train_dir (str): 訓練データフォルダ名。
#         val_dir (str): 検証データフォルダ名。
#         batch_size (int): バッチサイズ。
#         num_workers (int): ワーカー数。
#         num_frames (int): 各動画から抽出するフレーム数。
#         train_transform (transforms): 訓練データ用の前処理。
#         val_transform (transforms): 検証データ用の前処理。

#     Returns:
#         DataLoader: 訓練用データローダー。
#         DataLoader: 検証用データローダー。
#         int: クラス数。
#     """
#     root_train_dir = os.path.join(root, train_dir)
#     root_val_dir = os.path.join(root, val_dir)
#     assert os.path.exists(root_train_dir) and os.path.isdir(root_train_dir)
#     assert os.path.exists(root_val_dir) and os.path.isdir(root_val_dir)

#     train_dataset = VideoDataset(
#         root_dir=root_train_dir,
#         num_frames=num_frames,
#         transform=train_transform,
#     )
#     val_dataset = VideoDataset(
#         root_dir=root_val_dir,
#         num_frames=num_frames,
#         transform=val_transform,
#     )

#     assert sorted(os.listdir(root_train_dir)) == sorted(os.listdir(root_val_dir))
#     n_classes = len(os.listdir(root_train_dir))

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=num_workers,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=num_workers,
#     )

#     return train_loader, val_loader, n_classes
