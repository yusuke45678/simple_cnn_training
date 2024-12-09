import os
import av
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from torchvision.transforms import functional as F


class MyVideoDataset(Dataset):
    def __init__(self, root_dir: str, num_frames: int, transform: transforms = None):
        """
        Args:
            root_dir (str): 動画ファイルが保存されているルートディレクトリ
            num_frames (int): 各動画から抽出するフレーム数
            transform (transforms): フレームに適用する変換（オプション）
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform

        # ディレクトリ内のすべての動画ファイルのパスを取得
        self.video_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files
            if file.endswith(".avi")
        ]

        if not self.video_paths:
            raise ValueError(f"No video files found in {root_dir}")

        self.num_videos = len(self.video_paths)

    def __len__(self):
        """データセットのサイズを返す"""
        return len(self.video_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): データセット内のインデックス

        Returns:
            dict: 動画フレームのテンソルと対応するラベル（フォルダ名をラベルとする）
        """
        video_path = self.video_paths[index]
        label = os.path.basename(os.path.dirname(video_path))  # 親フォルダ名をラベルとして利用

        # PyAV を使って動画を読み込む
        video_container = av.open(video_path)
        frames = []

        # 動画から指定数のフレームを抽出
        for frame in video_container.decode(video=0):
            # フレームを PIL.Image に変換
            img = frame.to_image()
            frames.append(img)
            if len(frames) == self.num_frames:
                break

        # フレーム数が足りない場合は補完する
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # 最後のフレームを繰り返し使用

        # 変換を適用
        # if self.transform:
        #     frames = [self.transform(frame) for frame in frames]

        # テンソルとして返す（複数フレームを 1つのテンソルにスタック）
        # frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)
        frames = self.transform(frames)

        return {"frames": frames, "label": label}
