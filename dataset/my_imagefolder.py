# from torch.utils.data import Dataset
# from pathlib import Path
# from typing import List, Tuple
# import torch
# from PIL import Image
# import os


# class MyDataset(Dataset):
#     def __init__(self, root, transform=None):
#         """
#         root_dir (string): 画像とクラスラベルが保存されているルートディレクトリ。
#         transform (callable, optional): 画像に適用する変換（例: データ拡張）。
#         """
#         # self.root = root_dir
#         self.transform = transform
#         self.class_names = os.listdir(root, 'train')  # クラス名一覧
#         self.class_names.sort()  # クラス名をソートして、一貫性を保つ
#
#         # クラスラベルを文字列から整数にマッピング
#         self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
#
#         # 画像とラベルを格納するリストを作成
#         self.img_paths = []
#         self.labels = []
#
#         for class_name in self.class_names:
#             class_folder = os.path.join(root, 'train', class_name, 'images')
#             for img_name in os.listdir(class_folder):
#                 img_path = os.path.join(class_folder, img_name)
#                 self.img_paths.append(img_path)
#                 self.labels.append(self.class_to_idx[class_name])
#
#     def __len__(self):
#         """
#         データセットのサンプル数を返す。
#         """
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         """
#         指定されたインデックスの画像とラベルを返す。
#         """
#         img_path = self.img_paths[idx]
#         label = self.labels[idx]
#
#         # 画像を読み込む
#         img = Image.open(img_path).convert('RGB')
#
#         # 変換が指定されている場合は適用
#         if self.transform:
#             img = self.transform(img)
#
#         return img, label
#

# class MyDataset(Dataset):
#    def __init__(self, root: str, transform) -> None:
#        super().__init__()
#        self.transforms = transform
#        # fix to original data
#        self.data = list(Path(root).glob("**/*.JPEG"))
#        self.classes = self.findclasses(root)
#        self.class_to_idx = self.classes[1]
#        self.n_classes = len(self.class_to_idx)
#    # ここで取り出すデータを指定している
#
#    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
#        path = self.data[index]
#        data = Image.open(path)
#        # label = os.path.basename(os.path.dirname(path))
#        label_name = os.path.basename(os.path.dirname(path))
#        # クラス名を整数ラベルに変換
#        label = self.class_to_idx.get(label_name, -1)  # 存在しない場合は -1 に設
#        # 範囲外ラベルを検出し補正
#        if label == -1 or label < 0 or label >= self.n_classes:
#            raise ValueError(
#                f"Label '{label_name}' (mapped to {label}) is out of range for {self.n_classes} classes. "
#                f"Ensure the folder names in {os.path.dirname(path)} match the class dictionary.")
#
#        # データの変形 (transforms)
#        data = self.transforms(data)
#        return data, label_name
#
#    def findclasses(self, root):
#        # 指定フォルダ内のフォルダのリストを名前順にソート
#        dirs = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
#        # フォルダ名とインデックスのペア
#        classdict = {name: index for index, name in enumerate(dirs)}
#        # この method がないと DataLoader を呼び出す際にエラーを吐かれる
#        return dirs, classdict
#
#    def __len__(self) -> int:
#        return len(self.data)

import os
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (str): データセットのルートディレクトリ（例: Tiny-ImageNet/train）
            transform (callable, optional): 画像に適用する変換処理
        """
        self.root_dir = root
        self.transform = transform

        # フォルダごとにラベル付け
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 画像パスとラベルのリストを作成
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path.endswith(('.jpg', '.jpeg', '.png', '.JPEG')):  # 画像ファイルの拡張子チェック
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): インデックス
        Returns:
            tuple: (画像, ラベル)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 画像を開く
        image = Image.open(img_path).convert("RGB")

        # 必要に応じて変換処理を適用
        if self.transform:
            image = self.transform(image)

        return image, label
