from torchvision import transforms
from torch.utils.data import DataLoader
from my_imagefolder import MyDataset  # ファイル名を your_module に置き換えてください

# データセットのパス
# /mnt/HDD4TB-3/yamada/simple_cnn_training/dataset/ImageNet
dataset_path = "/mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/train"

# データ変換
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像サイズを224x224にリサイズ
    transforms.ToTensor()           # 画像をテンソルに変換
])

# MyDataset インスタンスの作成
dataset = MyDataset(root=dataset_path, transform=transform)

# DataLoader の作成
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f'{len(dataset)=}')

# データを取り出して確認
for images, labels in dataloader:
    print("Images shape:", images.shape)  # 画像データの形状
    print("Labels:", labels)              # ラベル
    break  # 最初のバッチだけ表示
