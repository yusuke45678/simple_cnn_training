# import os
# from pathlib2 import Path
# from torchvision.transforms import ToTensor, Compose

from torch.utils.data import DataLoader
from my_videofolder import MyVideoDataset

from torchvision import transforms
from pytorchvideo.data import labeled_video_dataset
# root_dir = '/mnt/NAS-TVS872XT/dataset/Kinetics400/train'
root_dir = '/mnt/NAS-TVS872XT/dataset-lab/UCF101.split/split01/train'

# paths = []
# labels = []
# for video_file in video_dir_path.glob('*.mp4'):
# file_name = video_file.stem
# label_id = file_name[:2]
#
# full_path = os.path.join(video_dir, video_file.name)
# paths.append(full_path)
# labels.append(label_id)
#
transform = transforms.Resize((224, 224))
# = transforms.Compose([
#     # transforms.Resize((224, 224)),  # サイズ変更
#     transforms.ToTensor()          # テンソル変換
# ])

data_set = MyVideoDataset(root_dir, num_frames=16, transform=transform)
print(f'{len(data_set)=}')

data_loader = DataLoader(data_set, batch_size=8, shuffle=True)

for frames, labels in data_loader:
    print(f'{len(frames)=}')
    print(f'{frames.shape=}')
    print(f'labels={labels}')
    break
