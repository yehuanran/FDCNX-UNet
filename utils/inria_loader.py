from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)
#我加的
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
#图像尺寸
height = 512
width = 512
batch_size = 6
def decode_segmap(seg, is_one_hot=False):
    colors = torch.tensor([
            [0, 0, 0],
            [255, 255, 255],
        ], dtype=torch.uint8)
    if is_one_hot:
        seg = torch.argmax(seg, dim=0)
    # convert classes to colors
    seg_img = torch.empty((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8)
    for c in range(colors.shape[0]):
        #c是0和1
        seg_img[seg == c, :] = colors[c]
    return seg_img.permute(2, 0, 1)
from torch.utils import data
class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        self.n_classes = 2
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        #cv2.imshow(image)
        #cv2.waitKey()
        mask = load_mask(img_file_name, self.problem_type)
        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary': #走的这个
                transform = transforms.ToTensor()
                #原来是这样：return transform(image), torch.from_numpy(np.expand_dims(mask, 0)).long()，现在我不需要增加新维度了
                return transform(image), torch.from_numpy(mask).long()
            else:
                transform = transforms.ToTensor()
                return transform(image), torch.from_numpy(mask).long()
        else:
            transform = transforms.ToTensor()
            return transform(image), str(img_file_name)
def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = 255
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = 85
    elif problem_type == 'instruments':
        factor = 32
        mask_folder = 'instruments_masks'

    # mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'tif'),0)
    mask = cv2.imread(str(path).replace('images', mask_folder), 0)
    return (mask / factor).astype(np.uint8)
def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=32):
    return DataLoader(
        dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
        shuffle=False,
        num_workers=12,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

from pathlib import Path
data_path = Path('data')
train_path = data_path / 'INRIA'
val_path = data_path / 'INRIA'

# print(f"Data path exists: {data_path.exists()}")
# print(f"Train path exists: {train_path.exists()}")
# print(f"Validation path exists: {val_path.exists()}")
def get_split():
    folds = [1,2,3,4,5]
    train_file_names = []
    val_file_names = []
    # chengshi = ["austin","chicago","kitsap","tyrol-w","vienna"]

    for i in range(1,37):
        if i in folds:
            val_file_names += list((val_path / ( "austin"+str(i)) / 'images').glob('*'))
            val_file_names += list((val_path / ("chicago" + str(i)) / 'images').glob('*'))
            val_file_names += list((val_path / ("kitsap" + str(i)) / 'images').glob('*'))
            val_file_names += list((val_path / ("tyrol-w" + str(i)) / 'images').glob('*'))
            val_file_names += list((val_path / ("vienna" + str(i)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ( "austin"+str(i)) / 'images').glob('*'))
            train_file_names += list((train_path / ("chicago" + str(i)) / 'images').glob('*'))
            train_file_names += list((train_path / ("kitsap" + str(i)) / 'images').glob('*'))
            train_file_names += list((train_path / ("tyrol-w" + str(i)) / 'images').glob('*'))
            train_file_names += list((train_path / ("vienna" + str(i)) / 'images').glob('*'))
    return train_file_names, val_file_names

train_file_names, val_file_names = get_split()
# print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

def train_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=height, min_width=width, p=1),
        RandomCrop(height=height, width=width, p=1),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1)
    ], p=p)

def val_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=height, min_width=width, p=1),
        CenterCrop(height=height, width=width, p=1),
        Normalize(p=1)
    ], p=p)
class InriaLoader():
    def __init__(self):
        # 这儿的batch_size是1，应该是等于外面我设置的batch_size,我这儿直接改了，试验过3090 batch_size=8跑不起来，6可以
        self.train_loader = make_loader(train_file_names, shuffle=False, transform=train_transform(p=1), problem_type='binary',
                               batch_size=batch_size)
        self.valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type='binary',
                               batch_size=batch_size)
        self.test_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type='binary',
                                        batch_size=batch_size)

    def get_loaders(self):
        return self.train_loader,self.valid_loader,self.test_loader