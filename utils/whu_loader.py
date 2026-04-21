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
height = 512
width = 512
def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'label'
        factor = 255
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = 85
    elif problem_type == 'instruments':
        factor = 32
        mask_folder = 'instruments_masks'
    path = str(path).replace('image', mask_folder).replace('tiff', 'tif')
    mask = cv2.imread(path,0)

    return (mask / factor).astype(np.uint8)
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

def test_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=height, min_width=width, p=1),
        CenterCrop(height=height, width=width, p=1),
        Normalize(p=1)
    ], p=p)
from pathlib import Path
data_path = Path('data')
train_path = data_path / 'WHU'/'train'
val_path = data_path / 'WHU'/'test'
test_path = data_path / 'WHU'/'test'
def get_split():
    train_file_names = []
    val_file_names = []
    test_file_names = []
    val_file_names += list((val_path /'image').glob('*'))
    train_file_names += list((train_path / 'image').glob('*'))
    test_file_names += list((test_path / 'image').glob('*'))
    return train_file_names, val_file_names,test_file_names
def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
    return DataLoader(
        dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
        shuffle=shuffle,
        num_workers=12,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
train_file_names, val_file_names,test_file_names = get_split()
batch_size = 6
class WHULoader():
    def __init__(self):
        # 这儿的batch_size是1，应该是等于外面我设置的batch_size,我这儿直接改了，试验过3090 batch_size=8跑不起来，6可以
        self.train_loader = make_loader(train_file_names, shuffle=False, transform=train_transform(p=1), problem_type='binary',
                               batch_size=batch_size)
        self.valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type='binary',
                               batch_size=batch_size)
        self.test_loader = make_loader(test_file_names, transform=test_transform(p=1), problem_type='binary',
                                        batch_size=batch_size)

    def get_loaders(self):
        return self.train_loader,self.valid_loader,self.test_loader
