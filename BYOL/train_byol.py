import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
import random
# test model, a resnet 50

resnet = models.resnet50(pretrained=True)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--train_folder',default=None, type=str,
                       help=' Path to the training image folder used for self-supervised learning')
parser.add_argument('--val_folder', default=None,type=str,
                       help='Path to the test image folder used for self-supervised learning')
args = parser.parse_args()

# constants

BATCH_SIZE = 64
EPOCHS     = 200
LR         = 3e-4
NUM_GPUS   = 4
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
LOCAL_RANK=1
# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        self.paths=self.get_img_info()
        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

    def get_img_info(self):
        data_info = []
        data = open(self.folder, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.split()
            if len(data_line)==2:
                img_pth = data_line[0]
                label = int(data_line[1])
            else:
                img_pth = data_line[0]+' '+data_line[1]
                label = int(data_line[-1])
            data_info.append(img_pth)

        random.shuffle(data_info)
        random.shuffle(data_info)
        return data_info

# main

if __name__ == '__main__':
    ds_train = ImagesDataset(args.train_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    ds_val = ImagesDataset(args.val_folder, IMAGE_SIZE)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 2048,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
