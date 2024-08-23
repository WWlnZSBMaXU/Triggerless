from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
import numpy as np

feature_sizes = []

class CINIC10(VisionDataset):
    def __init__(self, root="./dataset", train=True, transform=None, **kwargs):
        super(CINIC10, self).__init__(root, transform=transform)
        self.train = train
        root = os.path.join(root, "CINIC10")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the CINIC10 dataset to {} first! Download: https://github.com/BayesWatch/cinic-10".format(root))

        if train:
            fold = '/train'
        else:
            fold = '/test'
        image = torchvision.datasets.ImageFolder(root=root + fold, transform=transform)
        self.data = image.imgs
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class Criteo(Dataset):
    '''
    To load Criteo dataset.
    '''
    def __init__(self, root="./dataset", train=True, **kwargs):
        self.train = train
        root = os.path.join(root, "criteo")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the Criteo dataset to {} first! Download: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz".format(root))
        
        # sample data
        file_out = "train_sampled.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', nrows=70000, index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                if idx > 0 and idx <= 13:
                    self.csv_data[col] = self.csv_data[col].fillna(0,)
                elif idx >= 14:
                    self.csv_data[col] = self.csv_data[col].fillna('-1',)

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset sampling completed.")
        
        # process data
        file_out = "train_processed.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train_sampled.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                le = LabelEncoder()
                le.fit(self.csv_data[col])
                self.csv_data[col] = le.transform(self.csv_data[col])

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset processing completed.")

        self.csv_data = pd.read_csv(outpath, sep='\t', index_col=None)
        if train:
            global feature_sizes
            feature_sizes.clear()
            cols = self.csv_data.columns.values
            for col in cols:
                feature_sizes.append(len(self.csv_data[col].value_counts()))
            feature_sizes.pop(0)  # do not contain label

        self.train_data, self.test_data = train_test_split(self.csv_data, test_size=1/7, random_state=42)
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, idx):
        if self.train:
            x = self.train_data.iloc[idx].values
        else:
            x = self.test_data.iloc[idx].values
        x = np.array(x, dtype=np.float32)
        return x[1:], int(x[0])


datasets_choices = [
    "mnist",
    "fashionmnist",
    "cifar10",
    "cifar100",
    "criteo",
    "cinic10"
]

datasets_name = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "criteo": "Criteo",
    "cinic10": "CINIC10"
}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "criteo": Criteo,
    "cinic10": CINIC10
}

datasets_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "criteo": 2,
    "cinic10": 10
}

transforms_default = {
    "mnist": transforms.Compose([transforms.ToTensor()]),
    "fashionmnist": transforms.Compose([transforms.ToTensor()]),
    "cifar10": transforms.Compose([transforms.ToTensor()]),
    "cifar100": transforms.Compose([transforms.ToTensor()]),
    "criteo": None,
    "cinic10": transforms.Compose([transforms.ToTensor()])
}

transforms_augment = {
    "mnist": transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "fashionmnist": transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "criteo": None,
    "cinic10": transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
    ])
}