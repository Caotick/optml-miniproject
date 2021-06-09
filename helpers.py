import os
from torchvision.datasets import FashionMNIST
from torch.utils.data import ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from datasets import PIMADataset, HousingDataset


def load_data(dataname, k_folds = 10):
    dataset = None
    kfold = KFold(n_splits=k_folds, shuffle=True)

    if dataname.lower() == 'pima':
        dataset = PIMADataset('data/PIMA/diabetes.csv')

    elif dataname.lower() == 'californiahousing':
        dataset = HousingDataset('data/CaliforniaHousing/housing.csv')

    elif dataname.lower() == 'fashionmnist':
        # Prepare Fashion MNIST dataset by concatenating Train and Test, CV handled later
        dataset_train_part = FashionMNIST(os.getcwd() + 'data/FashionMNIST', download=True, transform=transforms.ToTensor(), train=True)
        dataset_test_part = FashionMNIST(os.getcwd() + 'data/FashionMNIST', download=True, transform=transforms.ToTensor(), train=False)
        dataset = ConcatDataset([dataset_train_part, dataset_test_part])

    else:
        raise Exception(f'Dataset {dataname} is not supported')

    return dataset, kfold.split(dataset)