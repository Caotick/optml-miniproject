from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os


def load_data(dataname, seed, k_folds = 10):
    X_train, X_test, y_train, y_test = None, None, None, None

    kfold = KFold(n_splits=k_folds, shuffle=True)

    if dataname.lower() == 'pima':
        df = pd.read_csv('data/PIMA/diabetes.csv')
        X = df.drop('Outcome', axis=1)
        X = (X - X.mean()) / X.std()
        y = df['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        X_train = torch.FloatTensor(X_train.values)
        X_test = torch.FloatTensor(X_test.values)
        y_train = torch.LongTensor(y_train.values)
        y_test = torch.LongTensor(y_test.values)

    elif dataname.lower() == 'californiahousing':
        # TODO
        df = pd.read_csv('data/CaliforniaHousing/housing.csv')
        X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
        X = (X - X.mean()) / X.std()
        y = df['median_house_value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    elif dataname.lower() == 'fashionmnist':
        # Prepare Fashion MNIST dataset by concatenating Train and Test, CV handled later
        dataset_train_part = FashionMNIST(os.getcwd() + 'data/FashionMNIST', download=True, transform=transforms.ToTensor(), train=True)
        dataset_test_part = FashionMNIST(os.getcwd() + 'data/FashionMNIST', download=True, transform=transforms.ToTensor(), train=False)
        dataset = ConcatDataset([dataset_train_part, dataset_test_part])

        return
    else:
        raise Exception(f'Dataset {dataname} is not supported')

    return None