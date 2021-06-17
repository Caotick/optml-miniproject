import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import FashionMNIST
from torch.utils.data import ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from datasets import PIMADataset, HousingDataset
from models import FashionCNN, MLP
from torch.optim import SGD, Adam, Adagrad


def check_path_and_create(path):
    """
    If path does not exists, creates it
    :param path: str, path to check or create
    :return: None
    """
    if not os.path.exists(path):
        os.mkdir(path)


def create_folders_structure():
    """
    Creates the data folder and subfolders
    :param problem: str, Dataset concerned
    :param optimizer: str, Optimizer used
    :return: None
    """
    current = os.getcwd()
    problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
    optimizers = ['SGD', 'Adagrad', 'Adam']

    try:
        check_path_and_create(current + "/data")
        check_path_and_create(current + "/data/test_run")
        check_path_and_create(current + "/data/graph")

        for problem in problems:
            check_path_and_create(current + f"/data/{problem}")
            check_path_and_create(current + f"/data/test_run/{problem}")
            for optimizer in optimizers:
                check_path_and_create(current + f"/data/test_run/{problem}/{optimizer}")

    except:
        print("There was a problem while creating the paths, check the directories")


def save_res(problem, optimizer, train_losses, val_losses, accuracies, nb_fold):
    """
    Save results in appropriate folder structure
    :param problem: str, Dataset concerned
    :param optimizer: str, Optimizer used
    :param train_losses: list, List of train losses
    :param val_losses: list, List of validation losses
    :param accuracies: list, List of accuracies
    :param nb_fold: int, Fold currently iterated
    :return: None
    """
    file_path = f"data/test_run/{problem}/{optimizer}/{nb_fold}"

    to_save = {"train_losses": train_losses, "val_losses": val_losses,
               "accuracies": accuracies}

    with open(file_path + ".pkl", "wb") as f:
        pickle.dump(to_save, f)


def load_data(dataname, k_folds=10):
    """
    Given the name of the dataset, provide the corresponding dataset and fold iterator
    :param dataname: str, Name of dataset to load
    :param k_folds: int, Number of cross validation folds
    :return: dataset, fold iterator
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=404)

    if dataname.lower() == 'pima':
        dataset = PIMADataset('data/PIMA/diabetes.csv')

    elif dataname.lower() == 'californiahousing':
        dataset = HousingDataset('data/CaliforniaHousing/housing.csv')

    elif dataname.lower() == 'fashionmnist':
        # Prepare Fashion MNIST dataset by concatenating Train and Test, CV handled later
        dataset_train_part = FashionMNIST(os.getcwd() + 'data/FashionMNIST', download=True,
                                          transform=transforms.ToTensor(), train=True)
        dataset_test_part = FashionMNIST(os.getcwd() + 'data/FashionMNIST', download=True,
                                         transform=transforms.ToTensor(), train=False)
        dataset = ConcatDataset([dataset_train_part, dataset_test_part])
        dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[
                                                   :len(dataset) // 2])  # select subset of FMNIST for speedup

    else:
        raise Exception(f'Dataset {dataname} is not supported')

    return dataset, kfold.split(dataset)


def get_model(dataname):
    """
    Given the name of the dataset, provide the corresponding deep learning model
    :param dataname: str, Name of dataset to load
    :return: torch.nn.Module, deep learning model
    """
    if dataname.lower() == 'pima':
        return MLP(in_dim=8, out_dim=2, nb_hidden=4, hidden_dim=30)

    elif dataname.lower() == 'californiahousing':
        return MLP(in_dim=8, out_dim=1, nb_hidden=4, hidden_dim=30)

    elif dataname.lower() == 'fashionmnist':
        return FashionCNN()
    else:
        raise Exception(f'Dataset {dataname} is not supported')


def get_criterion(dataname):
    """
    Given the name of the dataset, provide the corresponding loss function
    :param dataname: str, Name of dataset to load
    :return: torch.nn.Module, loss function
    """
    if dataname.lower() == 'pima' or dataname.lower() == 'fashionmnist':
        return nn.CrossEntropyLoss()

    elif dataname.lower() == 'californiahousing':
        return nn.L1Loss()

    else:
        raise Exception(f'Dataset {dataname} is not supported')


def get_optimizer(opt_name, parameters):
    """
    Given the name of the optimizer and paramters to optimize, provide the ready to use optimizer
    :param opt_name: str, Name of the optimizer
    :param parameters: Iterator[Parameter], Model parameters to optimize
    :return: torch.optim.optimizer, ready to use optimizer
    """
    if opt_name.lower() == 'sgd':
        return SGD(parameters, lr=0.01)

    elif opt_name.lower() == 'adam':
        return Adam(parameters, lr=0.001)

    elif opt_name.lower() == 'adagrad':
        return Adagrad(parameters, lr=0.001)
    else:
        raise Exception(f'Optimizer {opt_name} is not supported')
