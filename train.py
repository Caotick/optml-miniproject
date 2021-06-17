import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import *


def train(dataset, folds, prob, opt, nb_epochs):
    """
    Trains  and validates a model on the given dataset, using cross validation, and storing results

    :param dataset: torch.utils.data.Dataset, dataset employed
    :param folds: fold iterator
    :param prob: str, name of the dataset used
    :param opt: str, name of the optimizer used
    :param nb_epochs: int, number of epochs
    :return:
    """
    for fold, (train_ids, test_ids) in enumerate(folds):
        print(f'FOLD {fold}')
        print('--------------------------------')
        batch_size = 64

        # Sample elements randomly from a given list of ids, no replacement.
        torch.manual_seed(404)  # Used to preserve folds accross optimizers for a given problem
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=torch.cuda.is_available())
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=test_subsampler, num_workers=4, pin_memory=torch.cuda.is_available())

        model = get_model(prob)
        criterion = get_criterion(prob)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        criterion.to(device)

        optimizer = get_optimizer(opt, parameters=model.parameters())

        train_losses, val_losses, accuracies = train_single_fold(model, optimizer, criterion, trainloader, testloader,
                                                                 device, len(train_ids), len(test_ids),
                                                                 batch_size=batch_size, epochs=nb_epochs)

        save_res(prob, opt, train_losses, val_losses, accuracies, fold)


def train_single_fold(model, optimizer, criterion, trainloader, testloader, device, size_train, size_test, batch_size,
                      epochs=100):
    """
    Trains the model through the provided dataloaders

    :param model: torch.nn.Module, model to train
    :param optimizer: torch.optim.optimizer, ready to use optimizer
    :param criterion: torch.nn.Module, loss function
    :param trainloader: torch.utils.data.DataLoader, loader for training data
    :param testloader: torch.utils.data.DataLoader, loader for testing data
    :param device: str, device on which to allocate tensors
    :param size_train: int, size of the training set
    :param size_test: int, size of the testing set
    :param batch_size: int, batch size
    :param epochs: int, number of epochs
    :return: (list(float), list(float), list(float)), train_losses, val_losses, accuracies
    """
    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        val_loss = 0.0
        model.train()  # Set model to train mode

        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            if (output.size()[1] == 1):
                output = torch.squeeze(output, 1)
            loss = criterion(output, targets)
            train_loss += loss.data.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss * batch_size / size_train  # Necessary to have the mean train loss

        # Evaluation
        model.eval()  # Set model to eval mode
        num_correct = 0
        num_examples = 0

        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            if (output.size()[1] == 1):
                output = torch.squeeze(output, 1)
            loss = criterion(output, targets)
            val_loss += loss.data.item()
            if type(criterion) == nn.CrossEntropyLoss:
                correct = torch.eq(torch.max(F.softmax(output, dim=0), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]

        val_loss = val_loss * batch_size / size_test  # Necessary to have the mean val loss

        if type(criterion) == nn.CrossEntropyLoss:
            print(
                f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}, accuracy = {num_correct / num_examples:.2f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies.append(num_correct / num_examples)
        else:
            print(f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    return train_losses, val_losses, accuracies
