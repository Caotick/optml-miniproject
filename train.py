import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import *

def train(dataset, folds, prob, opt):
    for fold, (train_ids, test_ids) in enumerate(folds):
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, sampler=test_subsampler)

        model = get_model(prob)
        optimizer = get_optimizer(opt, parameters=model.parameters())
        criterion = get_criterion(prob)

        train_losses, val_losses, accuracies = train_single_fold(model, optimizer, criterion, trainloader, testloader, epochs = 10)

        save_res(prob, opt, train_losses, val_losses, accuracies, fold)


def train_single_fold(model, optimizer, criterion, trainloader, testloader, epochs=2):
    train_losses, val_losses, accuracies = [], [], []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    criterion.to(device)

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
            loss = criterion(output, targets)
            train_loss += loss.data.item()
            loss.backward()
            optimizer.step()

        # TODO
        # train_loss = train_loss * batch_size / len(train_set)  # Necessary to have the mean train loss

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
            loss = criterion(output, targets)
            val_loss += loss.data.item()
            if type(criterion) == nn.CrossEntropyLoss:
                correct = torch.eq(torch.max(F.softmax(output, dim=0), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]

        # TODO
        #val_loss = val_loss * batch_size / len(test_set)  # Necessary to have the mean val loss

        if type(criterion) == nn.CrossEntropyLoss:
            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}, accuracy = {num_correct / num_examples:.2f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies.append(num_correct / num_examples)
        else:
            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    return train_losses, val_losses, accuracies