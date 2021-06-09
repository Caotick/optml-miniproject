import torch
import torch.nn.functional as F


def train(model, optimizer, criterion, train_set, train_target,
          test_set, test_target, epochs=100, batch_size=64):
    train_losses, val_losses, accuracies = [], [], []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        val_loss = 0.0
        model.train()  # Set model to train mode

        for inputs, targets in zip(train_set.split(batch_size), train_target.split(batch_size)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss += loss.data.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss * batch_size / len(train_set)  # Necessary to have the mean train loss

        # Evaluation
        model.eval()  # Set model to eval mode
        num_correct = 0
        num_examples = 0
        for inputs, targets in zip(test_set.split(batch_size), test_target.split(batch_size)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            val_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=0), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

        val_loss = val_loss * batch_size / len(test_set)  # Necessary to have the mean val loss

        if epoch % 5 == 0:
            print(
                f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}, accuracy = {num_correct / num_examples:.2f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(num_correct / num_examples)

    return train_losses, val_losses, accuracies