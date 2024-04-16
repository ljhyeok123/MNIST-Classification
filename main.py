import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import dataset
from model import LeNet5, CustomMLP

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    trn_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trn_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    trn_loss /= len(trn_loader)
    acc = 100. * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    tst_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tst_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            tst_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    tst_loss /= len(tst_loader)
    acc = 100. * correct / total

    return tst_loss, acc

def plot_curves(train_losses, train_accuracies, test_losses, test_accuracies):
    """ Plot loss and accuracy curves for training and test datasets """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.show()

def main():
    """ Main function """

    # Hyperparameters
    lr = 0.01
    momentum = 0.9
    num_epochs = 10
    batch_size = 64

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader for training and testing
    train_dataset = dataset.MNIST(root='.', train=True)
    test_dataset = dataset.MNIST(root='.', train=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, and optimizer
    lenet_model = LeNet5().to(device)
    mlp_model = CustomMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    lenet_optimizer = optim.SGD(lenet_model.parameters(), lr=lr, momentum=momentum)
    mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=lr, momentum=momentum)

    # Training and testing
    lenet_train_losses, lenet_train_accuracies = [], []
    lenet_test_losses, lenet_test_accuracies = [], []
    mlp_train_losses, mlp_train_accuracies = [], []
    mlp_test_losses, mlp_test_accuracies = [], []

    for epoch in range(num_epochs):
        lenet_train_loss, lenet_train_acc = train(lenet_model, train_loader, device, criterion, lenet_optimizer)
        lenet_test_loss, lenet_test_acc = test(lenet_model, test_loader, device, criterion)
        lenet_train_losses.append(lenet_train_loss)
        lenet_train_accuracies.append(lenet_train_acc)
        lenet_test_losses.append(lenet_test_loss)
        lenet_test_accuracies.append(lenet_test_acc)

        mlp_train_loss, mlp_train_acc = train(mlp_model, train_loader, device, criterion, mlp_optimizer)
        mlp_test_loss, mlp_test_acc = test(mlp_model, test_loader, device, criterion)
        mlp_train_losses.append(mlp_train_loss)
        mlp_train_accuracies.append(mlp_train_acc)
        mlp_test_losses.append(mlp_test_loss)
        mlp_test_accuracies.append(mlp_test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'LeNet-5: Train Loss: {lenet_train_loss:.4f}, Test Loss: {lenet_test_loss:.4f}, '
              f'Train Acc: {lenet_train_acc:.2f}%, Test Acc: {lenet_test_acc:.2f}%')
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Custom MLP: Train Loss: {mlp_train_loss:.4f}, Test Loss: {mlp_test_loss:.4f}, '
              f'Train Acc: {mlp_train_acc:.2f}%, Test Acc: {mlp_test_acc:.2f}%')

    # Plotting loss and accuracy curves
    plot_curves(lenet_train_losses, lenet_train_accuracies, lenet_test_losses, lenet_test_accuracies)
    plot_curves(mlp_train_losses, mlp_train_accuracies, mlp_test_losses, mlp_test_accuracies)

if __name__ == '__main__':
    main()
