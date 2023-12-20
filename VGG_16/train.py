import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from model import VGG, cfg
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./loss_plot_{model_name}.png")

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_cfg", required=True, default="VGG16", type=str)
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-predict", action="store_true")
    parser.add_argument("-dataset_dir", default=None, type=str)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-optimizer", default="SGD", type=str)
    parser.add_argument("-weight_decay", default=1e-4, type=float)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-max_epochs", default=100, type=int)
    parser.add_argument("-loss_fn", default="ce", type=str)
    parser.add_argument("-early_stopping", default=4, type=int) # TODO implement early stopping
    parser.add_argument("-momentum", default=0.99, type=float)
    parser.add_argument("-save_dir", default=None, type=str, required=True)
    # parser.add_argument("-pretrained_path", default=None, type=str)
    
    return parser.parse_args()

def download_dataset(dataset_dir, transform, download):
    return torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=download, transform=transform), torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=download)


def split_crop_dataset(train_dataset, test_dataset, train_size, val_size, test_size):
    class_indices = [[] for _ in range(10)]
    for i, (image,label) in enumerate(train_dataset):
        class_indices[label].append(i)

    train_size_per_class = train_size
    val_size_per_class = val_size
    test_size_per_class = test_size

    train_indices = []
    val_indices = []
    test_indices = []

    for indices in class_indices:
        # TODO make below random selection to avoid possible inductive bias
        train_indices.extend(indices[:train_size_per_class])
        val_indices.extend(indices[train_size_per_class:train_size_per_class + val_size_per_class])
        test_indices.extend(indices[train_size_per_class + val_size_per_class:train_size_per_class + val_size_per_class + test_size_per_class])
        cifar10_train = Subset(train_dataset, train_indices)
        cifar10_val = Subset(train_dataset, val_indices)
        cifar10_test = Subset(test_dataset, test_indices)
    return cifar10_train, cifar10_val, cifar10_test

def get_dataloaders(batch_size, train, val, test):
    BATCH_SIZE = batch_size

    train_dataloader = DataLoader(train,
                                batch_size=BATCH_SIZE,
                                num_workers=2,
                                shuffle=True,
                                )
    val_dataloader = DataLoader(val,
                                batch_size=BATCH_SIZE,
                                num_workers=2,
                                shuffle=False)

    test_dataloader = DataLoader(test,
                                batch_size=BATCH_SIZE,
                                num_workers=2,
                                shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def train(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device):
    model.train()
    train_loss, train_acc = 0,0
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def validate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device):
    model.eval()
    val_loss, val_acc = 0,0
    with torch.no_grad():
        for batch, (X_val, y_val) in enumerate(dataloader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_val_pred = model(X_val)
            loss_val = loss_fn(y_val_pred, y_val)
            val_loss += loss_val.item()

            y_val_pred_class = torch.argmax(torch.softmax(y_val_pred, dim=1), dim=1)
            val_acc += (y_val_pred_class == y_val).sum().item() / len(y_val_pred)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def test(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> float:

    model.eval()

    test_accuracy = 0.0

    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = dataloader[0]
            print(X_test.shape)
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_test_pred = model(X_test)

            _, predicted = torch.max(y_test_pred, 1)
            test_accuracy += (predicted == y_test).sum().item()

    test_accuracy /= len(dataloader.dataset)
    return test_accuracy

def main():
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    args = parser_args()
    
    # take dataset
    train_dataset, test_dataset = download_dataset(args.dataset_dir, transform=data_transform, download=True)
    train_dataset, val_dataset, test_dataset = split_crop_dataset(train_dataset, test_dataset, train_size=600, val_size=200, test_size=200)    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args.batch_size, train_dataset, val_dataset, test_dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.test and not args.predict:
        try:
            model_cfg = cfg[args.model_cfg]
        except:
            print(f"{args.model_cfg} model is not exist.")
        model = VGG(args.model_cfg)
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise "Given optimizer not supported for now"

        model.to(device)
        max_epochs = args.max_epochs
        loss_fn = torch.nn.CrossEntropyLoss()
        train_losses = []
        val_losses = []
        for epoch in range(max_epochs):
            train_loss, train_acc = train(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
            val_loss, val_acc = validate(model=model,
                                    dataloader=val_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{max_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}.")
            train_losses.append(train_loss)
        plot_losses(train_losses, val_losses, args.model_cfg)
        torch.save(model.state_dict(), f"{args.save_dir}/{args.model_cfg}_{args.max_epochs}.pth")
    elif args.test:
        try:
            model_cfg = cfg[args.model_cfg]
        except:
            print(f"{args.model_cfg} model is not exist.")
        model = VGG(args.model_cfg).to(device)
        try:
            path = f"{args.save_dir}/{args.model_cfg}_{args.max_epochs}.pth"
        except:
            print(f"doesn't exist")
        states = torch.load(path)
        model.load_state_dict(states)
        print(f"Test Accuracy: {test(model, test_dataloader, device)}")
    elif args.predict:
        # randomly picks a test image and visualize it
        try:
            model_cfg = cfg[args.model_cfg]
        except:
            print(f"{args.model_cfg} model is not exist.")
        model = VGG(args.model_cfg).to(device)
        try:
            path = f"{args.save_dir}/{args.model_cfg}_{args.max_epochs}.pth"
        except:
            print(f"doesn't exist")
        states = torch.load(path)
        model.load_state_dict(states)
        rnd_idx = np.random.randint(len(test_dataset))
        img, label = test_dataset[rnd_idx][0], test_dataset[rnd_idx][1]
        img.save("./input_img.png")
        img = torch.Tensor(np.asarray(img)).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        print(img.shape)
        y_test_pred = model(img)
        model.save_feature_maps()
        _, predicted = torch.max(y_test_pred, 1)

        


if __name__ == "__main__":
    main()
    
