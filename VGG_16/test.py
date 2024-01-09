from sklearn.metrics import precision_score, f1_score, recall_score, precision_recall_fscore_support, accuracy_score, precision_recall_curve, average_precision_score
import torch
from model import VGG, cfg
from dataset import CIFAR10Dataset, Oxford102FlowersDataset, STL10Dataset
from train import get_dataloaders
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi
import torch.nn.functional as F

def vis_convs(filters):
    filters = filters - filters.min()
    filters = filters / filters.max()

    n_filters = filters.shape[0]  # Number of filters (64 in your case)

    # Create a grid for plotting
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter numbers
        if i < n_filters:
            # Get the filter
            filter = filters[i, :, :, :]

            # If your filters are for grayscale, remove the next line
            filter = filter.permute(1, 2, 0)  # Change to (Height, Width, Channels) for plotting
            
            # Plot the filter
            ax.imshow(filter.detach().cpu().numpy(), interpolation='nearest')

            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig("./figs/convs.png")


def save_df_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(12, 2)) # Set dimensions as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')

    plt.savefig(filename, dpi=300)
    plt.close()

def create_metrics_table(model_name, accuracy, precision, recall, f1, filename):
    data = {
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    df = pd.DataFrame.from_dict(data)
    save_df_as_image(df, filename)

def plot_precision_recall_curve(truth, predictions_proba, num_classes, filename):
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(truth[:, i], predictions_proba[:, i])
        plt.plot(recall, precision, lw=2, label=f'class {i}')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision vs. Recall curve")
    plt.savefig(filename, format='png')
    plt.close()

def plot_precision_recall_curve_mean(truth, predictions_proba, num_classes, filename):
    all_precision = []
    all_recall = [] 
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(truth[:, i], predictions_proba[:, i])
        all_precision.append(precision)
        all_recall.append(recall)

    # Calculate the average precision and recall across all classes
    average_precision = np.mean([np.mean(precision) for precision in all_precision])
    average_recall = np.mean([np.mean(recall) for recall in all_recall])

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(average_recall, average_precision, lw=2, label='Average Precision-Recall')

    plt.xlabel("Average Recall")
    plt.ylabel("Average Precision")
    plt.legend(loc="best")
    plt.title("Average Precision vs. Recall Curve")
    plt.savefig(filename, format='png')
    plt.close()


def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()

    predictions = []
    truth = []
    predictions_proba = []
    one_hot_truth = []
    
    with torch.no_grad():
        for batch, (X_val, y_val) in enumerate(dataloader):
            print(X_val.shape)
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_val_pred = model(X_val)
            y_val_pred_class = torch.argmax(torch.softmax(y_val_pred, dim=1), dim=1)
            softmax_probs = F.softmax(y_val_pred, dim=1)
            predictions_proba.append(softmax_probs.detach().cpu().numpy())
            one_hot = F.one_hot(y_val.long(), num_classes=10)
            one_hot_truth.append(one_hot.detach().cpu().numpy())
            predictions.append(y_val_pred_class.detach().cpu().numpy())
            truth.append(y_val.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    truth = np.concatenate(truth)
    predictions_proba = np.concatenate(predictions_proba, axis=0)
    one_hot_truth = np.concatenate(one_hot_truth, axis=0)
    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions, average='weighted')
    f1 = f1_score(truth, predictions, average='weighted')
    recall = recall_score(truth, predictions, average='weighted')

    metrics_table_filename = f"./figs/{model_name}_metrics_table.png"
    create_metrics_table(model_name, accuracy, precision, recall, f1, metrics_table_filename)

    pr_curve_filename = f"./figs/{model_name}_precision_recall_curve.png"
    pr_curve_filename_avg = f"./figs/{model_name}_precision_recall_curve_average.png"
    plot_precision_recall_curve(one_hot_truth, predictions_proba, num_classes=10, filename=pr_curve_filename)
    # plot_precision_recall_curve_mean(one_hot_truth, predictions_proba, num_classes=10, filename=pr_curve_filename_avg)

from torchvision.transforms import transforms
if __name__ == "__main__":
    input_size = (32, 32)
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(input_size),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    model_name = "VGG26_cifar10"
    model = VGG(vgg_name="VGG26", init_size=input_size, num_classes=10).to("cuda")
    train_dataset =  CIFAR10Dataset(root_dir="/home/emir/Desktop/dev/datasets/cs454_datasets", train=True, transform=None)
    test_dataset =  CIFAR10Dataset(root_dir="/home/emir/Desktop/dev/datasets/cs454_datasets", train=False, transform=test_transform)
    # train_dataset =  STL10Dataset(data_dir="/home/emir/Desktop/dev/datasets/cs454_datasets", train=True, transform=None)
    # test_dataset = STL10Dataset(data_dir="/home/emir/Desktop/dev/datasets/cs454_datasets", train=False, transform=test_transform)
    # train_dataset =  Oxford102FlowersDataset(root_dir="/home/emir/Desktop/dev/datasets/cs454_datasets", train=True, transform=None)
    # test_dataset = Oxford102FlowersDataset(root_dir="/home/emir/Desktop/dev/datasets/cs454_datasets", train=False, transform=test_transform)
    train_dataset, _, test_loader = get_dataloaders(batch_size=32, train=train_dataset, val=None, test=test_dataset)
    print(len(test_loader))
    # print(next(iter(test_loader)))
    model.load_state_dict(torch.load("/home/emir/Desktop/dev/ClassificationCIFAR10/VGG_16/VGG26_30_cifar10.pth"))
    # print(model.features[0].weight.data)
    # vis_convs(model.features[0].weight.data)
    test(model, test_loader, "cuda")
    