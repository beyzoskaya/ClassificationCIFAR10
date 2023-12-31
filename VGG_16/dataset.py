from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR10
import torch
import os
from scipy.io import loadmat
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import random

DATASET_CFG = ["cifar10", "102flowers", "chest_xray", "stl10"]

class STL10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, test_split_ratio=0.2, random_seed=42):
        """
        Initialize the dataset.
        :param data_dir: Directory where the dataset is stored.
        :param train: Boolean, whether to load the training set or the test set.
        :param transform: Optional transform to be applied on a sample.
        :param test_split_ratio: Ratio of the dataset to be used as the test set.
        :param random_seed: Seed for random number generator for reproducibility.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load all data
        all_data = self._read_binary_file(os.path.join(data_dir, 'STL10/img/'))

        # Split the data into train and test sets
        random.seed(random_seed)
        random.shuffle(all_data)
        split_index = int(len(all_data) * (1 - test_split_ratio))
        self.data = all_data[:split_index] if train else all_data[split_index:]

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        image_path, label = self.data[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def _read_binary_file(self, file_path):
        """
        Read a binary file containing either images or labels.
        """
        data = []
        for label in os.listdir(file_path):
            img_dir = os.path.join(file_path, label)
            for img in os.listdir(img_dir):
                data.append((os.path.join(img_dir, img), int(label)-1))
        return data


class ChestXrayDataset(Dataset):
    def __init__(self, dataset_dir, dataset_name, train=True, transform=None):
        """
        Initialize the dataset.
        :param dataset_dir: Directory where the dataset is stored.
        :param dataset_name: Name of the dataset.
        :param train: Boolean, whether to load the training set or the test set.
        :param transform: Optional transform to be applied on a sample.
        """
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.transform = transform

        # Load the dataset from disk
        full_dataset = load_from_disk(os.path.join(dataset_dir, dataset_name))
        print(full_dataset)
        # Select the appropriate split
        self.dataset =  ConcatDataset([full_dataset['train'], full_dataset['validation']]) if train else full_dataset['test']

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming each data point is a dictionary with 'image' and 'label' keys
        data = self.dataset[idx]
        # print(data['image'])
        # image = Image.open(data['image']).convert('RGB')
        image = data['image']
        label = data['labels']
        # print(image.size)
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Oxford102FlowersDataset(Dataset):
    #TODO possible label and image match might need fix ?
    def __init__(self, root_dir, train=True, transform=None, test_size=0.2, random_state=42):
        """
        Initialize the dataset.
        :param root_dir: Directory with the Oxford 102 Flowers JPEG images.
        :param mat_file: Path to the .mat file containing labels.
        :param train: Boolean, whether to load the training set or the test set.
        :param transform: Optional transform to be applied on a sample.
        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Random state for reproducible train-test splitting.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_labels = loadmat(os.path.join(root_dir, "102flowers/imagelabels_augmented.mat"))['labels'][0] - 1  # Adjust label indices to start from 0
        # print(self.image_labels)
        self.image_files = sorted([f for f in os.listdir(os.path.join(root_dir, "102flowers/jpg/"))])
        # print(self.image_files)
        # Splitting the dataset into train and test sets
        train_indices, test_indices = train_test_split(range(len(self.image_files)), test_size=test_size, random_state=random_state, stratify=self.image_labels)
        # print(train_indices)
        # Selecting the appropriate indices for train/test
        self.indices = train_indices if train else test_indices

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        true_idx = self.indices[idx]
        img_name = os.path.join(self.root_dir, "102flowers/jpg/", self.image_files[true_idx])
        # print(img_name)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.image_labels[true_idx]
        # print(label)
        return image, label

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Initialize the dataset.
        :param root_dir: Directory with the CIFAR-10 dataset.
        :param train: Boolean, whether to load the training set or the test set.
        :param transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = CIFAR10(root=self.root_dir, train=train, download=False, transform=self.transform)

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.data[idx]

        return img, label



def load_dataset_cfg(dataset_dir, dataset_name, train_transform, test_transform):
    if dataset_name in DATASET_CFG:
        if dataset_name == DATASET_CFG[0]:
            train_dataset = CIFAR10Dataset(root_dir=dataset_dir, train=True, transform=train_transform)
            test_dataset = CIFAR10Dataset(root_dir=dataset_dir, train=False, transform=test_transform)
            return train_dataset, test_dataset
        elif dataset_name == DATASET_CFG[1]:
            train_dataset = Oxford102FlowersDataset(root_dir=dataset_dir, train=True, transform=train_transform)
            test_dataset = Oxford102FlowersDataset(root_dir=dataset_dir, train=False, transform=test_transform)
            print(len(train_dataset))
            print(len(test_dataset))
            return train_dataset, test_dataset
        elif dataset_name == DATASET_CFG[2]:
            train_dataset = ChestXrayDataset(dataset_dir=dataset_dir, dataset_name=dataset_name, train=True, transform=train_transform)
            test_dataset = ChestXrayDataset(dataset_dir=dataset_dir, dataset_name=dataset_name, train=False, transform=test_transform)
            return train_dataset, test_dataset
        elif dataset_name == DATASET_CFG[3]:
            train_dataset = STL10Dataset(data_dir=dataset_dir, train=True, transform=train_transform)
            test_dataset = STL10Dataset(data_dir=dataset_dir, train=False, transform=test_transform)
            # print(len(train_dataset))
            # print(len(test_dataset))
            return train_dataset, test_dataset
    else:
        assert "dataset is not in CFG"
        

def augment_image(image_path, save_dir, transform, num_augmented_images=10):
    """
    Apply transformations to an image and save the augmented images.
    """
    image = Image.open(image_path)
    for i in range(num_augmented_images):
        augmented_image = transform(image)
        augmented_image_path = os.path.join(save_dir, f"augmented_{i}_{os.path.basename(image_path)}")
        augmented_image.save(augmented_image_path)

from torchvision.transforms import transforms

def augment_dataset_whole(dataset_dir):
    # Define your transformations here
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(0.75, 1.33))
    ])

    for image_dir in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, image_dir)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                if os.path.isfile(image_path):
                    augment_image(image_path, label_dir, transform)
       
       
import scipy             
def augment_dataset_flowers(dataset_dir, num_augmentations=3):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomRotation(30),
        # Add more transformations as needed
    ])

    image_dir = os.path.join(dataset_dir, "102flowers/jpg")
    label_data = loadmat(os.path.join(dataset_dir, "102flowers/imagelabels.mat"))['labels'][0]
    # print(len(label_data))
    for i, file_name in enumerate(os.listdir(image_dir)):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(image_dir, file_name)
            image = Image.open(image_path)
            label = label_data[i]
            for augment_idx in range(num_augmentations):
                augmented_image = transform(image)
                augmented_file_name = f"z{file_name.split('.')[0]}_aug_{augment_idx}.jpg"
                augmented_image_path = os.path.join(image_dir, augmented_file_name)
                augmented_image.save(augmented_image_path)
                # Append the label of the augmented image to the label data
                label_data = np.append(label_data, label)

    # # Save the updated label data
    new_label_file = os.path.join(dataset_dir, "102flowers/imagelabels_augmented.mat")
    print(len(label_data))
    scipy.io.savemat(new_label_file, {'labels': label_data})


if __name__ == "__main__":
    dataset_dir = "/home/emir/Desktop/dev/datasets/cs454_datasets/"
    # print(len(os.listdir(dataset_dir+"jpg")))
    # print(loadmat(dataset_dir+"setid.mat"))
    # augment_dataset_whole(dataset_dir)
    # augment_dataset_flowers(dataset_dir)
    # print(len(os.listdir(os.path.join(dataset_dir, "102flowers/jpg/"))))
    # label_data = loadmat(os.path.join(dataset_dir, "102flowers/imagelabels_augmented.mat"))['labels'][0]
    # print(len(label_data))
    
        