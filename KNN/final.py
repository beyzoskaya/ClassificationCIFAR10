import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-predict", action="store_true")
    parser.add_argument("-dataset_dir", default=None, type=str)
    
    return parser.parse_args()

args = parser_args()

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

cifar10_files = [
    rf'{args.dataset_dir}data_batch_1',
    rf'{args.dataset_dir}data_batch_2',
    rf'{args.dataset_dir}data_batch_3',
    rf'{args.dataset_dir}data_batch_4',
    rf'{args.dataset_dir}data_batch_5'
]

all_features = []
all_labels = []

for batch_file in cifar10_files:
    cifar10_data = unpickle(batch_file)
    all_features.append(cifar10_data[b'data'])
    all_labels.extend(cifar10_data[b'labels'])

features = np.concatenate(all_features, axis=0)
labels = np.array(all_labels)

features_reshaped = features.reshape(len(features), 3, 32, 32).transpose(0, 2, 3, 1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

y_pred = knn_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)