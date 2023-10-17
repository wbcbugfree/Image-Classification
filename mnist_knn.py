#!/usr/bin/env python
# coding: utf-8

import operator
import numpy as np
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

train_dataset = dsets.MNIST('./pymnist', train=True, transform=None, download=False)  
test_dataset = dsets.MNIST('./pymnist', train=False, transform=None, download=False)  

train_loader = DataLoader(dataset=train_dataset, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True)

# define KNN
def KNN_classify(k, train_data, train_label, test_data):
    num_test = test_data.shape[0]
    label_list = []
    for i in range(num_test):
        distances = np.sqrt(np.sum(((train_data - np.tile(test_data[i], (train_data.shape[0], 1))) ** 2), axis=1))
        nearest_k = np.argsort(distances)
        top_k = nearest_k[:k]
        class_count = {}
        for j in top_k:
            class_count[train_label[j]] = class_count.get(train_label[j], 0) + 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        label_list.append(sorted_class_count[0][0])
    return np.array(label_list)


# calculate mean
def getXmean(data):
    data = np.reshape(data, (data.shape[0], -1))
    mean_image = np.mean(data, axis=0)
    return mean_image

# normalized data
def centralized(data, mean_image):
    data = data.reshape((data.shape[0], -1))
    data = data.astype(np.float64)
    data -= mean_image
    return data


if __name__ == '__main__':
    # process train data
    train_data = train_loader.dataset.data.numpy()
    mean_image = getXmean(train_data)
    train_data = centralized(train_data, mean_image)
    train_label = train_loader.dataset.targets.numpy()

    # process test data
    test_data = test_loader.dataset.data.numpy()
    test_data = centralized(test_data, mean_image)
    test_label = test_loader.dataset.targets.numpy()

    # train
    test_label_pred = KNN_classify(3, train_data, train_label, test_data)

    # test
    Accuracy = accuracy_score(test_label,test_label_pred)
    Recall = recall_score(test_label,test_label_pred, average='macro')
    Precision = precision_score(test_label,test_label_pred, average='macro')
    F1 = f1_score(test_label,test_label_pred, average='macro')
    
    print("Accuracy:",Accuracy)
    print("Recall:",Recall)
    print("Precision:",Precision)
    print("F1:",F1)