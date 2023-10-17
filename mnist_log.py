# -*- coding: utf-8 -*-

import pickle
import gzip
import torch
import torch.nn.functional as F
import math
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib import pyplot as plt


with gzip.open(("./mnist/mnist.pkl.gz"), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")


x_train, y_train, x_valid, y_valid, x_test, y_test = map(
    torch.as_tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
)

x_train = torch.cat([x_train, x_valid]) #50000+10000=60000
y_train = torch.cat([y_train, y_valid])

n,c = x_train.shape

loss_func = F.cross_entropy

def accuracy(out, ytest):
    preds = torch.argmax(out, dim=1)
    return (preds == ytest).float().mean()

#define logistic regression
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__() #inheritance
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784)) #weight
        self.bias = nn.Parameter(torch.zeros(10)) #bias
        
    def forward(self, x_test): #forward propagation
        return x_test @ self.weights + self.bias
    
model = Mnist_Logistic()

epochs = 2000
lr = 1

loss_list = []
acc_list = []

# update parameters
def fit():
    for epoch in range(epochs):

        pred = model(x_train)
        loss = loss_func(pred, y_train)
        
        loss_list.append(loss.item())
            
        loss.backward() #back propagation
            
        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad * lr
                    
            model.zero_grad()
            
        acc_list.append(accuracy(model(x_test), y_test).item())
                
fit()

Recall = recall_score(y_test,torch.argmax(model(x_test), dim=1), average='macro')
Precision = precision_score(y_test,torch.argmax(model(x_test), dim=1), average='macro')
F1 = f1_score(y_test,torch.argmax(model(x_test), dim=1), average='macro')
    
print("Accuracy:", accuracy(model(x_test), y_test).item())
print("Recall:",Recall)
print("Precision:",Precision)
print("F1:",F1)

plt.plot(loss_list, label='Loss')
plt.plot(acc_list, label='Accuracy')
plt.xlabel('Iteration')
plt.title('Loss and Accuracy Curve of Logistic Regression')
plt.legend()
plt.show()

