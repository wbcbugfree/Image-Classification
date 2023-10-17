# -*- coding: utf-8 -*-

import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose,ToTensor,Normalize

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt

transform_fn = Compose([ToTensor(), Normalize(mean = (0.1307,),std = (0.3081,))])# mean and standard deviation of MNIST

train_data = torchvision.datasets.MNIST(root="./pymnist",train=True,transform=transform_fn,download=False)
test_data = torchvision.datasets.MNIST(root="./pymnist",train=False,transform=transform_fn,download=False)

train_loader = DataLoader(train_data,batch_size=128,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10000,shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__() #inheritance
        self.con1 = torch.nn.Conv2d(1,10,kernel_size=5) #convolutional
        self.con2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2) #polling
        self.fc = torch.nn.Linear(320,10) #fully connected
    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.con1(x))) #activation function
        x = F.relu(self.pooling(self.con2(x)))
        x = x.view(batch_size,-1) #output
        x = self.fc(x)
        return x
        
model = Net()

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.5)

loss_list = []
acc_list = []

def train(epoch):
    for i,(inputs,targets) in enumerate(train_loader,0):
        optimizer.zero_grad()
        outputs = model(inputs) #get predict
        loss = loss_func(outputs,targets)
        loss.backward() #back propagation
        optimizer.step() #update gradient
        loss_list.append(loss.item())
        pred = outputs.max(dim = -1)[-1] #get labels of predict
        cur_acc = pred.eq(targets).float().mean()
        acc_list.append(cur_acc.item())


def test():
    with torch.no_grad(): #do not calculate gradient while testing
        for (inputs,targets) in test_loader:
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data,dim=1)
            
    Accuracy = accuracy_score(targets,predicted)
    Recall = recall_score(targets,predicted, average='macro')
    Precision = precision_score(targets,predicted, average='macro')
    F1 = f1_score(targets,predicted, average='macro')
    
    print("Accuracy:",Accuracy)
    print("Recall:",Recall)
    print("Precision:",Precision)
    print("F1:",F1)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
    test()

plt.plot(loss_list, label='Loss')
plt.plot(acc_list, label='Accuracy')
plt.xlabel('Iteration')
plt.title('Loss and Accuracy Curve of CNN')
plt.legend()
plt.show()