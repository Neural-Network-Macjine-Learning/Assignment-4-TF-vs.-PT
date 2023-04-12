# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:14:28 2023

@author: zakis
"""
import time
# get the start time
st = time.process_time()

import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader



os.chdir('E:/Machine Learning/Assign4/Data_as_4')

df=pd.read_csv("Data_as4.csv")
features= ['WellDepth','Elevation','Rainfall','Tmin','claytotal_l', 'awc_l','ph_r']

X = df[features] # DATAFRAME OF INPUT FEATURES

y = df['Fluoride_avg'] # Label (assuming 2ppm as the threshold)

X=np.array(X)
y=np.array(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


#Creating Tensor Dataset
ds = torch.utils.data.TensorDataset(X, y)

train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

#Splitting into batch loader

train_dl = DataLoader(train_ds, batch_size = 32, shuffle=True) #torch.utils.data
test_dl = DataLoader(test_ds, batch_size =32)




for X, y in test_dl:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Creating models
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

model = nn.Sequential(
    nn.Linear(7, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

#Optimizing the Model Parameters
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


"""
Training function
"""
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""

"""
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)
print("Done!")


##Saving models

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")




model.eval()

# get the end time
et = time.process_time()
elapsed_time = (et - st)*1000
print('CPU Execution time:', elapsed_time, 'milliseconds')
