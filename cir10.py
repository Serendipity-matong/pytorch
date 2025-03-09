import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import py7zr
from io import BytesIO ## this too
import os

import torch


# 检查 GPU 是否可用
import tensorflow as tf
print("TensorFlow 支持 CUDA:", tf.test.is_built_with_cuda())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Conv2d(3, 64, 3,stride=1, padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.1),nn.MaxPool2d(2,2),
    nn.Conv2d(64,128,3,stride=1, padding=1),nn.BatchNorm2d(128),nn.LeakyReLU(0.1),nn.MaxPool2d(2,2),
    nn.Conv2d(128,256,3,stride=1, padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.1),nn.MaxPool2d(2,2),
    nn.Conv2d(256,512,3,stride=1, padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.LeakyReLU(0.1),
    nn.Linear(512,10)
)

optim = torch.optim.Adam(model.parameters(), lr=0.0003,weight_decay=0.0005)

scheduler = StepLR(optim, step_size=10, gamma=0.5)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,0.2435,0.2616)),
    transforms.RandomErasing(p=0.5,scale=(0.02,0.1),value=1.0,inplace=False),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
])

batch_size = 64

cifar10 = torchvision.datasets.CIFAR10(root=r'D:\KaggleNotebook\cir10', train=True, download=False, transform=transform_train)
cifar10_test = torchvision.datasets.CIFAR10(root=r'D:\KaggleNotebook\cir10', train=False, download=False, transform=transform_test)

batch_size = 64
n = batch_size
dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=True,num_workers=2)
dataloader_test = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False,num_workers=2)

def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
model.apply(init_weights_xavier)

i=0
losses=[]
steps=[]
losses_t=[]

for epoch in range(32):
    for batch in dataloader:
        i +=1
        x = batch[0].to(device)
        y = batch[1].to(device)
        model = model.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad()

        loss.backward()
        optim.step()

        losses.append(loss.item())
        losses_t.append(loss.item())
        steps.append(i)

        if(i%1000==0):
            print(loss)
    scheduler.step()

plt.plot(steps, losses)

total_correct = 0
total_predictions = 0

# Loop over training dataset
for x_batch, y_batch in dataloader:
    logits = model(x_batch.to("cuda")) # Forward pass on the mini-batch
    loss = F.cross_entropy(logits.cpu(), y_batch) # Compute loss

    # Calculate predictions for the batch
    pred_labels = torch.max(logits, dim=1).indices

    # Update total correct predictions and total predictions
    total_correct += (y_batch == pred_labels.cpu()).sum().item()
    total_predictions += y_batch.size(0)

# Calculate overall accuracy
overall_accuracy = total_correct / total_predictions
print(f"Training Accuracy: {overall_accuracy}")

total_correct = 0
total_predictions = 0

# Loop over testing dataset
for x_batch, y_batch in dataloader_test:
    logits = model(x_batch.to("cuda")) # Forward pass on the mini-batch
    loss = F.cross_entropy(logits.cpu(), y_batch) # Compute loss

    # Calculate predictions for the batch
    pred_labels = torch.max(logits, dim=1).indices

    # Update total correct predictions and total predictions
    total_correct += (y_batch == pred_labels.cpu()).sum().item()
    total_predictions += y_batch.size(0)

# Calculate overall accuracy and print it out.
overall_accuracy = total_correct / total_predictions
#DON'T FORGET TO PRINT OUT YOUR TESTING ACCURACY
print(f"Testing Accuracy: {overall_accuracy}")


# Create submission file with predictions for test images from .7z archive
test_filenames = []
test_images = []

with py7zr.SevenZipFile('/kaggle/input/cifar-10/test.7z', mode='r') as z:
    for name, file in z.readall().items():
        if name.endswith('.png'):
            img = Image.open(BytesIO(file.read()))
            test_images.append(transform_test(img))
            test_filenames.append(name)

test_images = torch.stack(test_images)
test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

# Prediction and CSV creation
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

result = []
with torch.no_grad():
    model.eval()
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        result.extend(predicted.cpu().numpy())

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': [os.path.basename(f).replace('.png', '') for f in test_filenames],  # Remove .png from filenames
    'label': [classes[label] for label in result]
})

# Save submission file
submission_df.to_csv(r'D:\KaggleNotebook\cir10\submission.csv', index=False)

print("Submission file created successfully!")
