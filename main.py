import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import py7zr
from io import BytesIO
import os
import torch.multiprocessing as mp

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
    nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(512, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1),
    nn.Linear(512, 10)
)

# 定义优化器和学习率调度器
optim = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0005)
scheduler = StepLR(optim, step_size=10, gamma=0.5)

# 定义数据增强和标准化
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=1.0, inplace=False),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 加载 CIFAR-10 数据集
cifar10 = torchvision.datasets.CIFAR10(root=r'D:\KaggleNotebook\cir10', train=True, download=False, transform=transform_train)
cifar10_test = torchvision.datasets.CIFAR10(root=r'D:\KaggleNotebook\cir10', train=False, download=False, transform=transform_test)

# 创建 DataLoader
batch_size = 64
dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=True, num_workers=2)
dataloader_test = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)

# 初始化权重
def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights_xavier)

# 训练函数
def train_model():
    model.to(device)
    losses = []
    steps = []

    for epoch in range(32):
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            x = batch[0].to(device)
            y = batch[1].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            steps.append(len(steps) + 1)

        scheduler.step()

    plt.plot(steps, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

# 测试函数
def test_model(dataloader, dataset_name):
    model.eval()
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc=f"Testing {dataset_name}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            pred_labels = torch.max(logits, dim=1).indices

            total_correct += (y_batch == pred_labels).sum().item()
            total_predictions += y_batch.size(0)

    accuracy = total_correct / total_predictions
    print(f"{dataset_name} Accuracy: {accuracy}")

# 主函数
def main():
    train_model()
    test_model(dataloader, "Training")
    test_model(dataloader_test, "Testing")

    # 创建测试集预测
    test_filenames = []
    test_images = []

    with py7zr.SevenZipFile(r'D:\KaggleNotebook\cir10\test.7z', mode='r') as z:
        for name, file in z.readall().items():
            if name.endswith('.png'):
                img = Image.open(BytesIO(file.read()))
                test_images.append(transform_test(img))
                test_filenames.append(name)

    test_images = torch.stack(test_images)
    test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

    # 预测
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    result = []

    with torch.no_grad():
        model.eval()                    7
        for inputs in tqdm(test_loader, desc="Predicting Test Images"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            result.extend(predicted.cpu().numpy())

    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': [os.path.basename(f).replace('.png', '') for f in test_filenames],
        'label': [classes[label] for label in result]
    })

    submission_df.to_csv(r'D:\KaggleNotebook\cir10\submission.csv', index=False)
    print("Submission file created successfully!")

# 保护主模块的入口点
if __name__ == '__main__':
    mp.freeze_support()  # 在 Windows 上支持多进程
    main()