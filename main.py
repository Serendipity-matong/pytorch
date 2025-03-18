import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
from tqdm import tqdm

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
df_train = pd.read_csv('/root/autodl-tmp/train.csv')
df_test = pd.read_csv('/root/autodl-tmp/test.csv')

# 计算文本长度
df_train["length"] = df_train["text"].apply(lambda x: len(x))
df_test["length"] = df_test["text"].apply(lambda x: len(x))

print("Train length:")
print(df_train["length"].describe())

print("Test length:")
print(df_test["length"].describe())

# 定义超参数
BATCH_SIZE = 2
NUM_TRAINING_EXAMPLES = df_train.shape[0]
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
EPOCHS = 2

# 划分数据集
X = df_train["text"]
y = df_train["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
X_test = df_test["text"]

# 自定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=60):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# 加载预训练的分词器和模型
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

# 创建数据集和数据加载器
train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset = NewsDataset(X_val, y_val, tokenizer)
test_dataset = NewsDataset(X_test, None, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1} Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Training Loss: {total_loss / len(train_dataloader)}')

    # 验证模型
    model.eval()
    val_loss = 0
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1} Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            val_predictions.extend(predictions.cpu().tolist())
            val_true_labels.extend(labels.cpu().tolist())

    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_dataloader)}')

# 定义混淆矩阵显示函数
def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Not Disaster", "Disaster"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_score = tp / (tp + ((fn + fp) / 2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))

# 训练集预测
train_predictions = []
train_true_labels = []
model.eval()
with torch.no_grad():
    for batch in tqdm(train_dataloader, desc='Training Set Prediction'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        train_predictions.extend(predictions.cpu().tolist())
        train_true_labels.extend(labels.cpu().tolist())

displayConfusionMatrix(train_true_labels, train_predictions, "Training")

# 验证集预测
displayConfusionMatrix(val_true_labels, val_predictions, "Validation")

# 测试集预测
test_predictions = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Test Set Prediction'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        test_predictions.extend(predictions.cpu().tolist())

sample_submission = pd.read_csv('/root/autodl-tmp/sample_submission.csv')
sample_submission["target"] = test_predictions
sample_submission.to_csv('/root/autodl-tmp/submission.csv',index=False)