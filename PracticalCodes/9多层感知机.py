from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
args = Namespace(
    seed=42,
    num_samples_per_class=500,
    dimension=2,
    num_classes=3,
    train_size=0.75,
    test_size=0.25,
    num_hidden_units=100,
    learning_rate=0.01,
    regularization=1e-3,
    num_epochs=1000
)
np.random.seed(args.seed)

def generate_data(num_samples_per_class, dimensions, num_classes):
    X_original = np.zeros((num_samples_per_class * num_classes, dimensions))
    y = np.zeros(num_samples_per_class * num_classes, dtype="uint8")
    for j in range(num_classes):
        ix = range(num_samples_per_class * j, num_samples_per_class * (j+1))
        r = np.linspace(0.0, 1, num_samples_per_class)
        t = np.linspace(j * 4, (j+1) * 4, num_samples_per_class) + \
        np.random.randn(num_samples_per_class) * 0.2
        X_original[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
        
    X = np.hstack([X_original])
    return X, y
X, y = generate_data(args.num_samples_per_class, args.dimension, args.num_classes)
print(X.shape, y.shape)

plt.title("Generated non-linear data")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# 数据打乱
shuffle_indicies = torch.LongTensor(random.sample(range(0, len(X)), len(X)))
X = X[shuffle_indicies]
y = y[shuffle_indicies]
# 数据集划分
test_start_index = int(len(X) * args.train_size)
X_train = X[:test_start_index]
y_train = y[:test_start_index]
X_test = X[test_start_index:]
y_test = y[test_start_index:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
class LogisticClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x_in, apply_softmax=False):
        a_1 = self.fc1(x_in)
        y_pred = self.fc2(a_1)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred

model = LogisticClassifier(args.dimensions,
                           args.num_hidden_units,
                           args.num_classes)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

def get_accuracy(y_pred, y_target):
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_target)
    return accuracy

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
model.train() # 设置为训练模式

for epoch in range(args.num_epochs):
    y_pred = model(X_train)
    
    _, predictions = y_pred.max(dim=1)
    accuracy = get_accuracy(predictions.long(), y_train)
    
    loss = loss_fn(y_pred, y_train)
    if epoch % 20 == 0:
        print(f"epoch:{epoch:02d} | loss:{loss.item():.4f} | acc:{accuracy*100:.1f}%")
    # 梯度清0
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

# 预测
_, pred_train = model(X_train, apply_softmax=True).max(dim=1)
_, pred_test = model(X_test, apply_softmax=True).max(dim=1)
# 评估
train_acc = get_accuracy(pred_train.long(), y_train)
test_acc = get_accuracy(pred_test.long(), y_test)
print(f"train acc: {train_acc*100:.1f}% | test acc: {test_acc*100:.1f}%")

# 可视化
def plot_multiclass_decision_boundary(model, X, y):
    # 扩展边界
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    # 生成二维数组，每一行列均生成101个点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    # 颜色映射
    cmap = plt.cm.Spectral
    # 转成张量
    X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    # 预测坐标点
    y_pred = model(X_test, apply_softmax=True)
    _, y_pred = y_pred.max(dim=1)
    y_pred = y_pred.shape(xx.shape)
    plt.contourrf*(xx, yy, y_pred, cmap=cmap, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, camp=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Visualize the decision boundary
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_multiclass_decision_boundary(model=model, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_multiclass_decision_boundary(model=model, X=X_test, y=y_test)
plt.show()

import itertools
from sklearn.metrics import classification_report, confusion_matrix
# 混淆矩阵
def plot_confusion_matrix(cm, classes):
    # 颜色映射，blues使用蓝色
    cmap=plt.cm.Blues 
    plt.imshow(cm, interpolation='nearest', cmap=cmap) # 将混淆矩阵作为图像显示，interpolation='nearest'：使用最近邻插值，确保矩阵格子边界清晰、不模糊。
    plt.title("Confusion Matrix")
    plt.colorbar() # 在图像右侧添加颜色条，标明颜色与数值的映射关系。
    
    tick_marks = np.arange(len(classes))
    # 将刻度位置替换为类别名称，并将 x 轴标签旋转 45 度避免重叠
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False) # 关闭网格线

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # 生成所有格子坐标 (i, j)
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",  # plt.text(j, i, ...)：在 (j, i) 位置写入数值文本
                 color="white" if cm[i, j] > thresh else "black") # ‘d’：格式化为整数
    # 若格子数值大于最大值的一半（thresh），文字设为白色，否则为黑色，确保与背景对比度足够。
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() # 自动调整子图参数，使标签和标题不重叠。

# 绘制混淆矩阵
cm = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(cm=cm, classes=[0, 1, 2])
print (classification_report(y_test, pred_test))

# # # 非线性模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x_in, apply_softmax=False):
        a_1 = F.relu(self.fc1(x_in)) # 激活函数
        y_pred = self.fc2(a_1)
        
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred
# 初始化
model = MLP(input_dim=args.dimensions,
            hidden_dim=args.num_hidden_units,
            output_dim=args.num_classes)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
model.train() # 设置为训练模式
for epoch in range(args.num_epochs):
    y_pred = model(X_train)
    _, predictions = y_pred.max(dim=1)
    accuracy = get_accuracy(predictions.long())
    loss = loss_fn(y_pred, y_train)
    if epoch % 20 == 0:
        print(f"epoch:{epoch:02d} | loss:{loss.item():.4f} | acc:{accuracy*100:.1f}%")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
_, pred_train = model(X_train, apply_softmax=True).max(dim=1)
_, pred_test = model(X_test, apply_softmax=True).max(dim=1)
train_acc = get_accuracy(pred_train.long(), y_train)
test_acc = get_accuracy(pred_test.long(), y_test)
print(f"train acc: {train_acc*100:.1f}% | test acc: {test_acc*100:.1f}%")
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_multiclass_decision_boundary(model=model, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_multiclass_decision_boundary(model=model, X=X_test, y=y_test)
plt.show()
cm = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(cm=cm, classes=[0, 1, 2])
print (classification_report(y_test, pred_test))