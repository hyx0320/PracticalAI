from argparse import Namespace 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request
args = Namespace(
    seed=42,
    data_file="titanic.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=10000,
)
np.random.seed(args.seed)

# 1.加载数据
'''
url = "https://raw.githubusercontent.com/LisonEvf/practicalAI-cn/master/data/titanic.csv"
response = urllib.request.urlopen(url)
html = response.read()
with open(args.data_file, "wb") as f:
    f.write(html)
'''
df = pd.read_csv(args.data_file, header=0)

# 2.模型实现
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 3.数据预处理
def preprocess(df):
    # 删除含有空值的行
    df = df.dropna()
    
    # 删除基于文本的特征
    features2drop = ["name", "ticket", "cabin"]
    df = df.drop(features2drop, axis=1)
    
    # pclass、sex、embarked特征进行独热编码
    categorical_features = ["pclass", "sex", "embarked"]
    df = pd.get_dummies(df, columns=categorical_features)
    
    return df

df = preprocess(df)

# 4.划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("survived", axis=1), df["survived"], 
    test_size=args.test_size, random_state=args.seed
)

# 5.特征标准化
X_scaler = StandardScaler().fit(X_train)

X_train = X_scaler.transform(X_train)
feature_columns = X_test.columns
X_test = X_scaler.transform(X_test)

print("mean of X_train:", X_train.mean(axis=0))
print("std of X_train:", X_train.std(axis=0))

# 6.训练模型
log_reg = SGDClassifier(loss="log_loss", 
                        penalty=None, max_iter=args.num_epochs,
                        random_state=args.seed)
log_reg.fit(X_train, y_train)

pred_test = log_reg.predict_proba(X_test)
print(pred_test[:5])

# 预测（原始数据）
pred_train = log_reg.predict(X_train)
pred_test = log_reg.predict(X_test)
print(pred_test)

# 7.评估模型
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 混淆矩阵
import itertools
from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(cm, classes):
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation="nearest", cmap=cmap) 
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)
    
    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

cm = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(cm, classes=["died", "Survived"])
print(classification_report(y_test, pred_test, target_names=["died", "Survived"]))

# 8.推论
X_infer = pd.DataFrame([{
    "name" : "Goku Mohandas", "cabin" : "E", "ticket" : "E44",
    "pclass" : 1, "age" : 24, "sibsp" : 1, "parch" : 2, "fare" : 100, "embarked" : "C",
    "sex" : "male"
}])

X_infer = preprocess(X_infer)

# 添加缺失列向量
missing_features = set(feature_columns) - set(X_infer.columns)
for feature in missing_features:
    X_infer[feature] = 0

X_infer = X_infer[feature_columns]

# 特征标准化
X_infer = X_scaler.transform(X_infer)

pred_infer = log_reg.predict(X_infer)
print(pred_infer)

# 9.可解释性
coef = log_reg.coef_ / X_scaler.scale_
intercept = log_reg.intercept_ - np.sum(coef * X_scaler.mean_) 
print("Coefficients:", coef)
print("Intercept:", intercept)

indices = np.argsort(coef)
features = list(feature_columns)
print ("Features correlated with death:", [features[i] for i in indices[0][:3]])
print ("Features correlated with survival:", [features[i] for i in indices[0][-3:]])

# 10.k折交叉验证
from sklearn.model_selection import cross_val_score
log_reg = SGDClassifier(loss="log_loss", penalty=None, max_iter=args.num_epochs, random_state=args.seed)
scores = cross_val_score(log_reg, X_train, y_train, cv=10, scoring="accuracy")

print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())
print("Standard deviation of CV scores:", scores.std())