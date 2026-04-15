from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib

args = Namespace(
    seed=42,
    data_file="titanic.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
    max_depth=4,
    min_samples_leaf=5,
    n_estimators=10, # 随机森林中的决策树的个数
)
np.random.seed(args.seed)

df = pd.read_csv(args.data_file, header=0)
print(df.head())

from sklearn.tree import DecisionTreeClassifier
# 预处理
def preprocess(df):
    df = df.dropna() # 删除缺失值
    # 删除基于文本的特征
    features_to_drop = ["name", "cabin", "ticket"]
    df = df.drop(features_to_drop, axis=1)
    # 将类别变量转换为数值
    df["sex"] = df["sex"].map({"female": 0, "male": 1}).astype(int)
    df["embarked"] = df["embarked"].map({"S":0, "C":1, "Q":2}).astype(int)
    
    return df

df = preprocess(df)
print(df.head())

# 划分训练集和测试集
mask = np.random.rand(len(df)) < args.train_size
train_df = df[mask]
test_df = df[~mask]
print("训练集大小:", len(train_df))
print("测试集大小:", len(test_df))

# 分离特征和标签
X_train = train_df.drop(["survived"], axis=1)
y_train = train_df["survived"]
X_test = test_df.drop(["survived"], axis=1)
y_test = test_df["survived"]

# 初始化模型
dtree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=args.seed,
    max_depth=args.max_depth,
    min_samples_leaf=args.min_samples_leaf # 叶子里面的最小样本数
)

dtree.fit(X_train, y_train)

# 预测
pred_train = dtree.predict(X_train)
pred_test = dtree.predict(X_test)

# 评估
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_test, 
                                                           average="binary")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 可解释性
import os
os.environ["PATH"] += os.pathsep + "D:/Miniconda3/Graphviz/bin/"

from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# 创建内存缓冲区
dot_data  = StringIO()

# 核心函数
export_graphviz(
    dtree,
    out_file=dot_data,
    feature_names=list(train_df.drop(["survived"], axis=1)),
    class_names=["died", "survived"],
    rounded=True,
    filled=True,
    special_characters=True
)

# 读取内存中的绘图数据，生成graphviz图形对象
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# 保存图像
graph.write_png("决策树可视化结果.png")  

# 特征重要分数
features = list(X_test.columns)
importances = dtree.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(features)
plt.figure()
plt.title("Feature Importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices])
plt.xlim([-1, num_features])
plt.show()




# -----------------------------------------随机森林-----------------------------------------

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=args.n_estimators,
    criterion="entropy",
    max_depth=args.max_depth,
    min_samples_leaf=args.min_samples_leaf,
    random_state=args.seed
)
forest.fit(X_train, y_train)

pred_train = forest.predict(X_train)
pred_test = forest.predict(X_test)

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print(f"随机森林训练集准确率: {train_acc:.4f}")
print(f"随机森林测试集准确率: {test_acc:.4f}")

score= precision_recall_fscore_support(y_test, pred_test, average="binary")
print(f"随机森林精确率: {score[0]:.4f}")
print(f"随机森林召回率: {score[1]:.4f}")    
print(f"随机森林F1分数: {score[2]:.4f}")

# 可解释性
features = list(X_test.columns)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
num_features = len(features)
plt.figure()
plt.title("Random Forest Feature Importances")
plt.bar(range(num_features), 
        importances[indices], 
        color="g", 
        yerr=std[indices], 
        align="center")
plt.xticks(range(num_features), [features[i] for i in indices])
plt.xlim([-1, num_features])
plt.show()

# 网格搜索
from sklearn.model_selection import GridSearchCV
param_grid = {
    "bootstrap" : [True],
    "max_depth" : [10, 20, 50],
    "max_features" : [len(features)],
    "min_samples_leaf" : [3, 4, 5],
    "min_samples_split" : [4, 8],
    "n_estimators" : [5, 10, 50]
}
forest = RandomForestClassifier(random_state=args.seed)

grid_search = GridSearchCV(estimator=forest,
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=1
                           )
grid_search.fit(X_train, y_train)
grid_search.best_params_
best_forest = grid_search.best_estimator_
best_forest.fit(X_train, y_train)

pred_train = best_forest.predict(X_train)
pred_test = best_forest.predict(X_test)

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print(f"网格搜索随机森林训练集准确率: {train_acc:.4f}")
print(f"网格搜索随机森林测试集准确率: {test_acc:.4f}")
score= precision_recall_fscore_support(y_test, pred_test, average="binary")
print(f"网格搜索随机森林精确率: {score[0]:.4f}")
print(f"网格搜索随机森林召回率: {score[1]:.4f}")
print(f"网格搜索随机森林F1分数: {score[2]:.4f}")
