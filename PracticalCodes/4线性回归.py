from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1.生成数据
# 设置参数
args = Namespace(
    seed = 42,
    data_file = "sample_data.csv",
    num_samples = 100,
    train_size = 0.75,
    test_size = 0.25,
    num_epochs = 10000,
)
# 随机数，种子
np.random.seed(args.seed)

def generate_data(num_samples):
    # [0, 1)之间的随机数，乘以100，得到0到100之间的随机数]
    X = np.random.rand(num_samples) * 100
    # 生成 从 0 开始的连续整数，长度为 num_samples
    #X = np.array(range(num_samples))
    y = 3.65 * X + 10
    return X, y
# 一维数组是没有行列区分的

X, y = generate_data(args.num_samples)
# np.vstack((X, y)) 垂直堆叠X和y，得到一个2行num_samples列的数组
data = np.vstack((X, y)).T
# 把 numpy 二维数组 转换成 Pandas 表格（DataFrame）
df = pd.DataFrame(data, columns=["X", "y"])

# 画散点图
plt.title("Scatter Plot of X and y")
plt.xlabel("X")
plt.ylabel("y") 
plt.scatter(df["X"], df["y"], color="blue", label="Data Points")
plt.legend()
plt.show()


# 2.scikit-learn 线性回归
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    # 机器学习模型要求特征（X）必须是二维数组：(样本数, 特征数)
    df["X"].values.reshape(-1, 1), df["y"], 
    test_size=args.test_size, random_state=args.seed
)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# 标准化（特征缩放），使得特征具有零均值和单位方差

X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))

# 对训练集和测试集进行标准化
X_train_scaled = X_scaler.transform(X_train)
# ravel() -> 把标准化后的二维 y，转回一维数组（和原始形状一致）
y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()

X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# 检查
print("mean of X_train_scaled:", np.mean(X_train_scaled))
print("std of X_train_scaled:", np.std(X_train_scaled))
print("mean of y_train_scaled:", np.mean(y_train_scaled))
print("std of y_train_scaled:", np.std(y_train_scaled))

# 初始化模型
lm = SGDRegressor(loss="squared_error", penalty=None, 
                  max_iter=args.num_epochs, random_state=args.seed)

# 训练模型
lm.fit(X=X_train_scaled, y=y_train_scaled)

# 预测(逆标准化后的原始数据)
pred_train = lm.predict(X_train_scaled) * y_scaler.scale_ + y_scaler.mean_
pred_test = lm.predict(X_test_scaled) * y_scaler.scale_ + y_scaler.mean_

# 3.评估
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print("train_mse: {:.2f}, test_mse: {:.2f}".format(train_mse, test_mse))

# 图例大小
plt.figure(figsize=(15, 5))
# 训练集图
# 1 行 2 列，左面的图
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(X_train, y_train, label="y_train")
plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")
plt.legend()

# 测试集图
# 1 行 2 列，右面的图
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(X_test, y_test, label="y_test")
plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
plt.legend()

plt.show()

# 4.推论
X_infer = np.array((0, 1, 2), dtype=np.float32)
standardized_X_infer = X_scaler.transform(X_infer.reshape(-1, 1))
pred_infer = lm.predict(standardized_X_infer) * y_scaler.scale_ + y_scaler.mean_
print("Inference for X_infer:", X_infer)

# 5.可解释性
coef = lm.coef_ * (y_scaler.scale_ / X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - coef * X_scaler.mean_
print("Learned coefficient (slope): {:.2f}".format(coef[0]))
print("Learned intercept: {:.2f}".format(intercept[0]))

# 6.正则化
lm = SGDRegressor(loss="squared_error", penalty="l2", alpha=0.01,
                  max_iter=args.num_epochs, random_state=args.seed)

lm.fit(X=X_train_scaled, y=y_train_scaled)
# 预测(逆标准化后的原始数据)
pred_train = lm.predict(X_train_scaled) * y_scaler.scale_ + y_scaler.mean_
pred_test = lm.predict(X_test_scaled) * y_scaler.scale_ + y_scaler.mean_
# 评估
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print("With L2 regularization - train_mse: {:.2f}, test_mse: {:.2f}".format(train_mse, 
                                                                            test_mse))
# 未标准化系数
coef = lm.coef_ * (y_scaler.scale_ / X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - coef * X_scaler.mean_
print("With L2 regularization - Learned coefficient (slope): {:.2f}".format(coef[0]))
print("With L2 regularization - Learned intercept: {:.2f}".format(intercept[0]))

# 7.类别变量（离散变量）
# 创建类别特征
cat_data = pd.DataFrame(['a', 'b', 'c', 'a'], columns=['favorite_letter'])
cat_data.head()

dummy_cat_data = pd.get_dummies(cat_data) #独热编码 one-hot encoding，与dummy变量不同要注意。
dummy_cat_data.head()