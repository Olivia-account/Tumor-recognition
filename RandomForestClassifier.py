from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
# 读取数据
data = pd.read_excel('data.xlsx', header=1)

# 转换列名为字符串类型
data.columns = data.columns.astype(str)

# 提取特征和目标变量
X = data.iloc[:, 1:-1] # 选择特征列
y = data.iloc[:, -1] # 选择目标变量列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义SVM模型
model = RandomForestClassifier()

# 定义超参数网格
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}

# 使用交叉验证和网格搜索选择最佳超参数设置
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 获取最佳超参数设置
best_params = grid_search.best_params_

# 使用最佳超参数重新训练模型
model = model.set_params(**best_params)
model.fit(X_train, y_train)

# 使用交叉验证评估模型性能
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
average_cv_score = cv_scores.mean()

# 预测
y_pred = model.predict(X_test)

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# 打印结果和指标
print("Model: RandomForestClassifier")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion_mat)