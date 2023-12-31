# Tumor-recognition

基于机器学习的肿瘤特征识别，使用了六个机器学习的模型进行交叉验证。

样本量是1256例，良性组900多，恶性组300多，总共30个变量。

验证指标：Accuracy、Precision、Recall、F1 Score、Confusion Matrix

以下是六个机器学习的模型：


支持向量机（Support Vector Machines，SVM）：使用超平面来分隔数据点，并尽量使不同类别的数据点之间的间隔最大化。

逻辑回归（Logistic Regression）：通过将输入特征与权重相乘并应用逻辑函数来进行二分类预测。

决策树（Decision Trees）：基于一系列特征的条件判断来构建树状模型，以实现分类任务。

随机森林（Random Forest）：由多个决策树构成的集成学习模型，通过投票或平均预测结果来进行分类。

梯度提升树（Gradient Boosting Trees）：通过迭代地训练多个决策树来提高模型性能，每次迭代都根据前一次的残差进行训练。

k近邻算法（k-Nearest Neighbors，k-NN）：基于距离度量来对新样本进行分类，即将其归类为与其最近的k个训练样本的多数类别。
