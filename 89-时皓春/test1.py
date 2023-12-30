import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.activation = nn.Softmax(dim=1)  # 使用softmax函数进行多分类归一化
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x):
        return self.activation(self.linear(x))  # 输出经过softmax分类

# 生成多分类样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[2] and x[0]>x[4]:
        return x, 0  # 标记为第0类
    elif x[2] > x[0] and x[2]>x[4]:  # 修改为三个类别
        return x, 1  # 标记为第1类
    else:
        return x, 2  # 标记为第2类

# 生成多分类训练数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 模型评估
def evaluate(model, x, y):
    model.eval()
    y_pred = model(x)
    _, predicted = torch.max(y_pred, 1)
    correct = predicted.eq(y).sum().item()
    print("正确的预测:", correct,"ccccccccccc",predicted)
    print("准确率:", correct / len(y))

def main():
    # 参数设置
    num_classes = 3  # 类别数量（多分类任务）
    input_size = 5  # 输入向量维度
    learning_rate = 0.001
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_num = 20
    batch_size = 20
    total_sample_num = 5000
    train_x, train_y = build_dataset(total_sample_num)
    for epoch in range(epoch_num):
        model.train()
        for batch_index in range(total_sample_num // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            y_pred = model(x)  # 前向传播
            loss = model.loss(y_pred, y)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
        print("轮数 {}: 损失：:{}".format(epoch+1, loss.item()))
        evaluate(model, train_x, train_y)

if __name__ == "__main__":
    main()
