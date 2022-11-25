import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])  #3行1列
y_data = torch.Tensor([[0], [0], [1]])

class LinearModel(torch.nn.Module):      #继承torch中torch.nn.Module的类
    def __init__(self):
        super(LinearModel, self).__init__()  #python2 中调用父类方法，形式为：super(Class, self).xxx ，python3中用super().xxx
        self.linear = torch.nn.Linear(1,1)   #y = xA^T + b
        self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()          #torch.nn.Linear类中有call方法，能让此类能够像函数一样传参数, in_features，out_features是输入、输出的列，
    
    def forward(self, x):      #一定要有这个方法，并且名字叫forward，用来覆盖掉torch.nn.Module中的同名方法
        x = self.linear(x)
        x = self.activate(x)
        y_pred = self.sigmoid(self.linear(x))
        return y_pred
model = LinearModel()    #callable实例

criterion = torch.nn.BCELoss(size_average=False)     #类，也是继承自nn.Module；求均方误差（MSE）损失函数；size_average = False代表不求均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  #优化器，不会构建计算图，lr为学习率

for epoch in range(1000):
    y_pred = model(x_data)  #前馈算y hat
    loss = criterion(y_pred, y_data) #算损失
    print(epoch, loss.item())  #打印

    # print('y_pred = ', y_pred.data)
    optimizer.zero_grad()   #将所有权重的梯度归0
    loss.backward()      #反向传播
    optimizer.step()       #更新权重w  

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0],[6.0], [3.0], [4.], [8.]])
y_test = model(x_test)
print('y_pred = ', y_test.data)



