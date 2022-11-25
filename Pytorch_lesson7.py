import torch
import numpy as np
xy = np.loadtxt(r'C:\Users\HP\Desktop\ML_bilibili\heart.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

#-----------------------------------------------------#

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(13,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
model = Model()
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)
#--------------------------------------------------------------------#

criterion = torch.nn.BCELoss(reduction='mean' )
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

#--------------------------------------------------------------------#

for epoch in range(100):
    #Forward
    x_data, y_data = x_data.to(device), y_data.to(device)
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Update
    optimizer.step()
    
xy = np.loadtxt(r'C:\Users\HP\Desktop\ML_bilibili\heart_test.csv', delimiter=',', dtype=np.float32)
x_test = torch.from_numpy(xy[:,:-1])
y_test = torch.from_numpy(xy[:,[-1]])

y_pred = model(x_test)

print('y_pred = ', y_pred.data)












