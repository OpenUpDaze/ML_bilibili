import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

filepath = ''
class Dataset1(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shap[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

dataset = Dataset1(filepath)
train_loader = DataLoader(dataset=dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0)
 # 构造模型                       
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
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    for i, data in enumerate(train_loader,0):
        # 1. prepare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        print(epoch,i,loss.item())

        # 3. Backward
        optimizer.zero_grad()
        loss.backward()

        # 4.Update
        optimizer.step()

        
