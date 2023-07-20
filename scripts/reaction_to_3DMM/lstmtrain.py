import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import csv

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=("cuda:1")
save_dir = os.path.dirname(os.path.abspath(__file__))
s2='./data/train/3D_FV_files'
s1='./data/train/emotion'

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.fc(lstm_out)
        return output

# 读取所有输入数据

with open(os.path.join(save_dir, 'train_idx.csv'), 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)
    # 将每一行数据转换为数组
    train_ids = np.array([row[0] for row in csv_reader])


#（750，25）的csv对应（751，1，58）的npy路径
def get_path(input_path):   # get 3dmm .npy
    new_path = os.path.join(s2, input_path)
    new_path = os.path.splitext(new_path)[0] + '.npy'               
    return new_path 

def back_path(key_path):  #get emotion .csv
    new_path = os.path.join(s1, key_path)
    new_path = os.path.splitext(new_path)[0] + '.csv'               
    return new_path 

from tensorboardX import SummaryWriter
writer = SummaryWriter(os.path.join(save_dir, "log"))
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, train_ids):
        self.train_ids = train_ids

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        train_id = self.train_ids[idx]
        input_file=back_path(train_id)
        target_file=get_path(train_id)
        data = pd.read_csv(input_file)
        target = np.load(target_file)
        target = np.squeeze(target)[:-1]
        input_data = torch.tensor(data.values, dtype=torch.float32)
        target_data = torch.tensor(target, dtype=torch.float32)

        return {'input': input_data, 'target': target_data}

dataloader = DataLoader(MyDataset(train_ids), batch_size=64, shuffle=True)

# 定义模型参数
# 创建LSTM模型、损失函数和优化器
seq_length, input_size = torch.Size([750, 25])
output_size = 58
hidden_size = 128
# model = LSTMModel(input_size, hidden_size, output_size).to(device)
model = LSTMModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.train()

num_epochs = 200
n_iter = 1
for epoch in tqdm(range(1, num_epochs+1)):
    for batch in dataloader:
        
        input_data = batch['input'].to(device)
        target_data = batch['target'].to(device)

    # 训练模型
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss_mse", float(loss), n_iter)
        n_iter += 1
        # print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    if epoch % 5 == 0 or epoch == num_epochs:
        torch.save(model.state_dict(), os.path.join(save_dir, f'lstm_model_{epoch}.pth'))
