import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import csv

save_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

seq_length, input_size = torch.Size([750, 25])
output_size = 58
hidden_size = 128
# model = LSTMModel(input_size, hidden_size, output_size).to(device)
model = LSTMModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load(os.path.join(save_dir, 'lstm_model_200.pth')))

#输出csv的目录
output_folder = os.path.join(save_dir, "3DMMlabel")
num_epochs = 1
#读取csv的目录，
folder_path = os.path.join(save_dir, 'emotion')
for epoch in range(num_epochs):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        input_data = torch.tensor(data.values, dtype=torch.float32).to(device)
       
        model.eval()
        with torch.no_grad():
            output = model(input_data.unsqueeze(1))
            predictions=output.cpu().numpy()
            predictions = np.expand_dims(predictions, axis=1)  

        output_file = os.path.join(output_folder,filename + '.npy')
        np.save(output_file,predictions)
        print('Predictions saved as', output_file)