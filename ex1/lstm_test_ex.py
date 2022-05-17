# https://hongl.tistory.com/247

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from lstm_ex import LSTM, sliding_windows


training_set = pd.read_csv('airline-passengers.csv')
training_set = training_set.iloc[:,1:2].values

# TRAINING PARAMETERS
batch_size = 32

norm = MinMaxScaler()
training_data = norm.fit_transform(training_set)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 2/3)
test_size = len(y) - train_size

dataX = (torch.Tensor(np.array(x)))
dataY = (torch.Tensor(np.array(y)))

testX = (torch.Tensor(np.array(x[train_size:len(x)])))
testY = (torch.Tensor(np.array(y[train_size:len(y)])))
test = torch.utils.data.TensorDataset(testX, testY)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)


lstm = torch.load('model.pt')
lstm.eval()
loop_cnt = -1
train_predict = lstm(dataX, loop_cnt)


data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = norm.inverse_transform(data_predict)
dataY_plot = norm.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot, label = 'Real')
plt.plot(data_predict, label = 'Predict')
plt.suptitle('Time-Series Prediction After Training')
plt.grid()
plt.legend()
plt.show()