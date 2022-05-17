# https://hongl.tistory.com/247

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torchsummary import summary as summary

training_set = pd.read_csv('airline-passengers.csv')
print(training_set.head())
training_set = training_set.iloc[:,1:2].values

# TRAINING PARAMETERS
batch_size = 1024
num_epochs = 3000
learning_rate = 0.001

# STRUCTURE PARAMETERS
input_size = 1  # could be thought as the number of input features
hidden_size = 3
num_layers = 2
num_classes = 1


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

norm = MinMaxScaler()
training_data = norm.fit_transform(training_set)

seq_length = 4
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 2/3)
test_size = len(y) - train_size

dataX = (torch.Tensor(np.array(x)))
dataY = (torch.Tensor(np.array(y)))

trainX = (torch.Tensor(np.array(x[0:train_size])))
trainY = (torch.Tensor(np.array(y[0:train_size])))

testX = (torch.Tensor(np.array(x[train_size:len(x)])))
testY = (torch.Tensor(np.array(y[train_size:len(y)])))

train = torch.utils.data.TensorDataset(trainX, trainY)
test = torch.utils.data.TensorDataset(testX, testY)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, loop_cnt):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0))

        if(loop_cnt == 1):
            print('\n===========================')
            print('batch size (= x(0)_size) :', x.size(0))
            print('x_size :', x.size(), '# [batch, seq_length, input_size]')
            print('\nc_0 size :', c_0.size(),
                       '# [num_layers, batch, hidden_size]')
            print('h_0 size :', h_0.size(),
                       '# [num_layers, batch, hidden_size]')
            print('===========================\n\n')

        #hn = hn.view(-1, self.hidden_size)
        # https://dhpark1212.tistory.com/entry/RNN-LSTM-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84pytorch
        # -1 :  h_output(마지막 hidden_state 출력 값)
        # 만약 아래처럼 말고, Output_layer를 리턴 값으로 쓰고자 한다면, Seq_len의 마지막 것만 써야 한다. 즉, Output[-1]
        # cf.  lstm_out은 모든 시간 단계(시퀀스의 모든 항목)에 대한 마지막 hidden state를 포함
        #      h_out(hidden cell output)은 시간 단계 수와 관련하여 마지막 hidden state를

        hn = hn[-1,:,:]
        hn = hn.view(-1, self.hidden_size)

        out = self.fc(hn)

        return out



if __name__ == "__main__":

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    count = 0
    for name, param in lstm.named_parameters():
        count += 1
        if count == 1:
            print(param.size())
        print(name)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    loop_cnt = 0
    for epoch in range(num_epochs):
        running_loss = 0.0

        for data in train_loader:
            loop_cnt+=1
            seq, target = data
            out = lstm(seq, loop_cnt)

            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        if epoch % 100 == 0:
            print("Epoch: %d, running_loss: %1.5f" % (epoch, running_loss))

    lstm.eval()
    torch.save(lstm, 'model.pt')
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
    plt.savefig('test.png')
    plt.show()
