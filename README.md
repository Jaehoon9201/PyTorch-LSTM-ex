# What is a LSTM ? 
------------------
+ The parameters in the figure are the same as the parameters written in the code

+ Figures are made by Jaehoon Shim

------------------

![image](https://user-images.githubusercontent.com/71545160/171369810-1b5b77bf-aa9b-4289-931f-dab0b2ea276a.png)

<br>
<br>
<br>

------------------

<br>
<br>
<br>

![image](https://user-images.githubusercontent.com/71545160/171369856-ecd95a73-5db6-455e-b406-176656010678.png)

<br>
<br>
<br>

------------------

<br>
<br>
<br>

![image](https://user-images.githubusercontent.com/71545160/171370215-376be51f-622d-4293-b925-184a4540bd3d.png)

<br>
<br>
<br>

------------------

<br>
<br>
<br>You can also refer to the below sites.

[Refernce sites](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=songblue61&logNo=221853600720)

Above sites represents the LSTM example structure very well.

Below one is one of the examples.

![image](https://user-images.githubusercontent.com/71545160/171370770-9033eb6c-58b9-4f81-af24-3ffc31933960.png)


[Refernce sites](https://wegonnamakeit.tistory.com/52)


# PyTorch-LSTM-ex
------------------

PyTorch-LSTM-ex

## Results of **lstm_ex.py**
```
E:\TorchProject\venv\Scripts\python.exe E:/TorchProject/venv/LSTM/ex1/lstm_ex.py
     Month  Passengers
0  1949-01         112
1  1949-02         118
2  1949-03         132
3  1949-04         129
4  1949-05         121
torch.Size([12, 1])
lstm.weight_ih_l0
lstm.weight_hh_l0
lstm.bias_ih_l0
lstm.bias_hh_l0
lstm.weight_ih_l1
lstm.weight_hh_l1
lstm.bias_ih_l1
lstm.bias_hh_l1
fc.weight
fc.bias

===========================
batch size (= x(0)_size) : 92
x_size : torch.Size([92, 4, 1]) # [batch, seq_length, input_size]

c_0 size : torch.Size([2, 92, 3]) # [num_layers, batch, hidden_size]
h_0 size : torch.Size([2, 92, 3]) # [num_layers, batch, hidden_size]
===========================


Epoch: 0, running_loss: 0.08336
Epoch: 100, running_loss: 0.01923
Epoch: 200, running_loss: 0.01791
Epoch: 300, running_loss: 0.01676
Epoch: 400, running_loss: 0.01455
Epoch: 500, running_loss: 0.00867
Epoch: 600, running_loss: 0.00420
Epoch: 700, running_loss: 0.00407
Epoch: 800, running_loss: 0.00390
Epoch: 900, running_loss: 0.00370
Epoch: 1000, running_loss: 0.00346
Epoch: 1100, running_loss: 0.00318
Epoch: 1200, running_loss: 0.00290
Epoch: 1300, running_loss: 0.00262
Epoch: 1400, running_loss: 0.00236
Epoch: 1500, running_loss: 0.00214
Epoch: 1600, running_loss: 0.00196
Epoch: 1700, running_loss: 0.00183
Epoch: 1800, running_loss: 0.00175
Epoch: 1900, running_loss: 0.00172
Epoch: 2000, running_loss: 0.00170
Epoch: 2100, running_loss: 0.00170
Epoch: 2200, running_loss: 0.00169
Epoch: 2300, running_loss: 0.00168
Epoch: 2400, running_loss: 0.00167
Epoch: 2500, running_loss: 0.00166
Epoch: 2600, running_loss: 0.00165
Epoch: 2700, running_loss: 0.00165
Epoch: 2800, running_loss: 0.00164
Epoch: 2900, running_loss: 0.00163

Process finished with exit code 0
```

![image](https://user-images.githubusercontent.com/71545160/168735286-d853cd05-b4fa-4f60-b091-d7df4887e4de.png)

## Results of **lstm_test_ex.py**

You might obtain same results with above. <br>
It repeats tesing without training a model.
