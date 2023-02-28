import torch
import torch.nn as nn
import numpy as np

class Energy_EncoderNet(nn.Module):
    def __init__(self,seq_len,input_size,hidden_size,number_layer,batch_first):
        super(Energy_EncoderNet, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.number_layer = number_layer
        self.RNN = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = number_layer,
            batch_first = batch_first,
            # dropout = 0,
        )
        self.attention = Self_Attention(a=hidden_size,
                                        q=hidden_size,
                                        k=hidden_size,
                                        v=hidden_size, h=1)
        self.linear = nn.Linear(hidden_size*number_layer,hidden_size)

    def forward(self,x,h0):
        output,hn = self.RNN(x,h0)
        out = hn.permute(1, 0, 2)
        out = out.reshape((len(x),self.number_layer*self.hidden_size))
        out = self.linear(out)
        out = self.attention(out)
        out = out.reshape((len(output),1,self.hidden_size))
        return out


class Encoder_Net(nn.Module):
    def __init__(self,seq_len,input_size, hidden_size, number_layer, batch_first):
        super(Encoder_Net, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.number_layer = number_layer
        self.RNN = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=number_layer,
                           batch_first=batch_first, )
        self.attention = Self_Attention(a=hidden_size*1,
                                        q=hidden_size*1,
                                        k=hidden_size*1,
                                        v=hidden_size*1, h=1)
        self.linear = nn.Linear(hidden_size*self.number_layer,hidden_size*1)
        self.embedding = nn.Embedding(365,4)
    def forward(self, x, time, h0, c0):
        time = time.long()
        a, b, c = time.shape
        time = self.embedding(time)
        time = time.reshape(a,b,c*4)
        input = torch.cat((x,time),dim=2)
        output,(hn,cn) = self.RNN(input,(h0,c0))
        out = hn.permute(1, 0, 2)
        out = out.reshape((len(input),self.number_layer*self.hidden_size))
        out = self.linear(out)
        out = self.attention(out)
        out = out.reshape((len(output),1,self.hidden_size))
        return out,hn,cn

class Self_Attention(nn.Module):
    def __init__(self, a, h, k, v, q, dropout=0.1):
        """
        :param a: Output dimensionality of the previous model
        :param h: number of heads
        :param k: Dimensionality of and keys
        :param v: Dimensionality of values
        :param q: Dimensionality of queries
        """
        super(Self_Attention, self).__init__()

        self.fc_q = nn.Linear(a, h * q)  # 带参数
        self.fc_k = nn.Linear(a, h * k)
        self.fc_v = nn.Linear(a, h * v)
        self.fc_o = nn.Linear(h * v, a)
        self.dropout = nn.Dropout(dropout)

        self.a = a
        self.k = k
        self.v = v
        self.h = h

    def forward(self, x):
        b_s, nq = x.shape[:2]
        q = self.fc_q(x)
        k = self.fc_k(x).view(b_s, nq).permute(1, 0)
        v = self.fc_v(x)

        att = torch.matmul(k, q) / np.sqrt(self.k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(v, att)
        out = self.fc_o(out)
        return out


class Decoder_Net(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, number_layer, batch_first):
        super(Decoder_Net, self).__init__()
        self.output_size = output_size
        self.input_size = input_size

        self.RNN = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=number_layer,
                           batch_first=batch_first, )
        self.Linear = nn.Linear(hidden_size * number_layer, output_size)
        self.embedding_time = nn.Embedding(365,4)

    def forward(self,x,time,angle,h0,c0):
        time = time.long()
        a,b,c = time.shape
        time = self.embedding_time(time)
        time = time.reshape((a,b,c*4))
        input = torch.cat((x,time,angle),dim=2)
        output, (hn, cn) = self.RNN(input, (h0, c0))
        out = hn.permute(1, 0, 2)
        a, b, c = out.shape
        out = out.reshape((a, b * c))
        out = self.Linear(out)
        out = out.reshape((a, 1, self.output_size))
        return out, hn, cn
