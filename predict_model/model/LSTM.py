import torch
import torch.nn as nn
import numpy as np


class LSTMSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(LSTMSubNet, self).__init__()
        self.lstm = nn.LSTM(in_c, hid_c, batch_first=True)
        self.fc = nn.Linear(hid_c, out_c)
        self.act = nn.LeakyReLU()

    def forward(self, inputs):
        """
        :param inputs: [B, N, T, C]
        :return: [B, N, T, D]
        """
        B, N, T, _ = inputs.size()
        inputs = inputs.view(B, N, -1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        outputs = self.act(outputs)
        return outputs


class LSTMNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(LSTMNet, self).__init__()
        self.subnet = LSTMSubNet(in_c, hid_c, out_c)

    def forward(self, data, device):
        flow = data["flow_x"]
        flow = flow.to(device)
        prediction = self.subnet(flow)
        return prediction
