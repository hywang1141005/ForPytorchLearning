import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class LogicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogicNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        return x

    def predict(self, x):
        pred = torch.softmax(self.forward(x), dim=1)
        return torch.argmax(pred, dim=1)

    def get_loss(self, x, y):
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        return loss

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]