from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\hywan\Dropbox\PC\Desktop\python\dl\pytorch\ch3")
import torch
from code_02_moons_fun import LogicNet, moving_average

np.random.seed(0)

X, Y = make_moons(200, noise=0.2)

arg = np.squeeze(np.argwhere(Y==0))
arg2 = np.squeeze(np.argwhere(Y==1))

plt.title("moons data")
plt.scatter(X[arg, 0], X[arg, 1], s=100, c="b", marker="+", label="data1")
plt.scatter(X[arg2, 0], X[arg2, 1], s=100, c="r", marker="o", label="data2")
plt.legend()
plt.show()

input_dim=2
hidden_dim=3
output_dim=2
model = LogicNet(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

xt = torch.from_numpy(X).type(torch.float32).to(device)
yt = torch.from_numpy(Y).type(torch.int64).to(device)


epoch = 1000
losses = []
for i in range(1000):
    loss = model.get_loss(xt, yt)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def plot_losses(losses):
    avgloss = moving_average(losses)
    plt.figure()
    plt.plot(range(len(avgloss)), avgloss, "b--")
    plt.xlabel("step number")
    plt.ylabel("training loss")
    plt.title("step number vs training loss")
    plt.show()
plot_losses(losses)

from sklearn.metrics import accuracy_score
accuracy_score(model.predict(xt).cpu(), yt.cpu())

def predict(x):
    x = torch.from_numpy(x).type(torch.float32)
    model.cpu()
    ans = model.predict(x)
    return ans.numpy()

def plot_decision_bondary(pred_func, X, Y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    z = pred_func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contour(xx, yy, z, cmap=plt.cm.Spectral)
    plt.title("Linear Predict")
    arg = np.squeeze(np.argwhere(Y==0))
    arg2 = np.squeeze(np.argwhere(Y==1))

    plt.scatter(X[arg, 0], X[arg, 1], color="b", s=100, marker="+", label="data1")
    plt.scatter(X[arg2, 0], X[arg2, 1], color="r", s=40, marker="o", label="data2")
    plt.legend()
    plt.show()

plot_decision_bondary(lambda x : predict(x), X, Y)