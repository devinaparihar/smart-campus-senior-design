import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions

INPUT_SIZE = 2
HIDDEN_STATE_SIZE = 8
OUTPUT_SIZE = 5

class DeepLSTM(nn.Module):

    def __init__(self):
        super(DeepLSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(INPUT_SIZE, HIDDEN_STATE_SIZE)
        self.lstm2 = nn.LSTMCell(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE)
        self.l1 = nn.Linear(INPUT_SIZE + HIDDEN_STATE_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        h_1_tm1 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        c_1_tm1 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        h_2_tm1 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        c_2_tm1 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        y_list = []
        for i in range(x.size()[1]):
            x_t = torch.squeeze(x.narrow(1, i, 1), 1)
            x_1_t = x_t
            print("x_1_{} = {}".format(i, x_1_t))
            h_1_t, c_1_t = self.lstm1(x_1_t, (h_1_tm1, c_1_tm1))
            print(x_t.size(), h_1_t.size(), c_1_t.size())
            x_2_t = torch.cat((x_t, h_1_t), dim=1)
            print("x_1_{} = {}".format(i, x_2_t))
            h_2_t, c_2_t = self.lstm2(x_2_t, (h_2_tm1, c_2_tm1))
            x_3_t = torch.cat((x_t, h_2_t), dim=1)
            print("x_1_{} = {}".format(i, x_3_t))
            y_t = self.l1(x_3_t)
            y_list.append(y_t)
            print("y_{} = {}".format(i, y_t))
        return torch.stack(y_list, dim=1)


"""
means = torch.from_numpy(np.array([[0, 0], [1, 1]])).float()
std_devs = [1, 1]
correlation = 0.5
covariance = std_devs[0] * std_devs[1] * correlation

covariance_matrix = torch.from_numpy(np.array([[std_devs[0]**2, covariance], [covariance, std_devs[1]**2]])).float()
matrices = torch.stack([covariance_matrix, covariance_matrix])

print(means)
print(matrices)

dist = distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=matrices)
print(dist)
sample1 = torch.from_numpy(np.array([0, 0])).float()
sample2 = torch.from_numpy(np.array([1, 1])).float()

print(dist.log_prob(sample1))
print(dist.log_prob(sample2))

samples = torch.stack([sample1, sample2])
print(-1 * dist.log_prob(samples))
exit()
"""


def y_to_dist(y):
    mu_1 = y.narrow(2, 0, 1)
    mu_2 = y.narrow(2, 1, 1)
    sigma_1 = torch.exp(y.narrow(2, 2, 1))
    sigma_2 = torch.exp(y.narrow(2, 3, 1))
    rho = torch.tanh(y.narrow(2, 4, 1))
    cov = sigma_1 * sigma_2 * rho
    means = torch.cat((mu_1, mu_2), dim=2)
    cov_matrices = torch.cat((sigma_1 * sigma_1, cov, cov, sigma_2 * sigma_2), dim=2).view(y.size()[0], y.size()[1], 2, 2)
    return distributions.multivariate_normal.MultivariateNormal(means, cov_matrices)

net = DeepLSTM()

traj1 = torch.tensor([[0., 0.], [1., 2.], [3., 3.], [4., 3.]])
traj2 = torch.tensor([[0., 0.], [2., 2.], [3., 3.], [5., 2.]])
traj3 = torch.tensor([[0., 0.], [2., 1.], [3., 3.], [4., 2.]])
x = torch.stack([traj1, traj2, traj3], dim=0)

y = net.forward(x)
print(y)
print(y.shape)
print(y_to_dist(y))
