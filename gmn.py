import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

INPUT_SIZE = 2
HIDDEN_STATE_SIZE = 16
OUTPUT_SIZE = 5

GRAD_CLIP = 10

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
            h_1_t, c_1_t = self.lstm1(x_1_t, (h_1_tm1, c_1_tm1))

            if h_1_t.requires_grad:
                h_1_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
            if c_1_t.requires_grad:
                c_1_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))

            x_2_t = torch.cat((x_t, h_1_t), dim=1)
            h_2_t, c_2_t = self.lstm2(x_2_t, (h_2_tm1, c_2_tm1))

            if h_2_t.requires_grad:
                h_2_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
            if c_2_t.requires_grad:
                c_2_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))

            x_3_t = torch.cat((x_t, h_2_t), dim=1)
            y_t = self.l1(x_3_t)

            y_list.append(y_t)
            h_1_tm1 = h_1_t
            c_1_tm1 = c_1_t
            h_2_tm1 = h_2_t
            c_2_tm1 = c_2_t
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

def y_to_params(y):
    mu_1 = y.narrow(2, 0, 1)
    mu_2 = y.narrow(2, 1, 1)
    sigma_1 = torch.exp(y.narrow(2, 2, 1))
    sigma_1 = torch.clamp(sigma_1, 0.01, 100)
    sigma_2 = torch.exp(y.narrow(2, 3, 1))
    sigma_2 = torch.clamp(sigma_2, 0.01, 100)
    rho = torch.clamp(torch.tanh(y.narrow(2, 4, 1)), -0.99, 0.99)
    cov = sigma_1 * sigma_2 * rho
    means = torch.cat((mu_1, mu_2), dim=2)
    cov_matrices = torch.cat((sigma_1 * sigma_1, cov, cov, sigma_2 * sigma_2), dim=2).view(y.size()[0], y.size()[1], 2, 2)
    return means, cov_matrices

traj1 = torch.tensor([[0., 0.], [1., 2.], [3., 3.], [4., 3.]])
traj2 = torch.tensor([[0., 0.], [2., 2.], [3., 3.], [5., 2.]])
traj3 = torch.tensor([[0., 0.], [2., 1.], [3., 3.], [4., 2.]])
x = torch.stack([traj1, traj2, traj3], dim=0)
y_true = x.narrow(1, 1, x.size()[1] - 1)
x_train = x.narrow(1, 0, x.size()[1] - 1)

min_loss = np.inf
min_means = None
min_cov = None
for state in range(1000):
    net = DeepLSTM()
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print(state)
    for i in range(1000):
        optimizer.zero_grad()
        y = net.forward(x_train)
        means, cov_matrices = y_to_params(y)
        dist = distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=cov_matrices)
        loss = -1 * torch.sum(dist.log_prob(y_true))
        if loss.item() < min_loss:
            min_loss = loss.item()
            min_means = means
            min_cov = cov_matrices
        loss.backward()
        optimizer.step()
    print(min_loss)
    print(min_means)
    print(min_cov)
