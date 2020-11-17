import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.distributions as distributions

INPUT_SIZE = 2
HIDDEN_STATE_SIZE = 256
OUTPUT_SIZE = 5

GRAD_CLIP = 10
LOSS_GRAD_CLIP = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

CUDA = False #cuda.is_available()

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

def numpy_data_to_pytorch(l_in):
    max_len = max([x.shape[0] for x in l_in])
    data_l = []
    mask_l = []
    for x in l_in:
        pytorch_x = torch.from_numpy(x).float()
        n = x.shape[0]
        k = max_len - n
        mask = torch.ones((n,))
        if k > 0:
            mask = torch.cat([mask, torch.zeros((k,))], dim=0)
            pytorch_x = torch.cat([pytorch_x, torch.zeros((k, 2))], dim=0)
        if CUDA:
            mask = mask.cuda()
            pytorch_x = pytorch_x.cuda()
        data_l.append(pytorch_x)
        mask_l.append(mask)
    return data_l, mask_l

def sample(data_l, mask_l, size):
    indices = np.random.choice(len(data_l), size=size, replace=False)
    data_stack = [data_l[i] for i in indices]
    mask_stack = [mask_l[i] for i in indices]
    return torch.stack(data_stack), torch.stack(mask_stack)

class DeepLSTM(nn.Module):

    def __init__(self):
        super(DeepLSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(INPUT_SIZE, HIDDEN_STATE_SIZE)
        self.r1 = nn.ReLU()
        self.lstm2 = nn.LSTMCell(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE)
        self.r2 = nn.ReLU()
        self.l1 = nn.Linear(INPUT_SIZE + HIDDEN_STATE_SIZE, OUTPUT_SIZE)

    def step(self, x_t, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1):
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

            return y_t, h_1_t, c_1_t, h_2_t, c_2_t

    def init_hidden(self):
        h_1 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        c_1 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        h_2 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        c_2 = torch.zeros((x.size()[0], HIDDEN_STATE_SIZE))
        if CUDA:
            h_1 = h_1.cuda()
            c_1 = c_1.cuda()
            h_2 = h_2.cuda()
            c_2 = c_2.cuda()

        return h_1, c_1, h_2, c_2

    def forward(self, x):
        h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1 = self.init_hidden()
        y_list = []

        for i in range(x.size()[1]):
            x_t = torch.squeeze(x.narrow(1, i, 1), 1)
            y_t, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1 = self.step(x_t, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1)
            y_list.append(y_t)

        return torch.stack(y_list, dim=1)

    def generate(self, x, steps):
        h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1 = self.init_hidden()
        y_tm1 = None

        for i in range(x.size()[1]):
            x_t = torch.squeeze(x.narrow(1, i, 1), 1)
            y_tm1, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1 = self.step(x_t, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1)
            means, cov_matrices = y_to_params(torch.stack([y_tm1], dim=1))

        new_points = []
        for i in range(steps):
            means, cov_matrices = y_to_params(torch.stack([y_tm1], dim=1))
            print("Mean: {}".format(means))
            print("Covariance: {}".format(means))
#            dist = distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=cov_matrices)
#            x_t = dist.sample().squeeze(dim=1)
            x_t = means.squeeze(dim=0)
            new_points.append(x_t)
            y_tm1, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1 = self.step(x_t, h_1_tm1, c_1_tm1, h_2_tm1, c_2_tm1)

        return torch.stack(new_points, dim=1)

def forward_pass_batch(x, mask):
    x_train = x.narrow(1, 0, x.size()[1] - 1)
    y_train = x.narrow(1, 1, x.size()[1] - 1)
    mask_train = mask.narrow(1, 1, mask.size()[1] - 1)
    output = net.forward(x_train)
    means, cov_matrices = y_to_params(output)
    dist = distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=cov_matrices)
    loss = -1 * torch.sum(dist.log_prob(y_train) * mask_train) / y_train.size()[0]
    if loss.requires_grad:
        loss.register_hook(lambda x: x.clamp(min=-1 * LOSS_GRAD_CLIP, max=LOSS_GRAD_CLIP))
    return loss, means, y_train, mask
