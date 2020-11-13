import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.distributions as distributions

INPUT_SIZE = 2
HIDDEN_STATE_SIZE = 32
OUTPUT_SIZE = 5

GRAD_CLIP = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.05

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
        mask = torch.cat([torch.zeros((5,)), torch.ones((n - 5,))], dim=0)
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
        self.lstm2 = nn.LSTMCell(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE)
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

        new_points = []
        for i in range(steps):
            means, cov_matrices = y_to_params(torch.stack([y_tm1], dim=1))
            print("Means:")
            print(means)
            print("Covariances:")
            print(cov_matrices)
            dist = distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=cov_matrices)
            x_t = dist.sample().squeeze(dim=1)
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
    loss = -1 * torch.sum(dist.log_prob(y_train) * mask_train)
    return loss


"""
traj1 = torch.tensor([[0., 0.], [1., 2.], [3., 3.], [4., 3.]])
traj2 = torch.tensor([[0., 0.], [2., 2.], [3., 3.], [5., 2.]])
traj3 = torch.tensor([[0., 0.], [2., 1.], [3., 3.], [4., 2.]])
x = torch.stack([traj1, traj2, traj3], dim=0)
y_true = x.narrow(1, 1, x.size()[1] - 1)
x_train = x.narrow(1, 0, x.size()[1] - 1)
"""

train_data, train_mask = numpy_data_to_pytorch(np.load("circles_train.npy", allow_pickle=True))
test_data, test_mask= numpy_data_to_pytorch(np.load("circles_test.npy", allow_pickle=True))
val_data, val_mask = numpy_data_to_pytorch(np.load("circles_validate.npy", allow_pickle=True))

min_loss = np.inf
min_means = None
min_cov = None
for state in range(100):
    print("Model: {}".format(state))
    net = DeepLSTM()
    if CUDA:
        net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    x, mask = sample(train_data, train_mask, BATCH_SIZE)

    for i in range(5000):
        print("Epoch: {}".format(i))
        optimizer.zero_grad()
        train_loss = forward_pass_batch(x, mask)
        train_loss.backward()
        optimizer.step()
        print("Training Loss: {}".format(train_loss.item()))
#        if i % 100 == 0:
        with torch.no_grad():
#                x, mask = sample(test_data, test_mask, BATCH_SIZE)
            test_loss = forward_pass_batch(x, mask)
            if test_loss.item() < min_loss:
                min_loss = test_loss.cpu().item()
#                x, mask = sample(val_data, val_mask, 1)
                np_mask = mask.cpu().numpy()
                n = max([i + 1 if np_mask[0, i] == 1 else 0 for i in range(np_mask.shape[1])])
                k = np.random.randint(1, n - 5)
                x_cut = x.narrow(1, 0, n - k)
                print("Input:\n{}".format(x_cut))
                print("Generated Output:\n{}".format(net.generate(x, k)))
                print("True Output:\n{}".format(x.narrow(1, n - k, k)))
                print("Epoch: {}".format(i))
                print("Training Loss: {}".format(train_loss.item()))
                print("Test Loss: {}".format(test_loss.item()))

        print("Min test loss: {}".format(min_loss))
