import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.distributions as distributions

INPUT_SIZE = 2
OUTPUT_SIZE = 5

GRAD_CLIP = 10
LOSS_GRAD_CLIP = 100

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

def params_to_mixture_model(g_params, weights, components):
    means_list = []
    cov_list = []
    for i in range(components):
        y_i = g_params.narrow(2, i * OUTPUT_SIZE, OUTPUT_SIZE)
        mean, cov = y_to_params(y_i)
        means_list.append(mean)
        cov_list.append(cov)
    means = torch.stack(means_list, dim=2)
    covariance_matrices = torch.stack(cov_list, dim=2)
    dist = distributions.multivariate_normal.MultivariateNormal(means, covariance_matrix=covariance_matrices)
    categorical = distributions.categorical.Categorical(probs=weights)
    mixture = distributions.mixture_same_family.MixtureSameFamily(categorical, dist)
    return mixture, means

class DeepLSTM(nn.Module):

    def __init__(self, n_hidden_layers, hidden_state_size, n_mixture_components):
        super(DeepLSTM, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_mixture_components = n_mixture_components
        self.hidden_state_size = hidden_state_size
        layers_list = []
        layers_list.append(nn.LSTMCell(INPUT_SIZE, hidden_state_size))
        for i in range(n_hidden_layers - 1):
            layers_list.append(nn.LSTMCell(INPUT_SIZE + hidden_state_size, hidden_state_size))
        self.layers = nn.ModuleList(layers_list)
        self.gaussian_params_linear = nn.Linear(INPUT_SIZE + hidden_state_size, OUTPUT_SIZE * n_mixture_components)
        self.weights_linear = nn.Linear(INPUT_SIZE + hidden_state_size, n_mixture_components)
        self.softmax = nn.Softmax(dim=-1)

    def step(self, x_t, h_tm1, c_tm1):
            x_1_t = x_t
            h_im1_t, c_im1_t = self.layers[0](x_1_t, (h_tm1[0], c_tm1[0]))

            if h_im1_t.requires_grad:
                h_im1_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
            if c_im1_t.requires_grad:
                c_im1_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))

            h_t = [h_im1_t]
            c_t = [c_im1_t]
            for i in range(self.n_hidden_layers - 1):
                x_i_t = torch.cat((x_t, h_im1_t), dim=1)
                h_i_t, c_i_t = self.layers[i + 1](x_i_t, (h_tm1[i + 1], c_tm1[i + 1]))
                h_t.append(h_i_t)
                c_t.append(c_i_t)
                if h_i_t.requires_grad:
                    h_i_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
                if c_i_t.requires_grad:
                    c_i_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
                h_im1_t = h_i_t
                c_im1_t = c_i_t

            x_n_t = torch.cat((x_t, h_im1_t), dim=1)
            g_params = self.gaussian_params_linear(x_n_t)
            weights = self.softmax(self.weights_linear(x_n_t))

            return g_params, weights, h_t, c_t

    def init_hidden(self, x):
        h_0 = []
        c_0 = []
        for i in range(self.n_hidden_layers):
            h_i_0 = torch.zeros((x.size()[0], self.hidden_state_size))
            c_i_0 = torch.zeros((x.size()[0], self.hidden_state_size))
            if CUDA:
                h_i_0 = h_i_0.cuda()
                c_i_0 = c_i_0.cuda()
            h_0.append(h_i_0)
            c_0.append(c_i_0)
        return h_0, c_0

    def forward(self, x):
        h_tm1, c_tm1 = self.init_hidden(x)
        g_param_list = []
        weights_list = []

        for i in range(x.size()[1]):
            x_t = torch.squeeze(x.narrow(1, i, 1), 1)
            g_params_t, weights_t, h_tm1, c_tm1 = self.step(x_t, h_tm1, c_tm1)
            g_param_list.append(g_params_t)
            weights_list.append(weights_t)

        return torch.stack(g_param_list, dim=1), torch.stack(weights_list, dim=1)

    def generate(self, x, steps):
        h_tm1, c_tm1 = self.init_hidden(x)
        g_params_tm1 = None
        weights_tm1 = None

        for i in range(x.size()[1]):
            x_t = torch.squeeze(x.narrow(1, i, 1), 1)
            g_params_tm1, weights_tm1, h_tm1, c_tm1 = self.step(x_t, h_tm1, c_tm1)

        new_points = []
        for i in range(steps):
            stacked_g_params = torch.stack([g_params_tm1], dim=1)
            stacked_weights = torch.stack([weights_tm1], dim=1)
            dist, _ = params_to_mixture_model(stacked_g_params, stacked_weights, self.n_mixture_components)
            x_t = dist.sample().squeeze(dim=0)
            new_points.append(x_t)
            g_params_tm1, weights_tm1, h_tm1, c_tm1 = self.step(x_t, h_tm1, c_tm1)

        return torch.stack(new_points, dim=1)

def forward_pass_batch(net, x, mask, components):
    x_train = x.narrow(1, 0, x.size()[1] - 1)
    y_train = x.narrow(1, 1, x.size()[1] - 1)
    mask_train = mask.narrow(1, 1, mask.size()[1] - 1)
    g_params, weights = net.forward(x_train)
    dist, means = params_to_mixture_model(g_params, weights, components)
    loss = -1 * torch.sum(dist.log_prob(y_train) * mask_train) / y_train.size()[0]
    if loss.requires_grad:
        loss.register_hook(lambda x: x.clamp(min=-1 * LOSS_GRAD_CLIP, max=LOSS_GRAD_CLIP))
    return loss, means, y_train, mask
