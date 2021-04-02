import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn, optim

from disc_tp_data_preprocessing_utils import get_transitions_only, spaceid_to_one_hot, random_sample_data, sequence_train_test_split

GRAD_CLIP = 10
LOSS_GRAD_CLIP = 100

class DeepLSTM(nn.Module):

    def __init__(self, n_hidden_layers, hidden_state_size, dim):
        super(DeepLSTM, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.dim = dim
        self.hidden_state_size = hidden_state_size
        layers_list = []
        layers_list.append(nn.LSTMCell(dim, hidden_state_size))
        for i in range(n_hidden_layers - 1):
            layers_list.append(nn.LSTMCell(dim + hidden_state_size, hidden_state_size))
        self.layers = nn.ModuleList(layers_list)
        self.linear = nn.Linear(dim + hidden_state_size, dim)
        self.softmax = nn.Softmax(dim=-1)

    def step(self, x_t, h_tm1, c_tm1, final):
            x_1_t = x_t
            h_im1_t, c_im1_t = self.layers[0](x_1_t, (h_tm1[0], c_tm1[0]))
            if h_im1_t.requires_grad:
                h_im1_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
            if c_im1_t.requires_grad and not final:
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
                if c_i_t.requires_grad and not final:
                    c_i_t.register_hook(lambda x: x.clamp(min=-1 * GRAD_CLIP, max=GRAD_CLIP))
                h_im1_t = h_i_t
                c_im1_t = c_i_t

            x_n_t = torch.cat((x_t, h_im1_t), dim=1)
            outputs = self.softmax(self.linear(x_n_t))

            return outputs, h_t, c_t

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
        output_list = []

        for i in range(x.size()[1]):
            x_t = torch.squeeze(x.narrow(1, i, 1), 1)
            output_t, h_tm1, c_tm1 = self.step(x_t, h_tm1, c_tm1, i == x.size()[1] - 1)
            output_list.append(output_t)

        return torch.stack(output_list, dim=1)

    def forward_next_step(self, x):
        outputs = self.forward(x)
        return outputs.index_select(1, torch.tensor([outputs.size()[1] - 1])).squeeze(dim=1)

def forward_pass_batch(net, x, loss):
    x_train = x.narrow(1, 0, x.size()[1] - 1)
    y_train = torch.argmax(x.narrow(1, 1, x.size()[1] - 1), dim=2)
    probs = net.forward(x_train)
    train_loss = loss(probs.permute(0, 2, 1), y_train)
    if train_loss.requires_grad:
        train_loss.register_hook(lambda x: x.clamp(min=-1 * LOSS_GRAD_CLIP, max=LOSS_GRAD_CLIP))
    return train_loss

def numpy_data_to_pytorch(data):
    output_list = []
    for x in data:
        pytorch_x = torch.from_numpy(x).float()
        if CUDA:
            pytorch_x = pytorch_x.cuda()
        output_list.append(pytorch_x)
    return output_list

IMAGE_DIR = './images/'

# Create Graph
df = pd.read_csv("TrainingData.csv")
graph = nx.Graph()

for user in range(1, df['USERID'].max() + 1):
    user_df = df[df['USERID'] == user].sort_values('TIMESTAMP')
    spaces = user_df['SPACEID'].to_numpy().ravel()

    condition = spaces[0]
    for i in range(1, spaces.shape[0] - 1):
        if spaces[i] > 30 and spaces[i] != 147 and spaces[i + 1] != 147 and spaces[i] != condition and spaces[i] != spaces[i + 1]:
            graph.add_edge(spaces[i], spaces[i + 1])
            condition = spaces[i]

pos = nx.spring_layout(graph, k=1/2)
nx.draw(graph, pos=pos, node_size=100)
plt.savefig(IMAGE_DIR + "original.png")

N_EPOCHS = 500
TRAIN_LENGTH = 10
N_PER_USER = 5
CUDA = torch.cuda.is_available()

df = pd.read_csv("OneHot.csv")

N_SPLITS = 50
HIDDEN_STATE_SIZE = 100
N_LAYERS = 3
DIM = df.values.shape[1] - 2
LR = 0.01

train_df, test_df = sequence_train_test_split(df, TRAIN_LENGTH)

network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, DIM)
if CUDA:
    network = network.cuda()
optimizer = optim.Adam(network.parameters(), lr=LR)
loss = nn.CrossEntropyLoss()

for i in range(N_EPOCHS):
    print("Epoch: {}".format(i))
    optimizer.zero_grad()
    x = np.array(random_sample_data(train_df, TRAIN_LENGTH, N_PER_USER))
    x = torch.from_numpy(x).float()
    train_loss = forward_pass_batch(network, x, loss)
    train_loss.backward()
    print("Training Loss: {}".format(train_loss.item()))
    optimizer.step()

VIS_INIT_LEN = 7
VIS_FINAL_LEN = 10
N_EXAMPLES = 10

for example in range(N_EXAMPLES):
    rs = random_sample_data(test_df, VIS_INIT_LEN, 1)
    data = np.stack([rs[np.random.choice(len(rs))]])

    coeffs = [1]
    NUM_EXPAND = 3
    EXPAND_DEPTH = VIS_FINAL_LEN - VIS_INIT_LEN
    x = torch.from_numpy(data).float()

    with torch.no_grad():
        for i in range(EXPAND_DEPTH):
            print(coeffs)
            inpu = x.numpy()
            print(inpu.shape)
            output = network.forward_next_step(x).numpy()
            new_coeffs = []
            new_inputs = []
            for j, y in enumerate(output.tolist()):
                ind = np.argpartition(y, -1 * NUM_EXPAND)[-1 * NUM_EXPAND:]
                for k, n in enumerate(ind):
                    new_coeffs.append(coeffs[j] * y[n])
                    new_point = np.zeros(inpu[j][0].shape)
                    new_point[n] = 1
#                print(inpu[j])
#                print(np.array([new_point]))
                    new_inputs.append(np.concatenate([inpu[j], np.array([new_point])], axis=0))

            coeffs = new_coeffs
            x = torch.from_numpy(np.array(new_inputs)).float()

    int_to_id = pickle.load(open('int_to_id.pickle', 'rb'))

    trajectories = x.numpy().argmax(axis=2)
    total = np.sum(coeffs)
    print(total)
    nodes_list = graph.nodes()
    node_to_ind = {node: i for i, node in enumerate(nodes_list)}
    for i in range(VIS_FINAL_LEN):
        dist = trajectories[:, i]
        print("Time {}".format(i))
        color_params = [0 for i in range(len(nodes_list))]
        for j, v in enumerate(dist):
            color_params[node_to_ind[int_to_id[v]]] += coeffs[j] / total
        colors = [(np.clip(2 * (1 - x), 0, 1), np.clip(2 * x, 0, 1), 0) if x != 0 else (0, 0, 0) for x in color_params]

        nx.draw(graph, pos=pos, nodelist=nodes_list, node_color=colors, node_size=100)
        plt.savefig(IMAGE_DIR + "{:02d}_{:02d}.png".format(example, i))
