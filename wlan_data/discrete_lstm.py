import torch
import pickle
import numpy as np
import pandas as pd

from torch import nn, optim

from disc_tp_data_preprocessing_utils import get_transitions_only, spaceid_to_one_hot, random_sample_data, sequence_train_test_split, split_trajectories, get_durations

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
            outputs = self.linear(x_n_t)

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

SOFTMAX = nn.Softmax(dim=-1)
def forward_pass_batch(net, x, lossCE, num_CE, lossMSE, NUM_MSE):
    x_train = x.narrow(1, 0, x.size()[1] - 1)
    y_train = torch.argmax(x.narrow(1, 1, x.size()[1] - 1).narrow(2, 0, num_CE), dim=2)
    probs = SOFTMAX(net.forward(x_train).narrow(2, 0, num_CE))
    train_loss = lossCE(probs.permute(0, 2, 1), y_train)
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

TIMEGAP_THRESH = 3600
NUM_USERS = 19

df = pd.read_csv("TrainingData.csv")
df_sorted = df.sort_values(by=['TIMESTAMP', 'USERID'])
df_sorted_transitions = split_trajectories(get_transitions_only(df_sorted), TIMEGAP_THRESH, NUM_USERS)
one_hot_df, int_to_id, id_to_int = spaceid_to_one_hot(df_sorted_transitions)

# get duration in space
df_sorted_transitions = get_durations(df_sorted_transitions)
df_sorted_transitions['DURATION_IN_SPACE_MINUTES'] = df_sorted_transitions['DURATION_IN_SPACE_SECONDS'].apply(lambda x: x/60 if type(x) is float else -1)

# append duration as feature to one hot encoded df
one_hot_df['STAYING_TIME'] = df_sorted_transitions['DURATION_IN_SPACE_MINUTES'].tolist()

one_hot_df.to_csv("OneHot.csv", index=False)
with open('int_to_id.pickle', 'wb') as handle:
    pickle.dump(int_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('id_to_int.pickle', 'wb') as handle:
    pickle.dump(id_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

N_EPOCHS = 500
TRAIN_LENGTH = 10
N_PER_USER = 2

CUDA = torch.cuda.is_available()

df = pd.read_csv("OneHot.csv")
#df = one_hot_df

N_SPLITS = 50
HIDDEN_STATE_SIZE = 100
N_LAYERS = 3
DIM = df.values.shape[1] - 2
N_LOCATIONS = DIM - 1
LR = 0.01

test_steps = [1, 2, 3, 4, 9]
means = np.zeros((len(test_steps),))

for split in range(N_SPLITS):
    print("Split: {}".format(split + 1))
    train_df, test_df = sequence_train_test_split(df, TRAIN_LENGTH)

    network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, DIM)
    if CUDA:
        network = network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=LR)
    loss1 = nn.CrossEntropyLoss()
    loss2 = torch.nn.MSELoss()

    best_accuracies = [0 for i in range(len(test_steps))]

    for i in range(N_EPOCHS):
        print("Epoch: {}".format(i))
        optimizer.zero_grad()
        x = np.array(random_sample_data(train_df, TRAIN_LENGTH, N_PER_USER))
        x = torch.from_numpy(x).float()
        train_loss_1 = forward_pass_batch(network, x, loss1, N_LOCATIONS, loss2, 1)
        train_loss_1.backward()
        print("Training Loss: {}".format(train_loss_1.item()))
        optimizer.step()
        with torch.no_grad():
            for j, n in enumerate(test_steps):
                data = np.stack(random_sample_data(test_df, n + 1, TRAIN_LENGTH // (n + 1)))
                x = torch.from_numpy(data[:, :-1]).float()
                y = np.argmax(data[:, -1], axis=1)
                output = np.argmax(network.forward_next_step(x).numpy(), axis=1)
                total = y.shape[0]
                correct = np.sum(np.equal(y, output))
                accuracy = correct / total
                if best_accuracies[j] < accuracy:
                    best_accuracies[j] = accuracy
                    #print("{}-step Accuracy: {} | {}/{}".format(n, accuracy, correct, total))
    print("Best Accuracies: {}".format(best_accuracies))
    print("Test Steps: {}".format(test_steps))

    means += np.array(best_accuracies)
    print("Mean Accuracies: {}".format(means / (split + 1)))
