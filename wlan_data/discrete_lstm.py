import torch
import pickle
import numpy as np
import pandas as pd

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
                if CUDA:
                           x_i_t.cuda() 
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
        if CUDA:
            return outputs.index_select(1, torch.tensor([outputs.size()[1] - 1]).cuda()).squeeze(dim=1)
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


df = pd.read_csv("TrainingData.csv")
df_sorted = df.sort_values(by=['TIMESTAMP', 'USERID'])
df_sorted_transitions = get_transitions_only(df_sorted)

one_hot_df, int_to_id, id_to_int = spaceid_to_one_hot(df_sorted_transitions)

one_hot_df.to_csv("OneHot.csv", index=False)
with open('int_to_id.pickle', 'wb') as handle:
    pickle.dump(int_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('id_to_int.pickle', 'wb') as handle:
    pickle.dump(id_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

N_EPOCHS = 500
TRAIN_LENGTH = 10
N_PER_USER = 5
CUDA = torch.cuda.is_available()
print(CUDA)

df = pd.read_csv("OneHot.csv")

N_SPLITS = 50
HIDDEN_STATE_SIZE = 100
N_LAYERS = 3
DIM = df.values.shape[1] - 2
LR = 0.01

test_steps = [1, 2, 3, 4, 9]
means = np.zeros((len(test_steps),))
means2 = np.zeros((len(test_steps),))
means3 = np.zeros((len(test_steps),))


for split in range(N_SPLITS):
    print("Split: {}".format(split + 1))
    train_df, test_df = sequence_train_test_split(df, TRAIN_LENGTH)

    network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, DIM)
    network2 = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, DIM)
    if CUDA:
        network = network.cuda()
        network2 = network2.cuda()
    optimizer = optim.Adam(network.parameters(), lr=LR)
    optimizer2 = optim.Adam(network2.parameters(), lr=LR)
    loss = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    best_accuracies = [0 for i in range(len(test_steps))]
    best_accuracies2 = [0 for i in range(len(test_steps))]
    best_accuracies3 = [0 for i in range(len(test_steps))]

    for i in range (N_EPOCHS):
        #print("Epoch: {}".format(i))
        optimizer.zero_grad()
        optimizer2.zero_grad()
        x = np.array(random_sample_data(train_df, TRAIN_LENGTH, N_PER_USER))
        if CUDA:
            x = torch.from_numpy(x).float().cuda()
        train_loss = forward_pass_batch(network, x, loss)
        train_loss.backward()
        train_loss2 = forward_pass_batch(network2, x, loss2)
        train_loss2.backward()
#        print("Training Loss: {}".format(train_loss.item()))
        optimizer.step()
        optimizer2.step()
        with torch.no_grad():
            for j, n in enumerate(test_steps):
                data = np.stack(random_sample_data(test_df, n + 1, TRAIN_LENGTH // (n + 1)))
                x = torch.from_numpy(data[:, :-1]).float()
                if CUDA:
                    x=x.cuda()
                y = np.argmax(data[:, -1], axis=1)
                forward1 = network.forward_next_step(x).cpu().numpy()
                forward2 = network2.forward_next_step(x).cpu().numpy()
                output = np.argmax(forward1,axis=1)
                output2 = np.argmax(forward2,axis=1)
                inside = np.mean(np.array([forward1, forward2]), axis=0)
                output3=np.argmax(inside, axis=1)
                total = y.shape[0]
                correct = np.sum(np.equal(y, output))
                correct2 = np.sum(np.equal(y,output2))
                correct3 = np.sum(np.equal(y, output3))
                accuracy = correct / total
                accuracy2 = correct2 / total
                accuracy3 = correct3 / total
                if best_accuracies[j] < accuracy:
                    best_accuracies[j] = accuracy
#                   print("{}-step Accuracy: {} | {}/{}".format(n, accuracy, correct, total))
                if best_accuracies2[j] < accuracy2:
                    best_accuracies2[j] = accuracy2
                if best_accuracies3[j] < accuracy3:
                    best_accuracies3[j] = accuracy3

    print("Best Accuracies: {}".format(best_accuracies))
    print("Test Steps: {}".format(test_steps))
    print("Best Accuracies: {}".format(best_accuracies2))
    print("Test Steps: {}".format(test_steps))
    print("Best Accuracies (ensembled): {}".format(best_accuracies3))
    print("Test Steps: {}".format(test_steps))

    means += np.array(best_accuracies)
    means2 += np.array(best_accuracies2)
    means3 += np.array(best_accuracies3)

print("Mean Accuracies: {}".format(means / (split + 1)))
print("Mean Accuracies2: {}".format(means2 / (split + 1)))
print("Mean Accuracies3 (ensembled): {}".format(means3 / (split + 1)))
