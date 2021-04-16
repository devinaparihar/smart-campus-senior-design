import torch
import pickle
import numpy as np
import pandas as pd

from torch import nn, optim
torch.autograd.set_detect_anomaly(True)

from disc_tp_data_preprocessing_utils import get_transitions_only, spaceid_to_one_hot, random_sample_data, sequence_train_test_split, split_trajectories, get_durations

GRAD_CLIP = 10
LOSS_GRAD_CLIP = 100

class DeepLSTM(nn.Module):


    def __init__(self, n_hidden_layers, hidden_state_size, input_dim, output_dim):
        super(DeepLSTM, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.hidden_state_size = hidden_state_size
        layers_list = []
        layers_list.append(nn.LSTMCell(input_dim, hidden_state_size))
        for i in range(n_hidden_layers - 1):
            layers_list.append(nn.LSTMCell(input_dim + hidden_state_size, hidden_state_size))
        self.layers = nn.ModuleList(layers_list)
        self.linear = nn.Linear(input_dim + hidden_state_size, output_dim)

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
        if CUDA:
            return outputs.index_select(1, torch.tensor([outputs.size()[1] - 1]).cuda()).squeeze(dim=1)
        return outputs.index_select(1, torch.tensor([outputs.size()[1] - 1])).squeeze(dim=1)


#def forward_pass_batch(net, x, loss):

SOFTMAX = nn.Softmax(dim=-1)
LOSS_CE = nn.CrossEntropyLoss()
def location_forward_pass_batch(net, x, n_locations):
    x_train = x.narrow(1, 0, x.size()[1] - 1)
    y_train = torch.argmax(x.narrow(1, 1, x.size()[1] - 1).narrow(2, 0, n_locations), dim=2)
    probs = SOFTMAX(net.forward(x_train))
    train_loss = LOSS_CE(probs.permute(0, 2, 1), y_train)
    if train_loss.requires_grad:
        train_loss.register_hook(lambda x: x.clamp(min=-1 * LOSS_GRAD_CLIP, max=LOSS_GRAD_CLIP))
    return train_loss


LOSS_MSE = torch.nn.MSELoss()
def time_forward_pass_batch(net, x, n_locations, train_length):
    x_train = x.narrow(1, 0, train_length)
    times = x.narrow(1, train_length - 1, 1).narrow(2, n_locations, 1).squeeze(dim=2)
    
    samples = []
    for y in times.cpu().numpy().ravel():
        samples.append([np.random.uniform(high=y)])
    samples = torch.tensor(samples)
    if CUDA:
        samples = samples.cuda()
    y_train = times - samples
    x_train[:, train_length - 1, n_locations] = samples[:, 0]
    outputs = net.forward_next_step(x_train)
    train_loss = LOSS_MSE(outputs, y_train)
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
# NEED TO DO SOMETHING ABOUT THESE NON FLOAT THINGS AT SOME POINT
df_sorted_transitions['DURATION_IN_SPACE_MINUTES'] = df_sorted_transitions['DURATION_IN_SPACE_SECONDS'].apply(lambda x: x/60 if type(x) is float else 1)
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
print(CUDA)
df = pd.read_csv("OneHot.csv")
#df = one_hot_df
N_SPLITS = 50
HIDDEN_STATE_SIZE = 100
N_LAYERS = 3

N_LOCATIONS = df.values.shape[1] - 3
INPUT_DIM = N_LOCATIONS + 1
LOCATION_LR = 0.01
TIME_LR = 0.005

test_steps = [1, 2, 3, 4, 9]
mean_accuracies = np.zeros((len(test_steps),))
mean_accuracies2 = np.zeros((len(test_steps),))
mean_accuracies3 = np.zeros((len(test_steps),))
mean_accuracies4 = np.zeros((len(test_steps),))
mean_errors = np.zeros((len(test_steps),))
mean_errors2 = np.zeros((len(test_steps),))
mean_errors3 = np.zeros((len(test_steps),))


for split in range(N_SPLITS):
    print("Split: {}".format(split + 1))
    train_df, test_df = sequence_train_test_split(df, TRAIN_LENGTH)

    location_network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, N_LOCATIONS)
    time_network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, 1)
    location_network2 = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, N_LOCATIONS)
    location_network3 = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, N_LOCATIONS)
    #time_network2 = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, 1)

    if CUDA:
        location_network = location_network.cuda()
        time_network = time_network.cuda()
        location_network2 = location_network2.cuda()
        #time_network2 = time_network2.cuda()
    location_optimizer = optim.Adam(location_network.parameters(), lr=LOCATION_LR)
    time_optimizer = optim.Adam(time_network.parameters(), lr=TIME_LR)
    location_optimizer2 = optim.Adam(location_network2.parameters(), lr=LOCATION_LR)
    location_optimizer3 = optim.Adam(location_network3.parameters(), lr=LOCATION_LR)
    #time_optimizer2 = optim.Adam(time_network2.parameters(), lr=TIME_LR)


    best_accuracies = [0 for i in range(len(test_steps))]
    best_accuracies2 = [0 for i in range(len(test_steps))]
    best_accuracies3 = [0 for i in range(len(test_steps))]
    best_accuracies4 = [0 for i in range(len(test_steps))]
    best_errors = [np.inf for i in range(len(test_steps))]
    best_errors2 = [np.inf for i in range(len(test_steps))]
    best_errors3 = [np.inf for i in range(len(test_steps))]


    for i in range(N_EPOCHS):
        #print("Epoch: {}".format(i))

        # Train the location prediction network
        location_optimizer.zero_grad()
        location_optimizer2.zero_grad()
        location_optimizer3.zero_grad()
        x = np.array(random_sample_data(train_df, TRAIN_LENGTH, N_PER_USER))
        x = torch.from_numpy(x).float()
        if CUDA:
            x=x.cuda()
        location_loss = location_forward_pass_batch(location_network, x, N_LOCATIONS)
        location_loss.backward()
        location_optimizer.step()
        location_loss2 = location_forward_pass_batch(location_network2, x, N_LOCATIONS)
        location_loss2.backward()
        location_optimizer2.step()

        location_loss3 = location_forward_pass_batch(location_network3, x, N_LOCATIONS)
        location_loss3.backward()
        location_optimizer3.step()
        #print("Location Loss: {}".format(location_loss.item()))
        #print("Location Loss2: {}".format(location_loss2.item()))

        # Train the time prediction network
        time_optimizer.zero_grad()
        #time_loss = torch.tensor(0).cuda().float()
        time_loss = torch.tensor(0).float()
        #time_optimizer2.zero_grad()
        #time_loss2 = torch.tensor(0).cuda().float()
        for j, n in enumerate(test_steps):
            x = np.array(random_sample_data(train_df, n + 1, TRAIN_LENGTH // (n + 1)))
            x = torch.from_numpy(x).float()
            if CUDA:
                x = x.cuda()
            time_loss += time_forward_pass_batch(time_network, x, N_LOCATIONS, n)
            #time_loss2 += time_forward_pass_batch(time_network2, x, N_LOCATIONS, n)
        time_loss.backward()
        time_optimizer.step()
        #time_loss2.backward()
        #time_optimizer2.step()
        #print("Time Loss: {}".format(time_loss.item()), flush=True)
        #print("Time Loss2: {}".format(time_loss2.item()), flush=True)

        with torch.no_grad():
            # Evaluate test location accuracy
            for j, n in enumerate(test_steps):
                data = np.stack(random_sample_data(test_df, n + 1, TRAIN_LENGTH // (n + 1)))
                x = torch.from_numpy(data[:, :-1]).float()
                if CUDA:
                    x=x.cuda()
                y = np.argmax(data[:, -1, :-1], axis=1)
                forward=location_network.forward_next_step(x).cpu().numpy()
                forward2=location_network2.forward_next_step(x).cpu().numpy()
                forward3=location_network3.forward_next_step(x).cpu().numpy()
                output=np.argmax(forward, axis=1)
                output2=np.argmax(forward2, axis=1)
                output3=np.argmax(forward3, axis=1)
                avg_prob = np.mean(np.array([forward, forward2, forward3]), axis=0)
                output4 = np.argmax(avg_prob, axis=1) #ensembled prediction
                #output = np.argmax(location_network.forward_next_step(x).numpy(), axis=1)
                total = y.shape[0]

                correct = np.sum(np.equal(y, output))
                correct2 = np.sum(np.equal(y,output2))
                correct3 = np.sum(np.equal(y, output3))
                correct4 = np.sum(np.equal(y, output4))
                accuracy = correct / total
                accuracy2 = correct2 / total
                accuracy3 = correct3 / total
                accuracy4 = correct4 / total

                if best_accuracies[j] < accuracy:
                    best_accuracies[j] = accuracy
                if best_accuracies2[j] < accuracy2:
                    best_accuracies2[j] = accuracy2
                if best_accuracies3[j] < accuracy3:
                    best_accuracies3[j] = accuracy3
                if best_accuracies4[j] < accuracy4:
                    best_accuracies4[j] = accuracy4
                    # print("{}-step Location Accuracy: {} | {}/{}".format(n, accuracy, correct, total))
                    # print("{}-step Location accuracy2: {} | {}/{}".format(n, accuracy2, correct2, total))
                    # print("{}-step Location Accuracy3: {} | {}/{}".format(n, accuracy3, correct3, total))


            # # Evaluate test time error
            # for j, n in enumerate(test_steps):
            #     data = np.stack(random_sample_data(test_df, n + 1, TRAIN_LENGTH // (n + 1)))
            #     x = torch.from_numpy(data[:, :-1]).float()
            #     if CUDA:
            #         x=x.cuda()
            #     times = x.narrow(1, n - 1, 1).narrow(2, N_LOCATIONS, 1).squeeze(dim=2)
            #     samples = []
            #     for y in times.cpu().numpy().ravel():
            #         samples.append([np.random.uniform(high=y)])
            #     samples = torch.tensor(samples)
            #     if CUDA:
            #         samples = samples.cuda()
            #     y = times - samples
            #     x[:, n - 1, N_LOCATIONS] = samples[:, 0]
            #     output_time = time_network.forward_next_step(x)
            #     error = LOSS_MSE(output_time, y).item()
            #     if best_errors[j] > error:
            #         best_errors[j] = error
            #         #print("{}-step Test Error: {}".format(n, error))
    print("Best Accuracies1: {}".format(best_accuracies))
    print("Best Accuracies2: {}".format(best_accuracies2))
    print("Best Accuracies3: {}".format(best_accuracies3))
    print("Best Accuracies (ensembled): {}".format(best_accuracies4))
    print("Test Steps: {}".format(test_steps))

    mean_accuracies += np.array(best_accuracies)
    mean_errors += np.array(best_errors)
    mean_accuracies2 += np.array(best_accuracies2)
    mean_errors2 += np.array(best_errors2)
    mean_accuracies3 += np.array(best_accuracies3)
    mean_accuracies4 += np.array(best_accuracies4)
    
    print("Mean Accuracies1: {}".format(mean_accuracies / (split + 1)))
    #print("Mean Errors1: {}".format(mean_errors / (split + 1)))
    print("Mean Accuracies2: {}".format(mean_accuracies2 / (split + 1)))
    #print("Mean Errors2: {}".format(mean_errors2 / (split + 1)))
    print("Mean Accuracies3: {}".format(mean_accuracies3 / (split + 1)))
    #print("Mean Errors2: {}".format(mean_errors2 / (split + 1)))
    print("Mean Accuracies4 (ensembled): {}".format(mean_accuracies4 / (split + 1)))



