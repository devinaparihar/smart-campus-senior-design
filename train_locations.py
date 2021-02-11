import numpy as np
import torch
import torch.cuda as cuda
import matplotlib.pyplot as plt

from load import load_cleaned_individuals
from gmn import *

CUDA = cuda.is_available()
SCALE_FACTOR = 1
def normalize_data(data, lat_index=1, long_index=2):
    latitudes = []
    longitudes = []
    for x in data:
        latitudes.append(x[:, lat_index].ravel())
        longitudes.append(x[:, long_index].ravel())
    latitudes = np.concatenate(latitudes)
    longitudes = np.concatenate(longitudes)

    lat_mean = latitudes.mean()
    lat_std = latitudes.std()
    long_mean = longitudes.mean()
    long_std = longitudes.std()
    normalized_data = []
    for x in data:
        lat_vec = (x[:, lat_index] - lat_mean) / lat_std * SCALE_FACTOR
        long_vec = (x[:, long_index] - long_mean) / long_std * SCALE_FACTOR
        norm_x = np.transpose(np.stack([lat_vec, long_vec]))
        normalized_data.append(norm_x)

    return normalized_data, lat_mean, lat_std, long_mean, long_std

def numpy_data_to_pytorch(data):
    output_list = []
    for x in data:
        pytorch_x = torch.from_numpy(x).float()
        if CUDA:
            pytorch_x = pytorch_x.cuda()
        output_list.append(pytorch_x)
    return output_list

def sample_data(data, length, batch_size):
    stack_list = []
    indices = np.random.choice(len(data), size=batch_size, replace=False)
    for i in indices:
        x = data[i]
        x_len = x.size()[0]
        start = np.random.randint(0, x_len - length + 1)
        stack_list.append(x.narrow(0, start, length))
    return torch.stack(stack_list)

def pytorch_to_scaled_numpy(pt_x, lat_mean, lat_std, long_mean, long_std, lat_index=0, long_index=1):
    x = pt_x.cpu().numpy()[0]
    lat_vec = x[:, lat_index] * lat_std / SCALE_FACTOR + lat_mean
    long_vec = x[:, long_index] * long_std / SCALE_FACTOR + long_mean
    return np.transpose(np.stack([lat_vec, long_vec]))

DIR = "sample_trajectories/"
def save_trajectory(provided, generated, model, epoch, lr, test_error):
    name = "{:04d}_{:04d}_{}_{}.npy".format(model, epoch, lr, test_error)
    np.save(DIR + name, (provided, generated))

LENGTH_MIN = 200
TEST_SET_RATIO = 0.05
TRAIN_LENGTH = LENGTH_MIN
MASK_NUMBER = 175

location_data = [x for x in load_cleaned_individuals("./cleaned-new-final") if x.shape[0] >= LENGTH_MIN]
print("Location data loaded")
normalized_data, lat_mean, lat_std, long_mean, long_std = normalize_data(location_data)
print("Location data normalized")
print("Latitude Mean: {}\nLatitude Standard Deviation: {}\nLongitude Mean: {}\nLongitude Standard Deviation: {}".format(lat_mean, lat_std, long_mean, long_std))

test_set_start = int(round(len(normalized_data) * (1 - TEST_SET_RATIO)))
train_data = numpy_data_to_pytorch(normalized_data[:test_set_start])
test_data = numpy_data_to_pytorch(normalized_data[test_set_start:])
print("Training set size: {}\nTest set size: {}".format(len(train_data), len(test_data)))


BATCH_SIZE = 64
HIDDEN_STATE_SIZE = 400
N_LAYERS = 3
N_COMPONENTS = 20
N_GENERATE = 20

LR_UP = 0.05
LR_LO = 0.00005

mask = torch.cat((torch.zeros((BATCH_SIZE, MASK_NUMBER)), torch.ones((BATCH_SIZE, TRAIN_LENGTH - MASK_NUMBER))), dim=1)
if CUDA:
    mask = mask.cuda()

print("Beginning training")

for state in range(100):
    print("Model: {}".format(state))
    net = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, N_COMPONENTS)
    lr = np.exp(np.random.uniform(low=np.log(LR_LO), high=np.log(LR_UP)))

    if CUDA:
        net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    min_loss = np.inf

    for i in range(500):
        print("Epoch: {}".format(i))
        optimizer.zero_grad()
        x = sample_data(train_data, TRAIN_LENGTH, BATCH_SIZE)
        train_loss, _, _, _ = forward_pass_batch(net, x, mask, N_COMPONENTS)
        train_loss.backward()
        optimizer.step()
        print("Training Loss: {}".format(train_loss.item()))
        with torch.no_grad():
            x = sample_data(test_data, TRAIN_LENGTH, BATCH_SIZE)
            test_loss, _, _, _ = forward_pass_batch(net, x, mask, N_COMPONENTS)
            print("Test Loss: {}".format(test_loss.item()))
            if test_loss.cpu().item() < min_loss:
                min_loss = test_loss.cpu().item()
                x = sample_data(test_data, TRAIN_LENGTH, 1)
                n = x.size()[1]
                k = N_GENERATE
                x_cut = x.narrow(1, 0, n - k)
                x_gen = net.generate(x_cut, k)
                np_x_cut = pytorch_to_scaled_numpy(x_cut, lat_mean, lat_std, long_mean, long_std)
                np_x_gen = pytorch_to_scaled_numpy(x_gen, lat_mean, lat_std, long_mean, long_std)
                save_trajectory(np_x_cut, np_x_gen, state + 1, i + 1, lr, test_loss.cpu().item())

        print("Min test loss: {}".format(min_loss))

