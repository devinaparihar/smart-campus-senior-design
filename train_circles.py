import numpy as np
import torch
import matplotlib.pyplot as plt

from gmn import *

DIR = "./images2/"
def save_as_image(input_points, generated_points, name):
    plt.figure()
    np_input = input_points.cpu().numpy()[0]
    np_gen = generated_points.cpu().numpy()[0]
    plt.scatter(np_input[:, 0].ravel(), np_input[:, 1].ravel(), c='b', label='input points')
    plt.scatter(np_gen[:, 0].ravel(), np_gen[:, 1].ravel(), c='r', label='generated points')
    plt.legend()
    plt.savefig(DIR + str(name) + ".png")
    plt.close()

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
min_y = None
min_mask = None
for state in range(1):
    print("Model: {}".format(state))
    net = DeepLSTM()
    if CUDA:
        net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for i in range(5000):
        print("Epoch: {}".format(i))
        optimizer.zero_grad()
        x, mask = sample(train_data, train_mask, BATCH_SIZE)
        train_loss, _, _, _ = forward_pass_batch(x, mask)
        train_loss.backward()
        optimizer.step()
        print("Training Loss: {}".format(train_loss.item()))
        with torch.no_grad():
            x, mask = sample(test_data, test_mask, BATCH_SIZE)
            test_loss, means, y_test, mask_test = forward_pass_batch(x, mask)
            print("Test Loss: {}".format(test_loss.item()))
            if test_loss.item() < min_loss:
                min_loss = test_loss.cpu().item()
                x, mask = sample(val_data, val_mask, 1)
                loss, means, y_test, mask_test = forward_pass_batch(x, mask)
                min_means = means
                min_y = y_test
                min_mask = mask_test
                np_mask = mask.cpu().numpy()
                n = max([i + 1 if np_mask[0, i] == 1 else 0 for i in range(np_mask.shape[1])])
                k = np.random.randint(3, n - 6)
                x_cut = x.narrow(1, 0, n - k)
                print("SAMPLE GENERATION")
                print("Input:\n{}".format(x_cut))
                x_gen = net.generate(x_cut, k)
                print("Generated Output:\n{}".format(x_gen))
                print("True Output:\n{}".format(x.narrow(1, n - k, k)))
                print("FORWARD PASS FOR REFERENCE")
                loss, means, y, mask = forward_pass_batch(x, mask)
                print("Means:\n{}".format(means))
                print("Target:\n{}".format(y))
                save_as_image(x_cut, x_gen, "Epoch_{:04d}_TestLoss_{}".format(i, test_loss.cpu().item()))

        print("Min test loss: {}".format(min_loss))
        print("Min test loss difference:\n{}".format(torch.abs(min_means - min_y)))
        print("Min test loss target:\n{}".format(min_y))
        print("Min test loss mask:\n{}".format(min_mask))
