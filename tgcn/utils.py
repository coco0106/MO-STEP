
import numpy as np
import os
import pickle
import torch
import fastdtw
import math

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]

                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size):
    data = {}
    for category in ['train', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'test']:
        data['x_' + category][...,0] = scaler.transform(data['x_' + category][..., 0])
        # data['y_' + category][...,0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(
        data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['test_loader'] = DataLoader(
        data['x_test'], data['y_test'], batch_size, shuffle=False)

    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def calculate_laplacian_with_self_loop(matrix):
   
    matrix = matrix + torch.eye(matrix.size(0)).cuda()
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


def masked_reward(preds, labels, actions, t):
    mask = (labels != np.nan)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
   
    
    loss = torch.where(actions.cuda() == 1, loss, torch.zeros_like(loss).cuda())
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    time = torch.where(loss != 0, torch.zeros_like(loss),
                       torch.ones_like(loss)*(t+1))
  
    return torch.stack([loss, time],0)


def reward(actions, predictions, reals):
    rewards = []
    for i in range(actions.shape[0]):
        
        r = masked_reward(predictions[i], reals, actions[i], i)
        rewards.append(r)
    return torch.stack(rewards,0)
def mean_batch(list):
    mean=sum(list)/len(list)
    return mean