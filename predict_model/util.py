import csv
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import random
import copy
import numpy as np
import json
import argparse
import pandas as pd
from scipy.sparse.linalg import eigs
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

c_path = '/home/zhangmin/toby/IBA_Project_24spr/data/ch_flight_data'
u_path = '/home/zhangmin/toby/IBA_Project_24spr/data/us_flight_data'


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def cal_mape(y_true, y_pred, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def test_error(y_predict, y_test):
    """
    Calculates MAE, RMSE, R2.
    :param y_test:
    :param y_predict.
    :return:
    """
    # print(y_predict.shape, y_test.shape)
    err = y_predict - y_test
    MAE = np.mean(np.abs(err[~np.isnan(err)]))
    s_err = err**2
    RMSE = np.sqrt(np.mean((s_err[~np.isnan(s_err)])))

    MAPE = cal_mape(y_true=y_test, y_pred=y_predict, null_val=0.0)

    return MAE, RMSE, MAPE


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_data(data_name, ratio=[0.6, 0.2]):
    if data_name == 'US':
        adj_mx = np.load(u_path+'/adj_mx.npy')
        od_power = np.load(u_path+'/od_pair.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(70):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load(u_path+'/udelay.npy')
        wdata = np.load(u_path+'/weather2016_2021.npy')

    if data_name == 'China':
        adj_mx = np.load(c_path+'/dist_mx.npy')
        od_power = np.load(c_path+'/od_mx.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(50):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load(c_path+'/delay.npy')
        data[data < -15] = -15
        wdata = np.load(c_path+'/weather_cn.npy')

    training_data = data[:, :int(ratio[0]*data.shape[1]), :]
    val_data = data[:, int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1]), :]
    # test_data = data[:, int((ratio[0] + ratio[1])*data.shape[1]):, :]
    training_w = wdata[:, :int(ratio[0]*data.shape[1])]
    val_w = wdata[:, int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1])]
    # test_w = wdata[:, int((ratio[0] + ratio[1])*data.shape[1]):]
    return adj, training_data, val_data, training_w, val_w


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_wmae(preds, labels, weights, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask * weights
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def train_dataloader(batch_index, num_batch, training_data, training_w, j, in_len, out_len):
    trainx = []
    trainy = []
    trainw = []
    for k in range(num_batch):
        trainx.append(np.expand_dims(
            training_data[:, batch_index[j * num_batch + k]: batch_index[j * num_batch + k] + in_len, :], axis=0))
        trainy.append(np.expand_dims(training_data[:, batch_index[j * num_batch + k] +
                      in_len:batch_index[j * num_batch + k] + in_len + out_len, :], axis=0))
        trainw.append(np.expand_dims(
            training_w[:, batch_index[j * num_batch + k]: batch_index[j * num_batch + k] + in_len], axis=0))
    trainx = np.concatenate(trainx)
    trainy = np.concatenate(trainy)
    trainw = np.concatenate(trainw)
    return trainx, trainy, trainw


def test_dataloader(val_index, val_data, val_w, i, in_len, out_len):
    testx = np.expand_dims(
        val_data[:, val_index[i]: val_index[i] + in_len, :], axis=0)
    testw = np.expand_dims(
        val_w[:, val_index[i]: val_index[i] + in_len], axis=0)
    return testx, testw


def label_loader(val_index, in_len, out_len, delay_index, val_data, graph=False):
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(
            val_data[:, val_index[i] + in_len:val_index[i] + in_len + out_len, :], axis=0))
    label = np.concatenate(label)
    '''
    if graph == False:
        if delay_index ==0:
            label = label[:,:,:,0] #if last dimension = 1, var 44.19 #if last dimension=0, var 115.615
        else:
            label = label[:,:,:,1]
    elif graph == True:
        label = label[:,:,:,delay_index]'''
    return label[:, :, :, delay_index]


def model_preprocess(model, lr, gamma, step_size, training_data, val_data, in_len, out_len):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=gamma, step_size=step_size)
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(
    ), training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    batch_index = list(range(training_data.shape[1] - (in_len + out_len)))
    val_index = list(range(val_data.shape[1] - (in_len + out_len)))
    return optimizer, scheduler, scaler, training_data, batch_index, val_index


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def slot_to_time(slot):
    # Calculate the number of days and the slot within that day.
    days_passed = slot // 36
    slot_in_day = slot % 36

    # Calculate the hour and minute based on the slot_in_day.
    hour = 6 + slot_in_day // 2
    minute = (slot_in_day % 2) * 30

    # Add the days_passed to the starting date (2016-01-01).
    date = datetime(2016, 1, 1) + timedelta(days=days_passed)
    date_str = date.strftime('%Y-%m-%d')

    return f"{date_str} {hour:02d}:{minute:02d}"


def plot_line(horizons, *data, data_name=None, colors, title='Title', y_name='Arrival Delay Time (min)', x_name='Time Step (30 min)', inset_ranges=None, width=1.7, size=7):
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 30

    plt.figure(figsize=(30, 10))

    plt.rcParams["font.family"] = "Times New Roman"  # 修改字体为Times New Roman
    plt.rcParams["font.size"] = 14
    plt.grid(visible=True, which='major', linestyle='-')
    plt.grid(visible=True, which='minor', linestyle='--', alpha=0.3)
    plt.minorticks_on()

    for i, dataset in enumerate(data):
        if data_name[i] == 'Truth':
            plt.plot(horizons, dataset, marker='s^odx'[
                     i], markersize=size, linestyle='-', label=data_name[i], color=colors[i], linewidth=width + 0.5, alpha=0.8)
        elif (data_name[i] == 'ASTGCN'):
            plt.plot(horizons, dataset, marker='s^odx'[
                     i], markersize=size, linestyle='-', label=data_name[i], color=colors[i], linewidth=width + 1)
        else:
            plt.plot(horizons, dataset, marker='s^odx*D'[
                i], markersize=size, linestyle='--', label=data_name[i], color=colors[i], linewidth=width, alpha=0.6)

    if inset_ranges:
        count = 0
        for inset_range in inset_ranges:
            plt.axvspan(inset_range[0], inset_range[1],
                        color='yellow', alpha=0.2)
            period_x = (inset_range[0] + inset_range[1]) / 2
            plt.text(period_x, plt.ylim()[
                     0], f'Period {count + 1}', ha='center', va='bottom', fontsize=10, color='black')
            count += 1

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.legend(loc='upper left', frameon=True, edgecolor='black',
               framealpha=0.5, borderaxespad=0.5, fontsize=10)

    # plt.ylim(-15, 35)

    plt.show()
    plt.close()


def get_airport_info(airport_name):
    file_path = '../data/us_flight_data/us_airport_loc.csv'
    count = 0
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Name'] == airport_name:
                return {
                    'idx': count,
                    'AIRPORT': row['AIRPORT'],
                    'CITY': row['CITY'],
                    'STATE': row['STATE'],
                    'COUNTRY': row['COUNTRY'],
                    'LATITUDE': row['LATITUDE'],
                    'LONGITUDE': row['LONGITUDE']
                }
            count += 1
    return None


def get_delay_time(airport, time):
    info = get_airport_info(airport)
    airport_idx = info['idx']
    step = 0
    length = 200
    ASTGCN_arr = np.load("../saves/ASTGCN/arr/arr_25.npz")['predict']
    ASTGCN_dep = np.load("../saves/ASTGCN/dep/dep_49.npz")['predict']
    true_time_step = 47359
    ASTGCN_arr = ASTGCN_arr[time, airport_idx, step]
    ASTGCN_dep = ASTGCN_dep[time, airport_idx, step]

    info['Arrival Delay'] = ASTGCN_arr
    info['Departure Delay'] = ASTGCN_dep
    info['Time'] = slot_to_time(true_time_step+time)

    return info
