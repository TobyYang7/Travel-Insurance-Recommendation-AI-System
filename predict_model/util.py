# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:42:10 2022

@author: AA
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import util
import argparse
import random
import copy
import numpy as np
import json
import argparse
import pandas as pd

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
