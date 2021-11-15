import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
from configparser import ConfigParser
import logging
import Metrics
from STTN import STTN
from Utils import get_pref_id, get_flow, get_adj, get_seq_data, getXSYS_single, getXSYS, get_onehottime, get_floatdaytime, get_twitter

def refineXSYS(XS, YS):
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    return XS, YS

def mergeInfo(*args):
    return np.concatenate(args, axis=-1)

def getModel():
    model = STTN(in_channels=opt.channelin, embed_size=64, num_layers=1, input_step=opt.his_len, pred_step=opt.seq_len, heads=8, forward_expansion=32, gpu=device, dropout=0).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
    logger.info('Model Training Started ...', time.ctime())
    logger.info('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    model = getModel()
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - opt.val_ratio))
    logger.info('XS_torch.shape:  ', XS_torch.shape)
    logger.info('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, opt.batch_size, shuffle=True)
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    min_val_loss = np.inf
    wait = 0   
    for epoch in range(opt.epoch):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), path + f'/{name}.pt')
        else:
            wait += 1
            if wait == opt.patience:
                logger.info('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        logger.info("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(path + f'/{name}_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, opt.batch_size, shuffle=False))
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(path + f'/{name}_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    logger.info("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    logger.info('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    logger.info('Model Testing Started ...', time.ctime())
    logger.info('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False)
    model = getModel()
    model.load_state_dict(torch.load(path + f'/{name}.pt'))
    if opt.loss == 'MSE': criterion = nn.MSELoss()
    if opt.loss == 'MAE': criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred) 
    YS, YS_pred = YS.reshape(-1, YS.shape[-1]), YS_pred.reshape(-1, YS_pred.shape[-1])
    YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
    YS, YS_pred = YS.reshape(-1, opt.seq_len, YS.shape[-1]), YS_pred.reshape(-1, opt.seq_len, YS_pred.shape[-1])
    logger.info('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(path + f'/{name}_prediction.npy', YS_pred)
    np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    logger.info('*' * 40)
    logger.info("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(path + f'/{name}_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    logger.info("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(opt.seq_len):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        logger.info("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    logger.info('Model Testing Ended ...', time.ctime())
        
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default='MAE', help="MAE, MSE, SELF")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--patience", type=float, default=10, help="patience used for early stop")
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the total ratio of training data and validation data')
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
parser.add_argument('--ex', type=str, default='typhoon-inflow', help='which experiment setting to run') 
parser.add_argument('--channelin', type=int, default=2, help='number of input channel')
parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
opt = parser.parse_args()

config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
exp = opt.ex
channel = config.getint(exp, 'channel')
event = config[exp]['event']
flow_type = config[exp]['flow_type']
flow_type = config[exp]['flow_type']
flow_path = config[exp]['flow_path']
adj_path = config[exp]['adj_path']
twitter_path = config[exp]['twitter_path']
pref_path = config[exp]['pref_path']
freq = config[exp]['freq']
flow_start_date = config[exp]['flow_start_date']
flow_end_date = config[exp]['flow_end_date']
twitter_start_date = config[exp]['twitter_start_date']
twitter_end_date = config[exp]['twitter_end_date']
target_start_date = config[exp]['target_start_date']
target_end_date = config[exp]['target_end_date']
target_area = eval(config[exp]['target_area'])
num_variable = len(target_area)

_, filename = os.path.split(os.path.abspath(sys.argv[0]))
filename = os.path.splitext(filename)[0]
model_name = filename.split('_')[-1]
path = f'./save/{exp}_{model_name}_' + time.strftime('%Y%m%d%H%M', time.localtime())
logging_path = f'{path}/logging.txt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('channel', channel)
logger.info('event', event)
logger.info('flow_type', flow_type)
logger.info('flow_path', flow_path)
logger.info('adj_path', adj_path)
logger.info('twitter_path', twitter_path)
logger.info('pref_path', pref_path)
logger.info('freq', freq)
logger.info('flow_start_date', flow_start_date)
logger.info('flow_end_date', flow_end_date)
logger.info('twitter_start_date', twitter_start_date)
logger.info('twitter_end_date', twitter_end_date)
logger.info('target_start_date', target_start_date)
logger.info('target_end_date', target_end_date)
logger.info('target_area', target_area)
logger.info('model_name', model_name)

device = torch.device("cuda:{}".format(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)

scaler = MinMaxScaler((-1, 1))
scaler_tw = MinMaxScaler((-1, 1))

def main():
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(start=flow_start_date, end=flow_end_date, freq=freq)]
    start_index, end_index = flow_all_times.index(target_start_date), flow_all_times.index(target_end_date)
    area_index = get_pref_id(pref_path, target_area)
    flow = get_flow(flow_type, flow_path, start_index, end_index, area_index)
    twitter = get_twitter(twitter_path, pref_path, target_start_date, target_end_date, target_area)
    data = scaler.fit_transform(flow)
    data_tw = scaler_tw.fit_transform(twitter)
    logger.info('flow.shape, twitter.shape', data.shape, data.min(), data.max(), data_tw.shape, data_tw.min(), data_tw.max())
    
    logger.info(opt.ex, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'train', opt.his_len, opt.seq_len, opt.trainval_ratio)
    trainXS, trainYS = refineXSYS(trainXS, trainYS)
    trainXS_tw, trainYS_tw = getXSYS(data_tw, 'train', opt.his_len, opt.seq_len, opt.trainval_ratio)
    trainXS_tw, trainYS_tw = refineXSYS(trainXS_tw, trainYS_tw)
    trainXS = mergeInfo(trainXS, trainXS_tw)
    logger.info('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(model_name, 'train', trainXS, trainYS)
        
    logger.info(opt.ex, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'test', opt.his_len, opt.seq_len, opt.trainval_ratio)
    testXS, testYS = refineXSYS(testXS, testYS)
    testXS_tw, testYS_tw = getXSYS(data_tw, 'test', opt.his_len, opt.seq_len, opt.trainval_ratio)
    testXS_tw, testYS_tw = refineXSYS(testXS_tw, testYS_tw)
    testXS = mergeInfo(testXS, testXS_tw)
    logger.info('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(model_name, 'test', testXS, testYS)

    
if __name__ == '__main__':
    main()



