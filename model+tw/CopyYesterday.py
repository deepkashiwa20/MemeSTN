import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from configparser import ConfigParser
import time
import sys
import logging
import shutil
import pandas as pd
import numpy as np
from Utils import get_pref_id, get_flow, get_seq_data, getXSYS
import Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the total ratio of training data and validation data')
parser.add_argument('--val_ratio', type=float, default=0.2, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seq_len', type=int, default=6, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=6, help='sequence length of observed historical values')
parser.add_argument('--ex', type=str, default='typhoon-inflow', help='which experiment setting to run') 
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

########################### used by HistoricalAverage ########################
def get_seq_data_idx(data, seq_len):
    seq_data_idx = [np.arange(i, i+seq_len) for i in range(0, data.shape[0]-seq_len+1)]
    return np.array(seq_data_idx)

def getXSYS_idx(data, mode, his_len, seq_len, trainval_ratio):
    seq_data = get_seq_data_idx(data, seq_len + his_len)
    XS, YS = seq_data[:, :his_len, ...], seq_data[:, seq_len:, ...]
    train_num = int(seq_data.shape[0] * trainval_ratio)
    if mode == 'train':    
        XS, YS = XS[:train_num, ...], YS[:train_num, ...]
    elif mode == 'test':
        XS, YS = XS[train_num:, ...], YS[train_num:, ...]    
    else:
        assert 'It should be either train or test'
    return XS, YS

def CopyYesterday(data, YS_index):
    HISTORYDAY, DAYTIMESTEP = 1, 24
    XS_Week = []
    for i in range(YS_index.shape[0]):
        Week = []
        for j in range(YS_index.shape[1]):
            index = YS_index[i, j]
            Week.append(data[index-HISTORYDAY*DAYTIMESTEP:index:DAYTIMESTEP, :])
        XS_Week.append(Week)
    XS_Week = np.array(XS_Week)
    YS_pred = np.mean(XS_Week, axis=2)
    logger.info('XS_Week.shape', XS_Week.shape)
    logger.info('YS_pred.shape', YS_pred.shape)
    return YS_pred
########################### used by HistoricalAverage ########################

def testModel(name, mode, data, YS, YS_index):
    print('TIMESTEP_IN, TIMESTEP_OUT', opt.his_len, opt.seq_len)
    YS_pred = CopyYesterday(data, YS_index)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(path + f'/{name}_prediction.npy', YS_pred)
    np.save(path + f'/{name}_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    f = open(path + f'/{name}_prediction_scores.txt', 'a')
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(opt.seq_len):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()

def main():
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(start=flow_start_date, end=flow_end_date, freq=freq)]
    start_index, end_index = flow_all_times.index(target_start_date), flow_all_times.index(target_end_date)
    area_index = get_pref_id(pref_path, target_area)
    flow = get_flow(flow_type, flow_path, start_index, end_index, area_index)
    logger.info('original flow data ...', flow.shape, flow.min(), flow.max())
    
    _, testYS = getXSYS(flow, 'test', opt.his_len, opt.seq_len, opt.trainval_ratio)
    _, testYS_idx = getXSYS_idx(flow, 'test', opt.his_len, opt.seq_len, opt.trainval_ratio)
    testModel(model_name, 'test', flow, testYS, testYS_idx)

    logger.info(event, flow_type, model_name, 'start and end time ...', time.ctime())
    
if __name__ == '__main__':
    main()
