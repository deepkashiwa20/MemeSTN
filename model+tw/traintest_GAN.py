import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import itertools
import struct
import argparse
from configparser import ConfigParser
import logging
import seaborn as sns
import time
import sys
import shutil
import pandas as pd
import numpy as np
from torchsummary import summary
from GAN import Generator
from GAN import Discriminator
from Utils import get_pref_id, get_flow, get_adj, get_twitter, get_onehottime, get_seq_data, sym_adj


def show_loss_hist(hist, path):
    x = range(len(hist['D_losses_train']))
    y1 = hist['D_losses_train']
    y2 = hist['G_losses_train']
    y3 = hist['D_losses_test']
    y4 = hist['G_losses_test']
    plt.plot(x, y1, label='D_loss_train')
    plt.plot(x, y2, label='G_loss_train')
    plt.plot(x, y3, label='D_loss_test')
    plt.plot(x, y4, label='G_loss_test')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def show_RMSE_hist(hist, path):
    x = range(len(hist['RMSE_test']))
    y1 = hist['RMSE_test']
    plt.plot(x, y1, label='RMSE_test')
    plt.xlabel('Iter')
    plt.ylabel('RMSE')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def traintest(D, G, x, y, adj, device):
    num_train_sample = int(x.shape[0] * opt.trainval_ratio)
    train_x, train_y, train_adj = x[:num_train_sample, ...], y[:num_train_sample, ...], adj[:num_train_sample, ...]
    test_x, test_y, test_adj = x[num_train_sample:, ...], y[num_train_sample:, ...], adj[num_train_sample:, ...]

    # dataset = Data.TensorDataset(x, y, adj)
    # train_loader = Data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    train_x, train_y, train_adj = torch.tensor(train_x).to(device), torch.tensor(train_y).to(device), torch.tensor(train_adj).to(device)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_adj)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    test_x, test_y, test_adj = torch.tensor(test_x).to(device), torch.tensor(test_y).to(device), torch.tensor(test_adj).to(device)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_adj)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size)  # no shuffle here.

    opt_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)
    opt_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    MAE_loss = nn.L1Loss()

    loss_hist = {}
    loss_hist['D_losses_train'] = []
    loss_hist['G_losses_train'] = []
    loss_hist['D_losses_test'] = []
    loss_hist['G_losses_test'] = []
    loss_hist['RMSE_test'] = []

    for epoch in range(opt.epoch):
        # learning rate decay
        if epoch == 10:  # or epoch == 15:
            opt_G.param_groups[0]['lr'] /= 10
            opt_D.param_groups[0]['lr'] /= 10

        # training
        D_losses_train = []
        G_losses_train = []
        D.train()
        G.train()
        for step, (b_x, b_y, b_adj) in enumerate(train_loader):
            ######################### Train Discriminator #######################
            opt_D.zero_grad()
            num_seq = b_x.size(0)  # batch size of sequences
            real_seq = Variable(b_x.to(device)).float()  # put tensor in Variable
            seq_label = Variable(b_y.to(device)).float()
            seq_adj = Variable(b_adj.to(device)).float()
            prob_real_seq_right_pair = D(real_seq, seq_adj, seq_label)

            noise = torch.randn(num_seq, opt.seq_len * opt.init_dim * num_variable).view(num_seq, opt.seq_len, num_variable, opt.init_dim)
            noise = Variable(noise.to(device))  # randomly generate noise

            fake_seq = G(noise, seq_adj, seq_label)
            prob_fake_seq_pair = D(fake_seq, seq_adj, seq_label)

            # sample real seqs from database(just shuffle this batch seqs)
            shuffled_row_idx = torch.randperm(num_seq)
            real_shuffled_seq = b_x[shuffled_row_idx]
            real_shuffled_seq = Variable(real_shuffled_seq.to(device)).float()
            shuffled_adj = b_adj[shuffled_row_idx]
            shuffled_adj = Variable(shuffled_adj.to(device)).float()
            prob_real_seq_wrong_pair = D(real_shuffled_seq, shuffled_adj, seq_label)

            D_loss = - torch.mean(torch.log(prob_real_seq_right_pair) + torch.log(1. - prob_fake_seq_pair) + torch.log(1. - prob_real_seq_wrong_pair))
            D_loss.backward()
            opt_D.step()
            D_losses_train.append(D_loss.item())

            ########################### Train Generator #############################
            opt_G.zero_grad()
            noise2 = torch.randn(num_seq, opt.seq_len * opt.init_dim * num_variable).view(num_seq, opt.seq_len, num_variable, opt.init_dim)
            noise2 = Variable(noise2.to(device))

            # create random label
            y_real = Variable(torch.ones(num_seq).to(device))
            G_result = G(noise2, seq_adj, seq_label)
            D_result = D(G_result, seq_adj, seq_label).squeeze()
            #logger.info(BCE_loss(D_result, y_real), MAE_loss(G_result, real_seq))   # check magnitude
            G_loss = BCE_loss(D_result, y_real)
            G_loss.backward()
            opt_G.step()
            G_losses_train.append(G_loss.item())

        D_losses_train = torch.mean(torch.FloatTensor(D_losses_train)).item()
        G_losses_train = torch.mean(torch.FloatTensor(G_losses_train)).item()
        loss_hist['D_losses_train'].append(D_losses_train)
        loss_hist['G_losses_train'].append(G_losses_train)

        # testing
        rmse_test = []
        D_losses_test = []
        G_losses_test = []
        D.eval()
        G.eval()
        with torch.no_grad():
            for step, (b_x, b_y, b_adj) in enumerate(test_loader):
                ######################### Test Discriminator #######################
                num_seq = b_x.size(0)  # batch size of sequences
                real_seq = Variable(b_x.to(device)).float()  # put tensor in Variable
                seq_label = Variable(b_y.to(device)).float()
                seq_adj = Variable(b_adj.to(device)).float()
                prob_real_seq_right_pair = D(real_seq, seq_adj, seq_label)

                noise = torch.randn(num_seq, opt.seq_len * opt.init_dim * num_variable).view(num_seq, opt.seq_len, num_variable, opt.init_dim)
                noise = Variable(noise.to(device))  # randomly generate noise

                fake_seq = G(noise, seq_adj, seq_label)
                prob_fake_seq_pair = D(fake_seq, seq_adj, seq_label)

                # sample real seqs from database(just shuffle this batch seqs)
                shuffled_row_idx = torch.randperm(num_seq)
                real_shuffled_seq = b_x[shuffled_row_idx]
                real_shuffled_seq = Variable(real_shuffled_seq.to(device)).float()
                shuffled_adj = b_adj[shuffled_row_idx]
                shuffled_adj = Variable(shuffled_adj.to(device)).float()
                prob_real_seq_wrong_pair = D(real_shuffled_seq, shuffled_adj, seq_label)

                D_loss = - torch.mean(torch.log(prob_real_seq_right_pair) + torch.log(1. - prob_fake_seq_pair) + torch.log(1. - prob_real_seq_wrong_pair))
                D_losses_test.append(D_loss.item())

                ########################### Test Generator #############################
                opt_G.zero_grad()
                noise2 = torch.randn(num_seq, opt.seq_len * opt.init_dim * num_variable).view(num_seq, opt.seq_len, num_variable, opt.init_dim)
                noise2 = Variable(noise2.to(device))

                # create random label
                y_real = Variable(torch.ones(num_seq).to(device))
                G_result = G(noise2, seq_adj, seq_label)
                D_result = D(G_result, seq_adj, seq_label).squeeze()
                G_loss = BCE_loss(D_result, y_real)
                G_losses_test.append(G_loss.item())

                pred = fake_seq.cpu().data.numpy()
                truth = b_x.cpu().data.numpy()
                rmse_test.append(np.sqrt(np.mean(np.square(pred - truth))))

        D_losses_test = torch.mean(torch.FloatTensor(D_losses_test)).item()
        G_losses_test = torch.mean(torch.FloatTensor(G_losses_test)).item()
        rmse_test = np.mean(rmse_test)
        loss_hist['D_losses_test'].append(D_losses_test)
        loss_hist['G_losses_test'].append(G_losses_test)
        loss_hist['RMSE_test'].append(rmse_test)

        logger.info('Epoch', epoch, time.ctime(), 'D_loss_train, G_loss_train, D_loss_test, G_loss_test, RMSE_test', D_losses_train, G_losses_train, D_losses_test, G_losses_test, rmse_test)

        if (epoch + 1) % 10 == 0:
            truth = b_x.cpu().data.numpy()
            fake = fake_seq.cpu().data.numpy()
            # logger.info(fake.shape, truth.shape) # the last batch size 839%64=23
            sns.set_style('darkgrid')
            fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
            ax[0].plot(fake[0, :, :, 0])
            ax[0].legend(target_area)
            ax[0].set_title('Generated Time Series')
            ax[0].set_ylim(-1.0, 1.0)
            ax[1].plot(truth[0, :, :, 0])
            ax[1].legend(target_area)
            ax[1].set_title('Ground-truth Time Series')
            ax[1].set_ylim(-1.0, 1.0)
            plt.savefig('{}/fake_seqs_{}.png'.format(path, epoch + 1))
            plt.close()

    show_loss_hist(loss_hist, path=f'{path}/loss_hist.png')
    show_RMSE_hist(loss_hist, path=f'{path}/RMSE_hist.png')
    torch.save(G.state_dict(), f'{path}/G_params.pkl')  # save parameters
    torch.save(D.state_dict(), f'{path}/D_params.pkl')


##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs of training") # original 1500
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the total ratio of training data and validation data')
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--adj_bar", type=float, default=0.47, help="adj bar")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--l2', type=float, default=0.1, help='l2 penalty')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of values, which should be even nums (2,4,6,12)')
parser.add_argument('--his_len', type=int, default=12, help='sequence length of observed historical values')
parser.add_argument('--num_head', type=int, default=7, help='number of heads in self-attention') # should be fully divided by num_variable
#parser.add_argument('--num_block', type=int, default=2, help='repeating times of buiding block') # original 3
parser.add_argument('--num_block_D', type=int, default=2, help='repeating times of buiding block for D') # original 3
parser.add_argument('--num_block_G', type=int, default=3, help='repeating times of buiding block for G') # original 3
parser.add_argument('--D_hidden_feat', type=int, default=16, help='hidden features of D')
parser.add_argument('--G_hidden_feat', type=int, default=64, help='hidden features of G')
parser.add_argument('--D_final_feat', type=int, default=1, help='output features of D')
parser.add_argument('--G_final_feat', type=int, default=1, help='output features of G')
parser.add_argument("--init_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument('--D_init_feat', type=int, default=1, help='input features of D (only include init features)') # should be equal to G_final_feat, dummy variable to G_final_feat
parser.add_argument('--G_init_feat', type=int, default=100, help='input features of G (include init features)')
tw_condition, his_condition = False, False
parser.add_argument('--cond_feat', type=int, default=32 + sum([tw_condition, his_condition]), help='condition features of D and G')
parser.add_argument('--cond_source', type=int, default=sum([1, tw_condition, his_condition]), help='1 is only time label, 2 is his_x or twitter label, 3 is time, twitter, his')
parser.add_argument('--ex', type=str, default='covid-inflow-kanto7', help='which experiment setting to run') 
# {'typhoon-inflow-kanto8', 'typhoon-outflow-kanto8', 'covid-inflow-kanto8', 'covid-outflow-kanto8'}
opt = parser.parse_args()

condition_flag = 't'
if tw_condition==False and his_condition==False:
    pass
elif tw_condition==True and his_condition==False:
    condition_flag += '-tw'
elif tw_condition==False and his_condition==True:
    condition_flag += '-his'
else:
    condition_flag += '-tw-his'

config = ConfigParser()
config.read('params_GAN.txt', encoding='UTF-8')
exp = opt.ex
channel = config.getint(exp, 'channel') # already in argument parser opt.
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
num_variable = len(target_area) # already in argument parser opt.

_, filename = os.path.split(os.path.abspath(sys.argv[0]))
filename = os.path.splitext(filename)[0]
model_name = filename.split('_')[-1]
path = f'./save/{exp}_{model_name}_{condition_flag}_head{opt.num_head}_' + time.strftime('%Y%m%d%H%M', time.localtime())
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
##########################################################################################################

# min-max normalization: x -> [-1,1]
def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    x = 2.0*x - 1.0
    return x # return x, min_x, max_x if we want to revert transform

def get_data(x, cond, tw, adj, num_variable, channel):  
    x = min_max_normal(x)
    x = x[:, :, np.newaxis].repeat(channel, axis=2) # final_feat=channel=1
    tw = min_max_normal(tw)
    tw = tw[:, :, np.newaxis].repeat(channel, axis=2) # final_feat=channel=1
    cond = cond[:, np.newaxis].repeat(num_variable, axis=1) # this is condition/label
    adj = sym_adj(adj)
    adj = adj[np.newaxis, :, :].repeat(x.shape[0], axis=0)
    return x, cond, tw, adj

def main():   
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(start=flow_start_date, end=flow_end_date, freq=freq)]
    start_index, end_index = flow_all_times.index(target_start_date), flow_all_times.index(target_end_date)
    area_index = get_pref_id(pref_path, target_area)
    flow = get_flow(flow_type, flow_path, start_index, end_index, area_index)
    onehottime = get_onehottime(target_start_date, target_end_date, freq)
    twitter = get_twitter(twitter_path, pref_path, target_start_date, target_end_date, target_area)
    adj = get_adj(adj_path, area_index)
    x, c, tw, adj = get_data(flow, onehottime, twitter, adj, num_variable, channel)
    seq_x, seq_c, seq_tw, seq_adj = get_seq_data(x, opt.seq_len+opt.his_len), get_seq_data(c, opt.seq_len+opt.his_len), \
                                    get_seq_data(tw, opt.seq_len+opt.his_len), get_seq_data(adj, opt.seq_len+opt.his_len)
    logger.info(flow.shape, twitter.shape, onehottime.shape, adj.shape)
    logger.info(x.shape, c.shape, tw.shape, adj.shape)
    logger.info(seq_x.shape, seq_c.shape, seq_tw.shape, seq_adj.shape)
    his_x, seq_x, seq_c, seq_tw, seq_adj = seq_x[:, :opt.his_len, ...], seq_x[:, -opt.seq_len:, ...], \
                                           seq_c[:, -opt.seq_len:, ...], seq_tw[:, -opt.seq_len:, ...], seq_adj[:, -opt.seq_len:, ...]
    logger.info(his_x.shape, seq_x.shape, seq_c.shape, seq_tw.shape, seq_adj.shape)
    # logger.info(his_x.min(), his_x.max(), seq_x.min(), seq_x.max(), seq_c.min(), seq_c.max(), seq_tw.min(), seq_tw.max(), seq_adj.min(), seq_adj.max())
    
    if tw_condition==False and his_condition==False:
        pass
    elif tw_condition==True and his_condition==False:
        seq_c = np.concatenate([seq_c, seq_tw], axis=-1)
    elif tw_condition==False and his_condition==True:
        seq_c = np.concatenate([seq_c, his_x], axis=-1)
    else:
        seq_c = np.concatenate([seq_c, seq_tw, his_x], axis=-1)
    logger.info(seq_x.shape, seq_c.shape, seq_adj.shape)
    logger.info('opt.cond_feat:', opt.cond_feat)
    
    print(opt.D_init_feat + opt.cond_feat)
    print(type(opt.D_init_feat + opt.cond_feat))
    print(opt.G_init_feat + opt.cond_feat)
    print(type(opt.G_init_feat + opt.cond_feat))
    
    D = Discriminator(opt.D_init_feat, opt.cond_feat, opt.D_hidden_feat, opt.D_final_feat, opt.num_head, opt.dropout, opt.num_block_D, num_variable, opt.seq_len).to(device)
    G = Generator(opt.G_init_feat, opt.cond_feat, opt.G_hidden_feat, opt.G_final_feat, opt.num_head, opt.dropout, opt.num_block_G, num_variable, opt.seq_len).to(device)
    
    logger.info('########## This is the model summary of Descriminator ############')
    # summary(D, [(opt.seq_len, num_variable, opt.D_init_feat), (opt.seq_len, num_variable, num_variable), (opt.seq_len, num_variable, opt.cond_feat)])
    logger.info('########## This is the model summary of Generator ############')
    # summary(G, [(opt.seq_len, num_variable, opt.G_init_feat), (opt.seq_len, num_variable, num_variable), (opt.seq_len, num_variable, opt.cond_feat)])
    
    start = time.ctime()
    traintest(D, G, seq_x, seq_c, seq_adj, device)
    end = time.ctime()
    logger.info('start and end time for total training process...', start, end)
    
if __name__ == '__main__':
    main()
