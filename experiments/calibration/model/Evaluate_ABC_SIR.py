import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from utils import read_model

from VI_run import add_exp_args
from model.model import get_model, get_loss, get_model_id, add_model_args
from about_data.data import get_data, add_data_args

from model.subclass.HPD import hpd_grid

from plotting import SIR_plotting


from plotting import SIR_plotting
from numpy import genfromtxt
import numpy as np

from model.SIR_AIS import only_forward

import os

##############################
#Handling Model Reading Error#
##############################
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()

add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)

setting_choices = {"ABC"}

parser.add_argument('--setting', type=str, default="ABC", choices=setting_choices)

args = parser.parse_args()
args.dataset= "SIR"
args.device="cpu"
#Part 1 Read model, load model
log_path, _ = read_model.read_file(args)

l = []
r = []
mu = []
mse = []
al = []
coverage = []


args.batch_size = 256
if __name__ == "__main__":

    for i in range(100):
        if i != 0:
            continue
        truex, data_shape = get_data(args, model_num=i)

        SIR_writer = SIR_plotting(log_path=log_path)

        #abs_path = '../R_data/SIR/'
        abs_path = log_path
        fname1 = '\ABC_{}_res_wide.csv'.format(i)
        fname2 = '\ABC_{}_forward_I_wide.csv'.format(i)
        fname3 = '\ABC_{}_forward_R_wide.csv'.format(i)

        path = abs_path + fname1
        params = genfromtxt(path, delimiter=',', skip_header=True)

        tl, tr, tmu, tmse, tcoverage = SIR_writer.param_img_store(params=params, model_num=i)

        pathI = abs_path + fname2
        pathR = abs_path + fname3
        I = genfromtxt(pathI, delimiter=',', skip_header=True)
        R = genfromtxt(pathR, delimiter=',', skip_header=True)

        groundtruth = np.load("../data/SIR/SIR.npy")

        samples = np.stack((I, R), axis=2)
        samples = torch.asarray(samples)
        #print(samples.shape)
        samples = only_forward(param_sample=params, args=args)

        print(samples.shape)
        res = SIR_writer.forward_img_store(samples=samples, truex=truex,
                                     groundtruth=groundtruth, model_num=i)
        #res.append(cover_perc)
        l.append(tl)
        r.append(tr)
        mu.append(tmu)
        mse.append(tmse)
        al.append(res)
        coverage.append(tcoverage)
        print(res)
        print("{} done".format(i), end='\r')




    l = np.asarray(l)
    r = np.asarray(r)
    mu = np.asarray(mu)
    mse = np.asarray(mse)
    al = np.asarray(al)
    tc = np.asarray(coverage)

    # print(l, r, mu, mse)
    res_list = [np.around(np.mean(x, axis=0), decimals=4) for x in (l, r, mu, mse, al, tc)]

    for res in res_list:
        print(res)

    for pair in zip(res_list[0], res_list[1]):
        print(pair, end="&")

