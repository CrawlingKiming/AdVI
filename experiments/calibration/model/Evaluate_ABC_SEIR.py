from plotting import SEIR_plotting
from numpy import genfromtxt
import numpy as np
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

from plotting import SEIR_plotting


from plotting import SIR_plotting
from numpy import genfromtxt
import numpy as np

from model.SEIR_AIS import only_forward

import os

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


concat_samples = True

args = parser.parse_args()
args.dataset= "SEIR"
args.device="cpu"
log_path, _ = read_model.read_file(args)

l = []
r = []
mu = []
mse = []
al = []
coverage = []

args.batch_size = 1024

if __name__ == "__main__":
    abs_path = log_path#'../R_data/SEIR/'
    S_plot = SEIR_plotting(log_path=abs_path)


    fname1 = 'results_case_2_ABC_SMC_MNN_gen_session28.csv'

    if concat_samples :

        for idx in range(10):
            fname1 = '\{}results_case_2_ABC_SMC_MNN_gen_100_8.csv'.format(idx+1)
            path = abs_path + fname1
            params = genfromtxt(path, delimiter=',', skip_header=True)
            params[:, [1, 3, 4, 0, 2]] = params[:, [0, 1, 2, 3, 4]]
            if idx == 0 :
                tt = params
                continue
            tt = np.vstack((tt, params))
        params = tt
    else :
        path = abs_path + fname1
        params = genfromtxt(path, delimiter=',', skip_header=True)
        params[:, [1, 3, 4, 0, 2]] = params[:, [0, 1, 2, 3, 4]]

    fname2 = '5000_SEIR_ABC_forward_8000.csv'
    fname3 = "5000_SEIR_age_8.csv"


    S_plot.param_ls_img_store(params=[params], extra="SMC_MNN_008")

    pathI = abs_path + fname2
    pathage = abs_path + fname3


    #samples_f = genfromtxt(pathI, delimiter=',', skip_header=True)
    #print(samples_f.shape)
    #age = genfromtxt(pathage, delimiter=',', skip_header=True)
    samples_f, age = only_forward(param_sample=params, args=args)
    #print(age.shape)
    #print(samples_f.shape)
    samples = (samples_f.detach().cpu().numpy(), age.detach().cpu().numpy())
    samples_age_prop = np.asarray(age)

    age_perc = np.asarray([[42, 30, 28]])
    res = np.mean(np.square((np.mean(samples_age_prop, axis=0)*100 - age_perc)))
    #perc_ = samples_age_prop * 100 - age_perc
    #print(perc_)
    #plt.hist(np.sqrt(np.sum(np.square(perc_),axis=1)))
    #plt.show()
    #age_25 = np.percentile(samples_age_prop, 2.5, axis=0)
    #age_975 = np.percentile(samples_age_prop, 97.5, axis=0)

    #print(age_25, age_975)
    print("Age MSE : {}".format(res))
    #age_result.append(res)
    test_data = np.genfromtxt("../data/Measles_data_time.csv", delimiter=',', skip_header=True)

    S_plot.forward_img_store(samples=samples, truex=test_data[:,2], extra="SMC_MNN_Forward")
    #print(my_data)