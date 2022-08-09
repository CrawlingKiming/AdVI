import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import read_model

from VI_run import add_exp_args
from model.model import get_model, get_loss, get_model_id, add_model_args
from about_data.data import get_data, add_data_args

from model.subclass.HPD import hpd_grid

from model.MSIR_full_AIS import only_forward

from plotting import Experiment_Writing
import os

def HPD(args, samples, model_num):
    hpd_l = []
    hpd_r = []
    modes_mu = []
    mse = []
    coverage = []

    if args.dataset == "SIR":
        raise NotImplementedError

    if args.dataset == "SEIR":
        raise NotImplementedError

    if args.dataset == "MSIR" or "MSIR_full":
        samples = samples[-1]

        true = [0.43, 7.35, 0.027, 0.9, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]
        #temp[:, 1] = temp[:, 1] * 1e3

        true_beta = [1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]
        MSIR_writer = Experiment_Writing(log_path=log_path)
        MSIR_writer.param_img_store(params=samples, model_num=model_num)

        temp = samples * 1e2

        for j in range(10):
            param_cov = 0

            temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
            tm = np.mean(np.square(temp[:, j] - true[j]*1e2)) #/ temp.shape[0]
            a, b = temp_hpd[0]
            c = temp_modes[0]
            hpd_l.append(a/1e2)
            hpd_r.append(b/1e2)
            modes_mu.append(c/1e2)
            mse.append(tm/1e4)

            if a/1e2 < true[j] and b/1e2 > true[j]:
                param_cov +=1

            coverage.append(param_cov)

    return hpd_l, hpd_r, modes_mu, mse, coverage

def q_return(samples):
    samples_mean = np.mean(samples, axis=0)
    samples_min = np.percentile(samples, 2.5, axis=0)
    samples_max = np.percentile(samples, 97.5, axis=0)

    return samples_min, samples_mean, samples_max

def for_samples(args, samples, model_num=1, log_path=None):

    if args.dataset == "SIR":
        return None

    if args.dataset == "SEIR":
        raise NotImplementedError

    if (args.dataset == "MSIR") or ("MSIR_full"):
        # glambd = np.load("../data/MSIR_lambda.npy")
        # glambd = glambd[0] # 118 * 6
        burden_estimate = np.load("../data/ground_truth.npy")
        data_ground = np.load("../data/MSIR_{}.npy".format(model_num))

        MSIR_writer = Experiment_Writing(log_path=log_path)

        res = MSIR_writer.forward_img_store(samples=samples, truex=truex, data_ground=data_ground,
                                            burden_estimate=burden_estimate, model_num=model_num)

        return res

parser = argparse.ArgumentParser()

add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)

# setting_choices = {"BTAF", "BFAF", "BTAT", "BFAT"}
# parser.add_argument('--setting', type=str, default="BTAF", choices=setting_choices)

args = parser.parse_args()
args.AIS = False

# Part 1 Read model, load model
log_path, filenames = read_model.read_samples(args)
truex, data_shape = get_data(args)


l = []
r = []
mu = []
mse = []
al = []
coverage = []
ii = 0
print(log_path)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
for f in filenames:
    if ii<46:
        ii += 1
        continue
    print(f)
    try:
        truex, data_shape = get_data(args, model_num=ii)
        ii += 1

        data_path = os.path.join(log_path, f)
        param_sample = np.genfromtxt(data_path, delimiter=',', skip_header=True)
        param_sample[:, [3, 2]] = param_sample[:, [2, 3]]
        #test_data = torch.tensor(test_data[1:], dtype=torch.float)

        #forward_results
        thinned_sample = param_sample[np.linspace(0, param_sample.shape[0]-1, num=500).astype(int), :]
        #print(thinned_sample.shape)

        samples = only_forward(param_sample= thinned_sample, args=args)

        res = for_samples(args, samples, model_num=ii-1, log_path=log_path)

        #param_results
        param_sample = param_sample[np.linspace(0, param_sample.shape[0] - 1, num=5000).astype(int), :]
        tl, tr, tmu, tmse, tcoverage = HPD(args, [param_sample], model_num=ii-1)
        print(res)

    except Exception as e:
        print("Failed account {}".format(ii))
        continue

    l.append(tl)
    r.append(tr)
    mu.append(tmu)
    mse.append(tmse)
    al.append(res)
    coverage.append(tcoverage)

    print("{} done".format(ii), end='\r')


l = np.asarray(l)
r = np.asarray(r)
mu = np.asarray(mu)
mse = np.asarray(mse)
al = np.asarray(al)
tc = np.asarray(coverage)

#print(l, r, mu, mse)
res_list = [np.around(np.mean(x, axis=0), decimals=4) for x in (l, r, mu, al, tc)]
res_list.append(np.mean(mse, axis=0))

for res in res_list :
    print(res)

for pair in zip(res_list[0], res_list[1]):
    print(pair, end=",")

