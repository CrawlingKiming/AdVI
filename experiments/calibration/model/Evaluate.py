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
from plotting import SEIR_plotting
from plotting import Experiment_Writing

import os

##############################
#Handling Model Reading Error#
##############################
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def HPD(args, samples, model_num=None, log_path = None):
    hpd_l = []
    hpd_r = []
    modes_mu = []
    mse = []
    coverage = []
    if args.dataset == "SIR":
        samples = samples[-1]
        temp = samples.cpu().detach().numpy()

        SIR_writer = SIR_plotting(log_path=log_path)
        hpd_l, hpd_r, modes_mu, mse, coverage = SIR_writer.param_img_store(params=temp, model_num=model_num)

    if args.dataset == "SEIR":
        samples = samples[-1]

        temp = samples.cpu().detach().numpy()
        #np.save("./assets/BTAT", temp)
        SEIR_writer = SEIR_plotting(log_path=log_path)
        SEIR_writer.param_ls_img_store(params=[temp])
        temp = temp * 1e6
        for j in range(5):
            temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
            #tm = np.sum(np.square(temp[:, j] - true[j])) / temp.shape[0]
            a, b = temp_hpd[0]
            c = temp_modes[0]
            hpd_l.append(a/1e6)
            hpd_r.append(b/1e6)
            modes_mu.append(c/1e6)

    if (args.dataset == "MSIR") or (args.dataset == "MSIR_full"):
        samples = samples[-1]
        temp = samples.cpu().detach().numpy()
        true = [0.43, 7.35, 0.027, 0.9, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]

        true_beta = [1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]
        MSIR_writer = Experiment_Writing(log_path=log_path)
        MSIR_writer.param_img_store(params=temp, model_num=model_num)

        temp = temp * 1e2

        for j in range(10):
            param_cov = 0

            try :
                temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
                tm = np.mean(np.square(np.mean(temp[:, j]) - true[j]*1e2))
                a, b = temp_hpd[0]
                c = temp_modes[0]
            except Exception:
                p = temp[0, j]
                a = p
                b = p
                c = p
                tm = 0

            hpd_l.append(a/1e2)
            hpd_r.append(b/1e2)
            modes_mu.append(c/1e2)
            mse.append(tm/1e4)
            print("MSE : {}".format(tm/1e4))
            if a/1e2 < true[j] and b/1e2 > true[j]:
                param_cov +=1

            coverage.append(param_cov)

    return hpd_l, hpd_r, modes_mu, mse, coverage

def q_return(samples):
    samples_mean = np.mean(samples, axis=0)
    samples_min = np.percentile(samples, 2.5, axis=0)
    samples_max = np.percentile(samples, 97.5, axis=0)

    return samples_min, samples_mean, samples_max

def for_samples(args, samples, truex, model_num=1, log_path=None):

    if args.dataset == "SIR":
        res = []

        SIR_writer = SIR_plotting(log_path=log_path)

        groundtruth = np.load("../data/SIR/SIR.npy")
        cover_perc = SIR_writer.forward_img_store(samples=samples, truex=truex,
                                     groundtruth=groundtruth, model_num=model_num)
        #print(cover_perc)
        res.append(cover_perc)
        return res

    if args.dataset == "SEIR":
        age_result = []
        samples_Iw, samples_age_prop = samples
        #samples_age_prop = samples_age_prop.cpu().detach().numpy()
        age_perc = np.asarray([[42, 30, 28]])
        SEIR_writer = SEIR_plotting(log_path=log_path)

        SEIR_writer.forward_img_store(samples, truex=truex[:, 2].detach().cpu().numpy(), extra=model_num)
        #print(samples_age_prop.shape)
        print(np.mean(samples_age_prop, axis=0)*100)
        res = np.mean(np.square((np.mean(samples_age_prop, axis=0)*100-age_perc)))
        age_result.append(res)
        return res

    if (args.dataset == "MSIR") or (args.dataset == "MSIR_full"):
        #glambd = np.load("../data/MSIR_lambda.npy")
        #glambd = glambd[0] # 118 * 6
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

setting_choices = {"BFAF", "BTAT"}

parser.add_argument('--setting', type=str, default="BTAT", choices=setting_choices)

args = parser.parse_args()

#Part 1 Read model, load model
log_path, filenames = read_model.read_file(args)
truex, data_shape = get_data(args)
loss_fn = get_loss(args)
mycan, model = get_model(args, data_shape=data_shape)
#print("Q")
#model.load_state_dict(checkpoint['state_dict'])
l = []
r = []
mu = []
mse = []
al = []
coverage = []
ii = 0

if args.dataset == "SIR":
    assert len(filenames) == 100
    args.eval_size = 2048

if args.dataset == "SEIR":
    args.smoothing = 1 # ONly for computation.
    args.smoothing2 = 1
    args.eval_size = 4096

if args.dataset == "MSIR":
    #print(len(filenames))
    assert len(filenames) == 50
    args.eval_size = 500



for f in filenames:
    #    ii +=1
    #    continue

    print(f)
    try :
        truex, data_shape = get_data(args, model_num=ii)
        ii += 1

        model_path = os.path.join(log_path, f)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict1'])

        samples = loss_fn(can_model=mycan, model=model, observation=truex, args=args, eval=True, recover=True, itr=100)
        res = for_samples(args,samples, truex=truex, model_num=ii-1, log_path=log_path)

        params_ls, _, _, _ = mycan(num_samples=5000, model=model)
        tl, tr, tmu, tmse, tcoverage = HPD(args, params_ls, model_num=ii-1, log_path=log_path)
        print(res)

    except NotImplementedError as e:
    #except Exception as e:
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
if args.dataset == "SEIR":
    res_list = [np.mean(x, axis=0) for x in (l, r, mu, mse, al, tc)]
else :
    res_list = [np.around(np.mean(x, axis=0), decimals=4) for x in (l, r, mu, al, tc)]
    res_list.append(np.mean(mse, axis=0))

for res in res_list :
    print(res)

for pair in zip(res_list[0], res_list[1]):
    print(pair, end="&")
